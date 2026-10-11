# Independently review timeline move proposals and produce auto-applicable moves.
# Policy: exact accepts with medium/high confidence are applied; modifications are
# applied only at high confidence; low-confidence decisions remain unresolved.
# Normally called by actor.trajectory_extraction.pipeline with explicit paths.
from actor.shared.paths import ROOT_DIR

import argparse
import json
import os
import sys

_HERE = ROOT_DIR

from actor.shared.llm import LLM
from actor.trajectory_extraction.detect_timeline_anomalies import validate_detection


REVIEW_PROMPT = """You are the independent reviewer of proposed temporal moves for scenes in the novel "{title}".

For each proposal, independently reconstruct the chronology from the supplied SOURCE PLOT TEXT, then decide whether the proposed move truly corrects the order of the PRIMARY actions/dialogue. Do not trust the proposal merely because it was produced by another model.

Rules:
1. Reject a move based only on remembered, discussed, planned, or document-dated events.
2. Check both the moved block and its proposed insertion neighborhood.
3. Preserve a contiguous companion scene when it clearly belongs to the same episode, even if it has no time anchor.
4. Return:
   - "accept": proposal boundaries and target are exactly correct;
   - "reject": current text order is already chronological or evidence is insufficient;
   - "modify": a move is needed but boundaries/target must change.
5. Use high confidence only for explicit temporal evidence; medium for strong narrative inference; low if source text is needed.

Strict JSON only:
{{
  "reviews": [
    {{
      "proposal_id": 0,
      "verdict": "accept" or "reject" or "modify",
      "start_scene_uid": "..." or null,
      "end_scene_uid": "..." or null,
      "insert_after_scene_uid": "..." or null,
      "confidence": "high" or "medium" or "low",
      "reason": "specific chronological reasoning",
      "time_corrections": [
        {{
          "scene_uid": "...",
          "time_text": "corrected current-scene time" or null,
          "time_evidence": "verbatim source evidence" or null,
          "reason": "why the existing time annotation was wrong"
        }}
      ]
    }}
  ]
}}

If the proposed move was caused by an incorrect/ambiguous time_text, reject the move and provide a time_correction grounded in source text. Otherwise return an empty time_corrections list.

==Proposals with local timeline neighborhoods==
{cases}
"""




def compact(scene):
    return {'scene_uid': scene['scene_uid'], 'text_order': scene['text_order'],
            'time_text': scene.get('time_text'),
            'description': scene.get('description')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeline', required=True)
    ap.add_argument('--proposals', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--plots-file', required=True,
                    help='source plot JSON used to provide full text for review')
    ap.add_argument('--model', default='gpt-5.6-luna')
    args = ap.parse_args()

    with open(args.timeline, encoding='utf-8') as f:
        timeline = json.load(f)
    with open(args.proposals, encoding='utf-8') as f:
        proposals = json.load(f)
    with open(args.plots_file, encoding='utf-8') as f:
        source_plots = {p['plot_id']: p.get('text', '')
                        for p in json.load(f)['plots']}
    scenes = timeline['scenes']
    uid_to_index = {s['scene_uid']: i for i, s in enumerate(scenes)}

    expanded_proposals = []
    for move in proposals.get('moves', []):
        expanded = dict(move)
        end_index = uid_to_index[expanded['end_scene_uid']]
        end_scene = scenes[end_index]
        # A moved episode carries immediately following same-plot scenes whose
        # time is unstated, until the next explicit anchor. Otherwise a companion
        # scene can be stranded at the later text position.
        while end_index + 1 < len(scenes):
            nxt = scenes[end_index + 1]
            if (nxt['plot_id'] == end_scene['plot_id'] and
                    nxt.get('time_text') is None):
                expanded['end_scene_uid'] = nxt['scene_uid']
                end_index += 1
                end_scene = nxt
            else:
                break
        expanded_proposals.append(expanded)

    cases = []
    for i, move in enumerate(expanded_proposals):
        start = uid_to_index[move['start_scene_uid']]
        end = uid_to_index[move['end_scene_uid']]
        target = move.get('insert_after_scene_uid')
        indexes = list(range(max(0, start - 2), min(len(scenes), end + 3)))
        if target is not None:
            ti = uid_to_index[target]
            indexes.extend(range(max(0, ti - 2), min(len(scenes), ti + 3)))
        local_scenes = [scenes[x] for x in sorted(set(indexes))]
        local = [compact(s) for s in local_scenes]
        plot_ids = sorted({s['plot_id'] for s in local_scenes})
        proposal_without_reason = {
            'start_scene_uid': move['start_scene_uid'],
            'end_scene_uid': move['end_scene_uid'],
            'insert_after_scene_uid': move.get('insert_after_scene_uid'),
        }
        cases.append({
            'proposal_id': i,
            'proposal': proposal_without_reason,
            'local_timeline': local,
            'source_plot_texts': [
                {'plot_id': pid, 'text': source_plots.get(pid, '')}
                for pid in plot_ids
            ],
        })

    if cases:
        llm = LLM(config_path=os.path.join(ROOT_DIR, 'config.json'), book=timeline.get('book'))
        result = llm.call_json(REVIEW_PROMPT.format(
            title=timeline['book'],
            cases=json.dumps(cases, ensure_ascii=False, indent=2)),
            model=args.model, max_tokens=16384)
        reviews = result.get('reviews', [])
    else:
        reviews = []

    by_id = {r['proposal_id']: r for r in reviews}
    accepted, rejected, unresolved = [], [], []
    for i, proposal in enumerate(expanded_proposals):
        review = by_id.get(i)
        if review is None:
            unresolved.append({'proposal': proposal, 'reason': 'missing review'})
            continue
        verdict = review.get('verdict')
        confidence = review.get('confidence')
        if verdict == 'accept' and confidence in ('high', 'medium'):
            accepted.append({**proposal, 'review_confidence': confidence,
                             'review_reason': review.get('reason')})
        elif verdict == 'modify' and confidence == 'high':
            accepted.append({
                'start_scene_uid': review['start_scene_uid'],
                'end_scene_uid': review['end_scene_uid'],
                'insert_after_scene_uid': review.get('insert_after_scene_uid'),
                'reason': review.get('reason'),
                'review_confidence': confidence,
                'modified_from_proposal': proposal,
            })
        elif verdict == 'reject':
            rejected.append({'proposal': proposal,
                             'review_confidence': confidence,
                             'reason': review.get('reason')})
        else:
            unresolved.append({'proposal': proposal, 'review': review,
                               'reason': 'low-confidence or unsupported modification'})

    time_corrections = []
    known_uids = set(uid_to_index)
    for review in reviews:
        for correction in review.get('time_corrections', []):
            if correction.get('scene_uid') in known_uids:
                time_corrections.append(correction)
            else:
                unresolved.append({'review': review,
                                   'reason': 'time correction has unknown scene_uid'})

    validation = validate_detection({'moves': accepted, 'uncertain': []}, timeline)
    if validation:
        unresolved.extend({'proposal': move, 'reason': error}
                          for move, error in zip(accepted, validation))
        accepted = []

    report = {
        'book': timeline['book'], 'review_model': args.model,
        'policy': ('accept exact medium/high; accept modifications high only; '
                   'otherwise unresolved'),
        'accepted_moves': accepted, 'rejected_moves': rejected,
        'time_corrections': time_corrections,
        'unresolved': unresolved, 'raw_reviews': reviews,
        'validation_errors': validation,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'accepted={len(accepted)}, rejected={len(rejected)}, '
          f'unresolved={len(unresolved)}')
    print(f'output: {args.output}')


if __name__ == '__main__':
    main()
