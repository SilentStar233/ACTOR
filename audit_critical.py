# Positive-only audit of is_critical labels across the whole book.
# First locates later trajectory evidence, then verifies each candidate against
# that evidence. Writes audit verdict and cited evidence ids back to trajectories.
# Normally called by run_pipeline_blocks.py with explicit paths.
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from extract_helpers_v2 import call_extraction_json
from llm import LLM
from utils import atomic_json_dump

TITLE = 'Pride and Prejudice'
TRAJ_DIR = f'data/trajectories_auto/{TITLE}'
REPORT = f'data/logs/{TITLE}/{TITLE}.audit_critical.json'

LOCATOR_PROMPT = """You are locating textual downstream evidence for candidate critical actions in the novel "{title}".

A candidate can be truly critical only if a later scene or plot contains a concrete event whose occurrence, relationship, decision, or consequence would plausibly differ if the candidate action were replaced by a different plausible choice.

For every case, use ONLY the supplied event ledger. Identify up to 3 specific LATER scene_uids that contain the proposed downstream consequence. Do not select the candidate's own scene, an earlier scene, or a scene that merely repeats the candidate's mood or wording. If no later evidence is identifiable, return an empty list. Do not use outside knowledge of the novel.

Output strict JSON only:
{{
  "locations": [
    {{"id": 0, "supporting_scene_uids": ["p000_s000"], "why": "brief evidence link"}}
  ]
}}

==Candidate cases==
{cases}

==Event ledger in source-text order==
{ledger}
"""


PLOT_LOCATOR_PROMPT = """You are locating plots that may contain downstream evidence for candidate critical actions in the novel "{title}".

A candidate can be truly critical only if a later scene or plot contains a concrete event whose occurrence, relationship, decision, or consequence would plausibly differ if the candidate action were replaced by a different plausible choice.

For each case, use ONLY the supplied plot index. Identify up to 3 LATER plots that may contain the proposed downstream consequence. Only plots listed as candidates are allowed. If none may contain the consequence, return an empty list. Do not use outside knowledge of the novel.

Output strict JSON only:
{{
  "locations": [
    {{"id": 0, "candidate_plot_ids": [35], "why": "brief evidence link"}}
  ]
}}

==Candidate cases==
{cases}

==Plot index (only plots that may contain later scenes)==
{plot_index}
"""


SCENE_LOCATOR_PROMPT = """You are locating specific downstream evidence scenes for one candidate critical action in the novel "{title}".

The candidate can be truly critical only if a later scene contains a concrete event whose occurrence, relationship, decision, or consequence would plausibly differ if the candidate action were replaced by a different plausible choice.

Use ONLY the supplied scene cards from pre-filtered plots. Identify up to 3 scene_uids that contain the proposed downstream consequence. Do not select scenes that merely repeat the candidate's mood or wording. If none contains the consequence, return an empty list. Do not use outside knowledge of the novel.

Output strict JSON only:
{{
  "supporting_scene_uids": ["p035_s001"],
  "why": "brief evidence link"
}}

==Candidate case==
character: {character}
action: {action}
stated reason: {reason}

==Scene cards from candidate plots (all later than the candidate)==
{scene_cards}
"""


VERIFIER_PROMPT = """You are independently auditing criticality labels of character actions extracted from the novel "{title}".

A candidate is truly CRITICAL only if replacing it with a different plausible choice would change a concrete event in a LATER scene or plot. Scene importance, mood, wit, or a more vivid current interaction do not count.

For each case, evaluate the candidate action against its supplied later trajectory evidence. Use ONLY that evidence; do not use outside knowledge of the novel. Return "keep" only when the evidence establishes a concrete downstream consequence and the causal connection is plausible. A kept case MUST cite at least one supplied evidence_scene_uid and at least one action_uid from those supplied scenes. If the evidence only confirms the current scene, is too vague, or does not support the stated consequence, return "downgrade" with empty evidence lists.

Output strict JSON only:
{{
  "verdicts": [
    {{
      "id": 0,
      "verdict": "keep" or "downgrade",
      "why": "one concise evidence-grounded sentence",
      "evidence_scene_uids": ["p000_s000"],
      "evidence_action_uids": ["p000_s000_t000_a000"]
    }}
  ]
}}

==Cases with located later evidence==
{cases}
"""


def plot_id_from_name(name):
    return int(name.split('_')[1].split('.')[0])


def scene_uid(plot_id, scene_id):
    return f'p{plot_id:03d}_s{scene_id:03d}'


def load_book_index(traj_dir):
    """Build source-text ordered scene cards and positive critical candidates."""
    files = sorted((name for name in os.listdir(traj_dir)
                    if name.startswith('plot_') and name.endswith('.json')),
                   key=plot_id_from_name)
    scene_index = {}
    cases = []
    text_order = 0
    for filename in files:
        path = os.path.join(traj_dir, filename)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        plot_id = data['plot_id']
        for plot in data.get('plots', []):
            for scene in sorted(plot.get('scenes', []),
                                key=lambda item: item.get('scene_id', 0)):
                scene_id = scene.get('scene_id', 0)
                uid = scene_uid(plot_id, scene_id)
                actions = []
                for trajectory_index, trajectory in enumerate(scene.get('trajectories', [])):
                    character = trajectory.get('character', '')
                    for step in trajectory.get('steps', []):
                        step_id = step.get('step_id', 0)
                        action_uid = step.get(
                            'action_uid',
                            f'p{plot_id:03d}_s{scene_id:03d}_t{trajectory_index:03d}_a{step_id:03d}')
                        actions.append({
                            'action_uid': action_uid,
                            'character': character,
                            'action': step.get('action', ''),
                            'observation': step.get('observation'),
                            'thought_evidence': step.get('thought_evidence'),
                        })
                        if step.get('is_critical'):
                            cases.append({
                                'id': len(cases),
                                'file': filename,
                                'plot_id': plot_id,
                                'scene_id': scene_id,
                                'source_scene_uid': uid,
                                'source_text_order': text_order,
                                'character': character,
                                'step_id': step_id,
                                'action_uid': action_uid,
                                'action': step.get('action', ''),
                                'reason': step.get('criticality_reason', ''),
                            })
                scene_index[uid] = {
                    'scene_uid': uid,
                    'plot_id': plot_id,
                    'scene_id': scene_id,
                    'text_order': text_order,
                    'description': scene.get('description', ''),
                    'participants': scene.get('participants', []),
                    'actions': actions,
                }
                text_order += 1
    return cases, scene_index


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def compact_action(action, include_evidence=False):
    text = f"[{action['action_uid']}] {action['character']}: {action['action'][:260]}"
    if include_evidence:
        observation = action.get('observation')
        thought_evidence = action.get('thought_evidence')
        if observation:
            text += f" | observation: {observation[:220]}"
        if thought_evidence:
            text += f" | text evidence: {thought_evidence[:220]}"
    return text


def compact_scene(scene, max_actions=3, include_evidence=False):
    participants = ', '.join(scene['participants']) or 'unknown'
    lines = [
        f"{scene['scene_uid']} | text_order={scene['text_order']} | "
        f"participants={participants}",
        f"scene: {scene['description'][:360]}",
    ]
    for action in scene['actions'][:max_actions]:
        lines.append(compact_action(action, include_evidence=include_evidence))
    if len(scene['actions']) > max_actions:
        lines.append(f"... {len(scene['actions']) - max_actions} additional actions omitted")
    return '\n'.join(lines)


def build_locator_ledger(scene_index, max_actions):
    return '\n\n'.join(
        compact_scene(scene, max_actions=max_actions)
        for scene in sorted(scene_index.values(), key=lambda item: item['text_order']))


def load_plot_summaries(plots_file):
    """Read plot summaries/characters from the final plots file, if supplied."""
    if not plots_file or not os.path.exists(plots_file):
        return {}
    with open(plots_file, encoding='utf-8') as f:
        data = json.load(f)
    summaries = {}
    for plot in data.get('plots', []):
        names = list(plot.get('key_character_names') or [])
        for character in plot.get('key_characters') or []:
            if isinstance(character, dict) and character.get('name'):
                names.append(character['name'])
            elif isinstance(character, str):
                names.append(character)
        summaries[plot['plot_id']] = {
            'summary': plot.get('summary', ''),
            'characters': list(dict.fromkeys(names)),
        }
    return summaries


def build_plot_index(scene_index, plot_summaries):
    """Aggregate scene cards into plot-level cards for coarse pre-filtering."""
    by_plot = {}
    for scene in sorted(scene_index.values(), key=lambda item: item['text_order']):
        entry = by_plot.setdefault(scene['plot_id'], {
            'plot_id': scene['plot_id'],
            'first_text_order': scene['text_order'],
            'last_text_order': scene['text_order'],
            'participants': [],
            'scene_uids': [],
        })
        entry['last_text_order'] = max(entry['last_text_order'], scene['text_order'])
        entry['scene_uids'].append(scene['scene_uid'])
        for participant in scene.get('participants', []):
            if participant not in entry['participants']:
                entry['participants'].append(participant)
    index = []
    for plot_id in sorted(by_plot):
        entry = by_plot[plot_id]
        info = plot_summaries.get(plot_id, {})
        index.append({
            'plot_id': plot_id,
            'first_text_order': entry['first_text_order'],
            'last_text_order': entry['last_text_order'],
            'summary': info.get('summary', ''),
            'characters': info.get('characters') or entry['participants'],
        })
    return index


def candidate_plots_for_case(plot_index, case):
    """Plots that may contain a scene later than the case's source scene."""
    return [plot for plot in plot_index
            if plot['last_text_order'] > case['source_text_order']]


def format_plot_index(plots):
    lines = []
    for plot in plots:
        characters = ', '.join(plot['characters'][:8]) or 'unknown'
        lines.append(
            f"plot_id={plot['plot_id']} | scenes {plot['first_text_order']}.."
            f"{plot['last_text_order']} | characters={characters}\n"
            f"summary: {plot['summary'][:400]}")
    return '\n\n'.join(lines)


def locate_plots_for_case(llm, model, title, case, plot_index, batch, batch_index):
    candidate_plots = candidate_plots_for_case(plot_index, case)
    return candidate_plots


def run_plot_level_locator(llm, model, title, cases, plot_index, batch_size):
    """Step 1: coarse filter — which plots may contain each case's evidence."""
    plot_candidates = {case['id']: candidate_plots_for_case(plot_index, case)
                       for case in cases}
    located = {}
    errors = []
    for chunk in chunks(cases, batch_size):
        # The plot index only includes plots that could contain later evidence
        # for at least one case in this batch.
        relevant_ids = set()
        for case in chunk:
            relevant_ids.update(plot['plot_id'] for plot in plot_candidates[case['id']])
        relevant = [plot for plot in plot_index if plot['plot_id'] in relevant_ids]
        prompt = PLOT_LOCATOR_PROMPT.format(
            title=title,
            cases=format_locator_cases(chunk),
            plot_index=format_plot_index(relevant),
        )
        result = call_extraction_json(llm, prompt, model, max_tokens=65536)
        valid_ids = {plot['plot_id'] for plot in relevant}
        case_ids = {case['id'] for case in chunk}
        for location in result.get('locations', []):
            case_id = location.get('id')
            if case_id not in case_ids:
                errors.append(f'plot locator returned unknown case id: {case_id}')
                continue
            raw_ids = location.get('candidate_plot_ids', [])
            if not isinstance(raw_ids, list):
                errors.append(f'case {case_id}: candidate_plot_ids not a list')
                continue
            allowed = {plot['plot_id'] for plot in plot_candidates[case_id]}
            valid = [pid for pid in raw_ids[:3]
                     if pid in valid_ids and pid in allowed]
            if len(valid) < len([pid for pid in raw_ids[:3] if pid in valid_ids]):
                errors.append(f'case {case_id}: plot not later than source')
            located[case_id] = {
                'candidate_plot_ids': valid,
                'why': location.get('why', ''),
            }
    return located, errors


def run_scene_level_locator(llm, model, title, case, plot_ids, scene_index):
    """Step 2: precise location within pre-filtered plots for one case."""
    scenes = [scene for scene in scene_index.values()
              if scene['plot_id'] in plot_ids
              and scene['text_order'] > case['source_text_order']]
    scenes.sort(key=lambda item: item['text_order'])
    if not scenes:
        return {'supporting_scene_uids': [], 'why': 'no later scene in candidate plots'}
    cards = '\n\n'.join(compact_scene(scene, max_actions=3) for scene in scenes)
    prompt = SCENE_LOCATOR_PROMPT.format(
        title=title,
        character=case['character'],
        action=case['action'][:500],
        reason=case['reason'][:500],
        scene_cards=cards,
    )
    result = call_extraction_json(llm, prompt, model, max_tokens=65536)
    raw_uids = result.get('supporting_scene_uids', [])
    allowed = {scene['scene_uid'] for scene in scenes}
    valid_uids = []
    errors = []
    if not isinstance(raw_uids, list):
        errors.append(f'case {case["id"]}: supporting_scene_uids not a list')
        raw_uids = []
    for uid in raw_uids[:3]:
        if uid not in allowed:
            errors.append(f'case {case["id"]}: invalid evidence scene {uid}')
        elif uid not in valid_uids:
            valid_uids.append(uid)
    return {
        'supporting_scene_uids': valid_uids,
        'why': result.get('why', ''),
        'errors': errors,
    }


def run_hierarchical_locator(llm, model, title, cases, scene_index,
                             plot_index, locator_batch, workers):
    """Two-level locator: plot-level pre-filter then scene-level precision."""
    plot_locations, plot_errors = run_plot_level_locator(
        llm, model, title, cases, plot_index, locator_batch)
    located = {}
    scene_errors = []

    def locate_one(case):
        plot_info = plot_locations.get(case['id'],
                                       {'candidate_plot_ids': [], 'why': ''})
        if not plot_info['candidate_plot_ids']:
            return case['id'], {
                'supporting_scene_uids': [],
                'why': plot_info['why'] or 'no candidate plot found',
                'errors': [],
            }
        result = run_scene_level_locator(
            llm, model, title, case,
            set(plot_info['candidate_plot_ids']), scene_index)
        result['plot_locator_why'] = plot_info['why']
        return case['id'], result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(locate_one, case): case for case in cases}
        for future in as_completed(futures):
            case_id, result = future.result()
            scene_errors.extend(result.pop('errors', []))
            located[case_id] = result
            print(f'  located evidence for case {case_id}')

    return located, plot_errors + scene_errors


def format_locator_cases(cases):
    return '\n\n'.join(
        f"[{case['id']}] source_scene={case['source_scene_uid']} | "
        f"source_text_order={case['source_text_order']}\n"
        f"character: {case['character']}\n"
        f"action: {case['action'][:500]}\n"
        f"stated reason: {case['reason'][:500]}"
        for case in cases)


def validate_locations(result, cases, scene_index):
    case_by_id = {case['id']: case for case in cases}
    located = {case['id']: {'supporting_scene_uids': [], 'why': ''}
               for case in cases}
    errors = []
    for location in result.get('locations', []):
        case_id = location.get('id')
        if case_id not in case_by_id:
            errors.append(f'locator returned unknown case id: {case_id}')
            continue
        raw_uids = location.get('supporting_scene_uids', [])
        if not isinstance(raw_uids, list):
            errors.append(f'case {case_id}: supporting_scene_uids is not a list')
            continue
        valid_uids = []
        for uid in raw_uids[:3]:
            scene = scene_index.get(uid)
            if scene is None:
                errors.append(f'case {case_id}: unknown evidence scene {uid}')
            elif scene['text_order'] <= case_by_id[case_id]['source_text_order']:
                errors.append(f'case {case_id}: non-later evidence scene {uid}')
            elif uid not in valid_uids:
                valid_uids.append(uid)
        located[case_id] = {
            'supporting_scene_uids': valid_uids,
            'why': location.get('why', ''),
        }
    return located, errors


def build_verifier_cases(cases, locations, scene_index, max_actions):
    records = []
    for case in cases:
        location = locations[case['id']]
        evidence_scenes = [scene_index[uid]
                           for uid in location['supporting_scene_uids']]
        records.append({
            'id': case['id'],
            'case': case,
            'located_scene_uids': location['supporting_scene_uids'],
            'locator_why': location['why'],
            'evidence_scenes': evidence_scenes,
            'provided_action_uids': {
                action['action_uid']
                for scene in evidence_scenes
                for action in scene['actions'][:max_actions]
            },
            'evidence_text': '\n\n'.join(
                compact_scene(scene, max_actions=max_actions, include_evidence=True)
                for scene in evidence_scenes),
        })
    return records


def format_verifier_cases(records):
    return '\n\n'.join(
        f"== Case {record['id']} ==\n"
        f"candidate action [{record['case']['action_uid']}]: "
        f"{record['case']['character']}: {record['case']['action'][:500]}\n"
        f"stated reason: {record['case']['reason'][:500]}\n"
        f"locator rationale: {record['locator_why'][:400]}\n"
        f"allowed evidence_scene_uids: {record['located_scene_uids']}\n"
        f"== Later trajectory evidence ==\n{record['evidence_text']}"
        for record in records)


def validate_verdicts(result, records):
    record_by_id = {record['id']: record for record in records}
    verdicts = {}
    errors = []
    for verdict in result.get('verdicts', []):
        case_id = verdict.get('id')
        if case_id not in record_by_id:
            errors.append(f'verifier returned unknown case id: {case_id}')
            continue
        label = verdict.get('verdict')
        if label not in {'keep', 'downgrade'}:
            errors.append(f'case {case_id}: invalid verdict {label!r}')
            continue
        record = record_by_id[case_id]
        allowed_scenes = set(record['located_scene_uids'])
        allowed_actions = record['provided_action_uids']
        evidence_scenes = verdict.get('evidence_scene_uids', [])
        evidence_actions = verdict.get('evidence_action_uids', [])
        if not isinstance(evidence_scenes, list) or not isinstance(evidence_actions, list):
            errors.append(f'case {case_id}: evidence fields must be lists')
            continue
        evidence_scenes = list(dict.fromkeys(evidence_scenes))
        evidence_actions = list(dict.fromkeys(evidence_actions))
        if label == 'keep' and (not evidence_scenes or not evidence_actions):
            errors.append(f'case {case_id}: keep has no cited evidence')
            continue
        if not set(evidence_scenes).issubset(allowed_scenes):
            errors.append(f'case {case_id}: cites unlocated evidence scene')
            continue
        if not set(evidence_actions).issubset(allowed_actions):
            errors.append(f'case {case_id}: cites unknown evidence action')
            continue
        if label == 'downgrade':
            evidence_scenes, evidence_actions = [], []
        verdicts[case_id] = {
            'verdict': label,
            'why': verdict.get('why', ''),
            'evidence_scene_uids': evidence_scenes,
            'evidence_action_uids': evidence_actions,
            'locator_why': record['locator_why'],
        }
    return verdicts, errors


def write_verdicts(traj_dir, cases, verdicts):
    by_file = {}
    for case in cases:
        verdict = verdicts[case['id']]
        by_file.setdefault(case['file'], []).append((case, verdict))
    for filename, items in by_file.items():
        path = os.path.join(traj_dir, filename)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        lookup = {
            (case['scene_id'], case['character'], case['step_id']): verdict
            for case, verdict in items
        }
        for plot in data.get('plots', []):
            for scene in plot.get('scenes', []):
                for trajectory in scene.get('trajectories', []):
                    for step in trajectory.get('steps', []):
                        key = (scene.get('scene_id'), trajectory.get('character'),
                               step.get('step_id'))
                        if key not in lookup:
                            continue
                        verdict = lookup[key]
                        step['audit_verdict'] = verdict['verdict']
                        step['audit_why'] = verdict['why']
                        step['audit_evidence_scene_uids'] = verdict[
                            'evidence_scene_uids']
                        step['audit_evidence_action_uids'] = verdict[
                            'evidence_action_uids']
                        step['audit_locator_why'] = verdict['locator_why']
        atomic_json_dump(path, data)


def main():
    ap = argparse.ArgumentParser(
        description='source-grounded positive criticality audit')
    ap.add_argument('--title', default=TITLE)
    ap.add_argument('--traj-dir', default=TRAJ_DIR)
    ap.add_argument('--report', default=REPORT)
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--locator-batch', type=int, default=25)
    ap.add_argument('--verify-batch', type=int, default=10)
    ap.add_argument('--locator-actions', type=int, default=3,
                    help='compact actions shown per scene to the locator')
    ap.add_argument('--evidence-actions', type=int, default=12,
                    help='actions shown per selected evidence scene to the verifier')
    ap.add_argument('--limit', type=int, default=None,
                    help='audit only the first N positive candidates')
    ap.add_argument('--dry-run', action='store_true',
                    help='write the report but leave trajectory JSON files unchanged')
    ap.add_argument('--plots-file', default=None,
                    help='final plots JSON supplying plot summaries for pre-filtering')
    ap.add_argument('--legacy-locator', action='store_true',
                    help='use the old whole-book event-ledger locator for comparison')
    ap.add_argument('--workers', type=int, default=4,
                    help='parallel workers for per-case scene-level location')
    args = ap.parse_args()
    if args.locator_batch < 1 or args.verify_batch < 1:
        raise ValueError('batch sizes must be positive')

    all_cases, scene_index = load_book_index(args.traj_dir)
    cases = all_cases[:args.limit] if args.limit is not None else all_cases
    print(f'{len(cases)}/{len(all_cases)} critical steps to audit, model={args.model}')
    llm = LLM(config_path='config.json', book=args.title)

    if args.legacy_locator:
        locator_mode = 'legacy_whole_book_ledger'
        ledger = build_locator_ledger(scene_index, args.locator_actions)
        print(f'event ledger: {len(scene_index)} scenes, {len(ledger)} chars')
        locations = {}
        locator_errors = []
        for chunk in chunks(cases, args.locator_batch):
            result = llm.call_json(
                LOCATOR_PROMPT.format(
                    title=args.title,
                    cases=format_locator_cases(chunk),
                    ledger=ledger,
                ),
                model=args.model, max_tokens=65536)
            chunk_locations, errors = validate_locations(result, chunk, scene_index)
            locations.update(chunk_locations)
            locator_errors.extend(errors)
            print(f'  located evidence for {chunk[0]["id"]}..{chunk[-1]["id"]}')
    else:
        locator_mode = 'hierarchical_plot_then_scene'
        plot_summaries = load_plot_summaries(args.plots_file)
        plot_index = build_plot_index(scene_index, plot_summaries)
        print(f'plot index: {len(plot_index)} plots '
              f'({len(plot_summaries)} with summaries)')
        locations, locator_errors = run_hierarchical_locator(
            llm, args.model, args.title, cases, scene_index, plot_index,
            args.locator_batch, args.workers)

    final_verdicts = {}
    records = []
    for case in cases:
        location = locations.get(case['id'], {'supporting_scene_uids': [], 'why': ''})
        if not location['supporting_scene_uids']:
            final_verdicts[case['id']] = {
                'verdict': 'downgrade',
                'why': 'No specific later-scene evidence was located for the claimed downstream consequence.',
                'evidence_scene_uids': [],
                'evidence_action_uids': [],
                'locator_why': location['why'],
            }
            continue
        records.extend(build_verifier_cases([case], locations, scene_index,
                                            args.evidence_actions))

    verifier_errors = []
    for chunk in chunks(records, args.verify_batch):
        result = llm.call_json(
            VERIFIER_PROMPT.format(
                title=args.title,
                cases=format_verifier_cases(chunk),
            ),
            model=args.model, max_tokens=65536)
        chunk_verdicts, errors = validate_verdicts(result, chunk)
        final_verdicts.update(chunk_verdicts)
        verifier_errors.extend(errors)
        print(f'  verified evidence for {chunk[0]["id"]}..{chunk[-1]["id"]}')

    for case in cases:
        if case['id'] not in final_verdicts:
            final_verdicts[case['id']] = {
                'verdict': 'unresolved',
                'why': 'The verifier did not return a valid evidence-grounded verdict.',
                'evidence_scene_uids': [],
                'evidence_action_uids': [],
                'locator_why': locations.get(case['id'], {}).get('why', ''),
            }

    if not args.dry_run:
        write_verdicts(args.traj_dir, cases, final_verdicts)

    counts = {
        label: sum(verdict['verdict'] == label
                   for verdict in final_verdicts.values())
        for label in ('keep', 'downgrade', 'unresolved')
    }
    report = {
        'book': args.title,
        'model': args.model,
        'method': 'two_stage_downstream_trajectory_evidence',
        'locator_mode': locator_mode,
        'dry_run': args.dry_run,
        'total_available': len(all_cases),
        'total_audited': len(cases),
        **counts,
        'locator_errors': locator_errors,
        'verifier_errors': verifier_errors,
        'verdicts': [
            {
                'id': case['id'],
                'action_uid': case['action_uid'],
                'source_scene_uid': case['source_scene_uid'],
                'plot_id': case['plot_id'],
                'character': case['character'],
                'action': case['action'],
                'stated_reason': case['reason'],
                **final_verdicts[case['id']],
            }
            for case in cases
        ],
    }
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'\nkeep={counts["keep"]} downgrade={counts["downgrade"]} '
          f'unresolved={counts["unresolved"]}')
    if locator_errors or verifier_errors:
        print(f'WARNING: locator_errors={len(locator_errors)} '
              f'verifier_errors={len(verifier_errors)}')
    print(f'report: {args.report}')
    print(llm.cost_summary())


if __name__ == '__main__':
    main()
