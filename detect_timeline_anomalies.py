# Detect temporal anomalies in the default scene timeline.
# The LLM returns correction proposals only; this script NEVER applies moves.
# Normally called by run_pipeline_blocks.py with explicit paths.
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from llm import LLM

TITLE = 'Pride and Prejudice'
DEFAULT_PATH = f'data/timelines/{TITLE}.auto.default.json'
OUT_PATH = f'data/timelines/{TITLE}.auto.anomalies.json'

DETECTION_PROMPT = """You are auditing the temporal order of scenes in the novel "{title}".

The scenes below are listed in TEXT ORDER (the order in which they are presented in the novel). Most novels are primarily chronological, so preserve text order by default. Your task is NOT to rewrite or return the full ordering. Report only confident temporal anomalies.

A temporal anomaly means that the PRIMARY actions and dialogue of an entire scene or a contiguous block of scenes actually occur at an earlier story time and therefore should be moved in the reconstructed timeline.

CRITICAL rules:
1. Do NOT move a current-time scene merely because characters remember, discuss, read about, or narrate past events inside it. The scene's time is when its primary actions/dialogue occur.
2. Do NOT move a scene merely because it mentions a future plan or document date.
3. A move may cover a contiguous block of scenes. Use the first and last scene_uid of the block.
4. Propose a move only when the supplied time anchors and descriptions support it with high or medium confidence. Otherwise put it in "uncertain".
5. "insert_after_scene_uid" identifies the scene after which the moved block belongs chronologically. Use null only if it belongs before every scene in the book.
6. Do not include ordinary chronological progression in moves.

Output strict JSON only:
{{
  "moves": [
    {{
      "start_scene_uid": "p000_s000",
      "end_scene_uid": "p000_s000",
      "insert_after_scene_uid": null,
      "confidence": "high" or "medium",
      "reason": "Specific temporal evidence"
    }}
  ],
  "uncertain": [
    {{
      "start_scene_uid": "p000_s000",
      "end_scene_uid": "p000_s000",
      "reason": "Why its placement cannot be resolved",
      "needed_context": "What original text is needed"
    }}
  ],
  "consistency_notes": ["Any non-move time-anchor conflict worth reviewing"]
}}

==Scenes in text order==
{scenes}
"""


def compact_scene(scene):
    time = scene.get('time_text') or 'null'
    evidence = scene.get('time_evidence') or 'null'
    description = ' '.join((scene.get('description') or '').split())
    participants = ', '.join(scene.get('participants', []))
    return (f"{scene['scene_uid']} | text_order={scene['text_order']} | "
            f"chapter={scene.get('chapter_title')} | time={time} | "
            f"evidence={evidence} | participants={participants} | "
            f"scene={description}")


def validate_detection(result, timeline):
    scenes = timeline['scenes']
    uid_to_order = {s['scene_uid']: s['text_order'] for s in scenes}
    errors = []
    ranges = []
    for i, move in enumerate(result.get('moves', [])):
        start = move.get('start_scene_uid')
        end = move.get('end_scene_uid')
        target = move.get('insert_after_scene_uid')
        if start not in uid_to_order or end not in uid_to_order:
            errors.append(f'move {i}: unknown start/end uid')
            continue
        a, b = uid_to_order[start], uid_to_order[end]
        if a > b:
            errors.append(f'move {i}: start follows end in text order')
        if target is not None and target not in uid_to_order:
            errors.append(f'move {i}: unknown insertion target {target}')
        if target is not None and a <= uid_to_order[target] <= b:
            errors.append(f'move {i}: insertion target lies inside moved block')
        ranges.append((a, b, i))
    ranges.sort()
    for (_, prev_end, prev_i), (cur_start, _, cur_i) in zip(ranges, ranges[1:]):
        if cur_start <= prev_end:
            errors.append(f'moves {prev_i} and {cur_i}: overlapping ranges')

    for i, item in enumerate(result.get('uncertain', [])):
        if (item.get('start_scene_uid') not in uid_to_order or
                item.get('end_scene_uid') not in uid_to_order):
            errors.append(f'uncertain {i}: unknown scene uid')
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--input', default=DEFAULT_PATH)
    ap.add_argument('--output', default=OUT_PATH)
    args = ap.parse_args()

    with open(args.input, encoding='utf-8') as f:
        timeline = json.load(f)
    scene_text = '\n'.join(compact_scene(s) for s in timeline['scenes'])
    llm = LLM(config_path='config.json', book=timeline.get('book'))
    result = llm.call_json(
        DETECTION_PROMPT.format(title=timeline['book'], scenes=scene_text),
        model=args.model,
        max_tokens=65536,
    )
    errors = validate_detection(result, timeline)
    report = {
        'book': timeline['book'],
        'model': args.model,
        'status': 'proposals_only_not_applied',
        'input_scene_count': len(timeline['scenes']),
        'validation_errors': errors,
        'moves': result.get('moves', []),
        'uncertain': result.get('uncertain', []),
        'consistency_notes': result.get('consistency_notes', []),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"moves={len(report['moves'])}, uncertain={len(report['uncertain'])}, "
          f"validation_errors={len(errors)}")
    print(f'saved: {args.output}')


if __name__ == '__main__':
    main()
