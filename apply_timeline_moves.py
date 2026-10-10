# Apply reviewed temporal block moves to a default timeline and assign final
# timeline_order. The original text_order and trajectory files are untouched.
# Normally called by run_pipeline_blocks.py with explicit paths.
import argparse
import json
import os

TITLE = 'Pride and Prejudice'
DEFAULT_INPUT = f'data/timelines/{TITLE}.auto.default.json'
DEFAULT_MOVES = f'data/timelines/{TITLE}.auto.reviewed_moves.json'
DEFAULT_OUTPUT = f'data/timelines/{TITLE}.auto.final.json'


def apply_move(scenes, move):
    uids = [s['scene_uid'] for s in scenes]
    start = uids.index(move['start_scene_uid'])
    end = uids.index(move['end_scene_uid'])
    if start > end:
        raise ValueError(f'invalid move range: {move}')
    block = scenes[start:end + 1]
    del scenes[start:end + 1]
    target = move.get('insert_after_scene_uid')
    if target is None:
        insert_at = 0
    else:
        remaining = [s['scene_uid'] for s in scenes]
        if target not in remaining:
            raise ValueError(f'missing/inside target: {target}')
        insert_at = remaining.index(target) + 1
    scenes[insert_at:insert_at] = block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=DEFAULT_INPUT)
    ap.add_argument('--moves', default=DEFAULT_MOVES)
    ap.add_argument('--output', default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    with open(args.input, encoding='utf-8') as f:
        timeline = json.load(f)
    with open(args.moves, encoding='utf-8') as f:
        review = json.load(f)

    scenes = list(timeline['scenes'])
    original_uids = [s['scene_uid'] for s in scenes]
    by_uid = {s['scene_uid']: s for s in scenes}
    for correction in review.get('time_corrections', []):
        uid = correction['scene_uid']
        if uid not in by_uid:
            raise ValueError(f'unknown time-correction scene: {uid}')
        by_uid[uid]['time_text'] = correction.get('time_text')
        by_uid[uid]['time_evidence'] = correction.get('time_evidence')
        by_uid[uid]['time_correction_reason'] = correction.get('reason')
    for move in review['accepted_moves']:
        apply_move(scenes, move)
    final_uids = [s['scene_uid'] for s in scenes]
    if len(final_uids) != len(set(final_uids)):
        raise ValueError('duplicate scene after moves')
    if sorted(final_uids) != sorted(original_uids):
        raise ValueError('scene set changed while applying moves')
    for index, scene in enumerate(scenes):
        scene['timeline_order'] = index

    out = {
        'book': timeline['book'],
        'status': 'cleaned_and_reviewed_timeline',
        'scene_count': len(scenes),
        'accepted_moves': review['accepted_moves'],
        'rejected_moves': review['rejected_moves'],
        'time_corrections': review.get('time_corrections', []),
        'scenes': scenes,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    changed = sum(s['text_order'] != s['timeline_order'] for s in scenes)
    print(f"scenes={len(scenes)}, moved-position scenes={changed}")
    print(f'output: {args.output}')


if __name__ == '__main__':
    main()
