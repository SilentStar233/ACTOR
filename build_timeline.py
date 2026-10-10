# Build the default scene timeline from v2 trajectory files.
# Pure code: text order is the initial timeline order; no LLM and no reordering.
# Normally called by run_pipeline_blocks.py with explicit paths.
import argparse
import json
import os

TITLE = 'Pride and Prejudice'
DEFAULT_TRAJ_DIR = f'data/trajectories_auto/{TITLE}'
DEFAULT_OUT_PATH = f'data/timelines/{TITLE}.auto.default.json'


def plot_id_from_name(name):
    return int(name.split('_')[1].split('.')[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', default=TITLE)
    ap.add_argument('--traj-dir', default=DEFAULT_TRAJ_DIR)
    ap.add_argument('--output', default=DEFAULT_OUT_PATH)
    args = ap.parse_args()

    files = sorted((f for f in os.listdir(args.traj_dir)
                    if f.startswith('plot_') and f.endswith('.json')),
                   key=plot_id_from_name)
    scenes = []
    for filename in files:
        path = os.path.join(args.traj_dir, filename)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        plot_id = data['plot_id']
        for plot in data.get('plots', []):
            for scene in sorted(plot.get('scenes', []),
                                key=lambda s: s.get('scene_id', 0)):
                scene_id = scene.get('scene_id', 0)
                text_order = len(scenes)
                scenes.append({
                    'scene_uid': f'p{plot_id:03d}_s{scene_id:03d}',
                    'plot_id': plot_id,
                    'scene_id': scene_id,
                    'chapter_title': data.get('chapter_title'),
                    'text_order': text_order,
                    'timeline_order': text_order,
                    'time_text': scene.get('time_text'),
                    'time_evidence': scene.get('time_evidence'),
                    'description': scene.get('description'),
                    'participants': scene.get('participants', []),
                })

    uids = [s['scene_uid'] for s in scenes]
    if len(uids) != len(set(uids)):
        raise ValueError('duplicate scene_uid found')
    if [s['text_order'] for s in scenes] != list(range(len(scenes))):
        raise ValueError('text_order is not contiguous')

    out = {
        'book': args.title,
        'status': 'default_text_order',
        'description': ('Default timeline before temporal anomaly correction. '
                        'timeline_order initially equals text_order.'),
        'scene_count': len(scenes),
        'scenes': scenes,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'{len(scenes)} scenes -> {args.output}')


if __name__ == '__main__':
    main()
