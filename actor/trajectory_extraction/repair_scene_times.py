# Re-annotate scene time_text/time_evidence from full source plot text.
# With --apply, validated corrections are written back to trajectory files.
# The production pipeline invokes this script with --all --apply.
# Usage: cd repository-root && python -m actor.trajectory_extraction.repair_scene_times --all --apply --title "Book Title" \
#   --plots-file "data/plots_final/Book Title.plots.json" \
#   --traj-dir "data/trajectories_auto/Book Title" \
#   --output "data/logs/Book Title.scene_time_relabel.json"
from actor.shared.paths import ROOT_DIR

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = ROOT_DIR

from actor.shared.llm import LLM
from actor.shared.utils import atomic_json_dump


TIME_REPAIR_PROMPT = """You are correcting scene-level time annotations for the novel "{title}".

For every provided scene, record ONLY when the actions and dialogue IN THAT SCENE actually occur.

Output fields:
- "time_text": the actual time or complete time span of this scene, directly grounded in the plot text; null if the current scene's time is not established.
- "time_evidence": exact verbatim quote(s) supporting that time; null when time_text is null. For a span supported by separate passages, join excerpts with " ... ".
- "correction_reason": briefly explain why the old annotation was kept, corrected, or removed.

CRITICAL boundary:
- A future plan or appointment is not the current scene's time.
- A recalled past event is not the current scene's time.
- A date printed on a letter/document is not automatically the time when it is read.
- If a scene spans multiple times, put the complete span in the single time_text string.
- Do not infer from knowledge outside the supplied plot text.

Examples:
- "The next morning she was sitting by herself." -> "the next morning".
- Current actions run from Saturday morning through "On Sunday, after morning service" -> "from Saturday morning to Sunday after morning service".
- "The ball was fixed for the following Tuesday" in the invitation scene -> null.
- A character says "They will join me early to-morrow" -> null unless separate narration establishes the current scene's time.
- "The letter was dated Monday, August 2" -> null unless the reading scene is separately established to occur then.

Return exactly one result for every supplied scene_id. Strict JSON only:
{{
  "scenes": [
    {{
      "scene_id": 0,
      "time_text": "..." or null,
      "time_evidence": "..." or null,
      "correction_reason": "..."
    }}
  ]
}}

==Scenes to annotate==
{scenes}

==Full plot text==
{plot_text}
"""


def load_plot_inputs(plot_id, plots_file, traj_dir):
    with open(plots_file, encoding='utf-8') as f:
        source = json.load(f)
    plot = next(p for p in source['plots'] if p['plot_id'] == plot_id)
    traj_path = os.path.join(traj_dir, f'plot_{plot_id}.json')
    with open(traj_path, encoding='utf-8') as f:
        traj = json.load(f)
    scenes = []
    for pl in traj.get('plots', []):
        for s in pl.get('scenes', []):
            scenes.append({
                'scene_id': s.get('scene_id'),
                'description': s.get('description'),
                'participants': s.get('participants', []),
                'old_time_text': s.get('time_text'),
                'old_time_evidence': s.get('time_evidence'),
            })
    ids = [s['scene_id'] for s in scenes]
    if len(ids) != len(set(ids)):
        # Multi-sub-plot files make fixes collide silently (one scene's fix
        # applied to every same-id scene). Fail fast; re-run the extract
        # stage, which now flattens sub-plots and renumbers scene_ids.
        raise ValueError(f'{traj_path}: duplicate scene_id in {ids}')
    return plot, scenes


def repair_one(llm, model, title, plots_file, traj_dir, plot_id):
    plot, scenes = load_plot_inputs(plot_id, plots_file, traj_dir)
    prompt = TIME_REPAIR_PROMPT.format(
        title=title,
        scenes=json.dumps(scenes, ensure_ascii=False, indent=2),
        plot_text=plot['text'],
    )
    result = llm.call_json(prompt, model=model, max_tokens=65536)
    fixed = {s['scene_id']: s for s in result.get('scenes', [])}
    comparisons = []
    for old in scenes:
        new = fixed.get(old['scene_id'])
        comparisons.append({
            'scene_id': old['scene_id'],
            'description': old['description'],
            'old_time_text': old['old_time_text'],
            'old_time_evidence': old['old_time_evidence'],
            'new_time_text': new.get('time_text') if new else None,
            'new_time_evidence': new.get('time_evidence') if new else None,
            'correction_reason': new.get('correction_reason') if new else 'MISSING RESULT',
        })
    return {'plot_id': plot_id, 'chapter_title': plot.get('chapter_title'),
            'comparisons': comparisons}


def apply_results(results, traj_dir):
    """Write corrected time fields back only for plots with complete scene results."""
    applied = []
    skipped = []
    for r in results:
        pid = r['plot_id']
        comparisons = r.get('comparisons')
        if not comparisons or any(c['correction_reason'] == 'MISSING RESULT'
                                  for c in comparisons):
            skipped.append(pid)
            continue
        path = os.path.join(traj_dir, f'plot_{pid}.json')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        fixes = {c['scene_id']: c for c in comparisons}
        seen = set()
        for pl in data.get('plots', []):
            for scene in pl.get('scenes', []):
                sid = scene.get('scene_id')
                if sid in fixes:
                    scene['time_text'] = fixes[sid]['new_time_text']
                    scene['time_evidence'] = fixes[sid]['new_time_evidence']
                    seen.add(sid)
        if seen != set(fixes):
            skipped.append(pid)
            continue
        atomic_json_dump(path, data)
        applied.append(pid)
    return applied, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plot_ids', nargs='*', type=int, default=[])
    ap.add_argument('--title', required=True)
    ap.add_argument('--plots-file', required=True)
    ap.add_argument('--traj-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--all', action='store_true', help='relabel every plot in --plots-file')
    ap.add_argument('--apply', action='store_true',
                    help='write corrected time fields back after complete validation')
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be at least 1')
    with open(args.plots_file, encoding='utf-8') as f:
        all_plot_ids = [p['plot_id'] for p in json.load(f)['plots']]
    plot_ids = all_plot_ids if args.all or not args.plot_ids else args.plot_ids
    out_path = args.output

    llm = LLM(config_path=os.path.join(ROOT_DIR, 'config.json'), book=args.title)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(repair_one, llm, args.model, args.title,
                      args.plots_file, args.traj_dir, pid): pid
            for pid in plot_ids
        }
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f'plot {pid}: {len(r["comparisons"])} scenes')
                for c in r['comparisons']:
                    if c['old_time_text'] != c['new_time_text']:
                        print(f'  scene {c["scene_id"]}: {c["old_time_text"]!r} -> '
                              f'{c["new_time_text"]!r}')
            except Exception as e:
                print(f'plot {pid}: FAILED - {str(e)[:200]}')
                results.append({'plot_id': pid, 'error': str(e)})

    results.sort(key=lambda x: x['plot_id'])
    applied = skipped = []
    if args.apply:
        applied, skipped = apply_results(results, args.traj_dir)
        print(f'applied={len(applied)}, skipped={skipped}')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report = {'book': args.title, 'model': args.model, 'requested_plots': plot_ids,
              'applied_plots': applied, 'skipped_plots': skipped,
              'plots': results}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'saved: {out_path}')
    failed_plots = [item['plot_id'] for item in results if item.get('error')]
    if args.all and args.apply and (failed_plots or skipped):
        raise SystemExit(
            'Scene-time relabel failed to apply all plots: '
            f'failed={failed_plots}, skipped={skipped}')


if __name__ == '__main__':
    main()
