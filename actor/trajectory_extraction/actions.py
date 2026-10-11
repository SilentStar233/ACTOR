"""Extract actions from aligned plots, with per-scene fallback and resume."""
from actor.shared.paths import ROOT_DIR

import argparse
import json
import os
import sys

_HERE = ROOT_DIR

from actor.shared.llm import LLM
from actor.trajectory_extraction.check_plot_overlaps import find_overlaps, locate_plot_spans
from actor.shared.utils import load_book
from actor.shared.utils import atomic_json_dump, safe_title
from actor.trajectory_extraction.helpers import (SINGLE_CALL_MAX_CHARS, build_key_characters,
                                call_extraction_json, check_quotes,
                                extract_per_scene, load_inputs)
from actor.trajectory_extraction.prompts import build_extraction_prompt


def flatten_sub_plots(data):
    """Normalize to a single 'plots' entry with scene_ids 0..n-1.

    The LLM occasionally returns several 'plots' entries for one plot_id
    (dense chapters split into threads). Every downstream consumer assumes
    one file = one plot with unique scene_ids: build_timeline would raise on
    duplicate scene_uid p<pid>_s<sid>, and scene-time relabel would silently
    apply one scene's fix to every same-id scene.
    """
    plots = data.get('plots') or []
    if len(plots) <= 1:
        return data
    scenes = []
    for pl in plots:
        scenes.extend(pl.get('scenes', []))
    for new_id, scene in enumerate(scenes):
        scene['scene_id'] = new_id
    print(f'  plot {data.get("plot_id")}: LLM returned {len(plots)} '
          f'sub-plots; flattened to {len(scenes)} scenes')
    data['plots'] = [{'plot_id': data.get('plot_id'),
                      'chapter_title': data.get('chapter_title'),
                      'scenes': scenes}]
    return data


def extract_one(llm, title, author, plot, plot_ctx):
    """Single-call for short plots, per-scene for long ones; fall back on failure."""
    prompt = build_extraction_prompt(
        title=title, author=author, text=plot['text'],
        chapter_title=plot.get('chapter_title'), plot_summary=plot.get('summary'),
        key_characters=build_key_characters(plot, plot_ctx), character_context=plot_ctx,
    )
    single_first = len(plot['text']) <= SINGLE_CALL_MAX_CHARS
    modes = ['single', 'per_scene'] if single_first else ['per_scene', 'single']
    last_err = None
    for mode in modes:
        try:
            if mode == 'single':
                data = call_extraction_json(
                    llm, prompt, llm.default_model, max_tokens=65536)
            else:
                data = extract_per_scene(llm, title, author, plot, plot_ctx)
            data['plot_id'] = plot['plot_id']
            data['chapter_title'] = plot.get('chapter_title')
            data['prominence'] = plot.get('prominence')
            data['summary'] = plot.get('summary')
            data['extraction_mode'] = mode
            data = flatten_sub_plots(data)
            return data, None
        except Exception as e:
            print(f'  plot {plot["plot_id"]}: {mode} failed ({str(e)[:80]}), '
                  f'trying the other mode')
            last_err = e
    return None, str(last_err)


def main():
    ap = argparse.ArgumentParser(description='full-book action extraction')
    ap.add_argument('--title', required=True)
    ap.add_argument('--plots-file', default=None)
    ap.add_argument('--contexts-dir', default=os.path.join(ROOT_DIR, 'data/contexts'))
    ap.add_argument('--output_dir', default=os.path.join(ROOT_DIR, 'data/trajectories_auto'))
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', type=int, default=None)
    ap.add_argument('--plots', type=str, default=None)
    ap.add_argument('--src', default=os.path.join(ROOT_DIR, 'data/src/all_books.jsonl'))
    ap.add_argument('--allow-overlap', action='store_true',
                    help='explicitly allow overlapping plot source intervals')
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be at least 1')

    plots_data, contexts = load_inputs(
        title=args.title, plots_file=args.plots_file,
        contexts_dir=args.contexts_dir)
    all_plots = list(plots_data['plots'])
    missing_text = [plot['plot_id'] for plot in all_plots if not plot.get('text')]
    if missing_text:
        raise SystemExit(
            f'Plots without aligned source text: {missing_text}. '
            'Run the plot source-alignment stage before action extraction.')
    if args.only is None and not args.plots:
        book = load_book(args.src, args.title)
        overlaps = find_overlaps(locate_plot_spans(book['content'], all_plots))
        if overlaps and not args.allow_overlap:
            pairs = [(o['left_plot_id'], o['right_plot_id']) for o in overlaps]
            raise SystemExit(
                f'Persisted plot spans overlap {pairs}. Stop before trajectory extraction; '
                'repair plot boundaries or pass --allow-overlap only for an explicit '
                'experiment followed by scene deduplication.')
    plots = list(all_plots)
    if args.only is not None:
        plots = [p for p in plots if p['plot_id'] == args.only]
    if args.plots:
        ids = {int(x) for x in args.plots.split(',')}
        plots = [p for p in plots if p['plot_id'] in ids]
    print(f'{len(plots)} plots to extract, model={args.model}')

    out_dir = os.path.join(args.output_dir, safe_title(args.title))
    os.makedirs(out_dir, exist_ok=True)
    llm = LLM(config_path=os.path.join(ROOT_DIR, 'config.json'), book=args.title)
    llm.default_model = args.model

    def out_path(p):
        return os.path.join(out_dir, f'plot_{p["plot_id"]}.json')

    def is_complete(p):
        """A plot file counts as done only if it parses, has steps, and
        satisfies the shape invariant: exactly one 'plots' entry with unique
        scene_ids. Legacy multi-sub-plot files fail this check and are
        re-extracted (LLM cache hit), so they self-heal on the next run."""
        path = out_path(p)
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return False
        plots = data.get('plots') or []
        if len(plots) != 1:
            return False
        scene_ids = [s.get('scene_id')
                     for pl in plots for s in pl.get('scenes', [])]
        if len(scene_ids) != len(set(scene_ids)):
            return False
        steps = sum(len(t.get('steps', []))
                    for pl in plots for s in pl.get('scenes', [])
                    for t in s.get('trajectories', []))
        return steps > 0

    todo = [p for p in plots if not is_complete(p)]
    print(f'{len(todo)} not yet extracted (skipping {len(plots) - len(todo)} existing)')

    from concurrent.futures import ThreadPoolExecutor, as_completed
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, llm, plots_data['book'], plots_data['author'],
                          p, contexts.get(str(p['plot_id']), {})): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            data, err = fut.result()
            if err:
                failures.append(p['plot_id'])
                print(f'  plot {p["plot_id"]}: FAILED - {err[:120]}')
                continue
            n_steps = sum(len(t.get('steps', []))
                          for pl in data.get('plots', []) for s in pl.get('scenes', [])
                          for t in s.get('trajectories', []))
            n_crit = sum(1 for pl in data.get('plots', []) for s in pl.get('scenes', [])
                         for t in s.get('trajectories', []) for st in t.get('steps', [])
                         if st.get('is_critical'))
            n_q, n_qhit = check_quotes(data, p['text'])
            data['quality'] = {'steps': n_steps, 'critical': n_crit,
                               'quotes_verbatim': f'{n_qhit}/{n_q}'}
            atomic_json_dump(out_path(p), data)
            print(f'  plot {p["plot_id"]}: {n_steps} steps, {n_crit} critical, '
                  f'quotes {n_qhit}/{n_q} [{data["extraction_mode"]}]')

    print(f'\nDone. extracted={len(todo) - len(failures)}, failed={len(failures)}')
    if failures:
        print('failed plot_ids:', failures)
        raise SystemExit('Trajectory extraction failed; refusing to continue with incomplete plots.')
    expected_ids = {plot['plot_id'] for plot in plots}
    output_ids = {
        int(filename.split('_')[1].split('.')[0])
        for filename in os.listdir(out_dir)
        if filename.startswith('plot_') and filename.endswith('.json')
    }
    missing_ids = expected_ids - output_ids
    if missing_ids:
        raise SystemExit(f'Missing trajectory outputs for plot_ids: {sorted(missing_ids)}')
    print(llm.cost_summary())


if __name__ == '__main__':
    main()
