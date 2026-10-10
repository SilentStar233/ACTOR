# Stage B: extract T-A-O trajectories per plot, in parallel.
# Single-call first; on structural failure or suspicious empty result,
# fall back to two-pass mode (scene split -> per-scene extraction).
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import LLM
from prompts import build_extraction_prompt, SCENE_SPLIT_PROMPT
from assemble_plots import find_forward, canon


def load_inputs(title, plots_dir, contexts_dir):
    with open(os.path.join(plots_dir, f'{title}.plots.json'), encoding='utf-8') as f:
        plots_data = json.load(f)
    ctx_path = os.path.join(contexts_dir, title, 'character_context.json')
    with open(ctx_path, encoding='utf-8') as f:
        contexts = json.load(f)
    alias_path = os.path.join(contexts_dir, title, 'alias_map.json')
    with open(alias_path, encoding='utf-8') as f:
        alias_map = json.load(f)['alias_map']
    return plots_data, contexts, alias_map


def count_steps(data):
    return sum(len(t.get('steps', []))
               for pl in data.get('plots', [])
               for s in pl.get('scenes', [])
               for t in s.get('trajectories', []))


def is_valid_structure(data):
    """Valid: has 'plots' list; each plot has 'scenes' list."""
    plots = data.get('plots')
    if not isinstance(plots, list) or not plots:
        return False
    return all(isinstance(pl.get('scenes'), list) for pl in plots)


def is_suspicious_empty(data, plot):
    """Valid JSON but zero steps in a long, prominent plot: likely truncation."""
    if count_steps(data) > 0:
        return False
    has_scenes = any(pl.get('scenes') for pl in data.get('plots', []))
    return has_scenes and len(plot.get('text', '')) > 3000 and (plot.get('prominence') or 0) >= 50


def build_key_characters(plot, plot_ctx):
    if plot_ctx:
        return [{'name': char,
                 'experience': next((c.get('experience', '') for c in plot.get('key_characters', [])
                                     if c.get('name') == char), '')}
                for char in plot_ctx.keys()]
    return plot.get('key_characters')


def single_pass(llm, model, title, author, plot, plot_ctx):
    prompt = build_extraction_prompt(
        title=title, author=author, text=plot['text'],
        chapter_title=plot.get('chapter_title'),
        plot_summary=plot.get('summary'),
        key_characters=build_key_characters(plot, plot_ctx),
        character_context=plot_ctx,
    )
    return llm.call_json(prompt, model=model)


def slice_scene_texts(plot_text, scenes):
    """Slice scene texts by their first_sentence anchors (scenes are consecutive)."""
    content = canon(plot_text)
    anchors = []
    pos = 0
    for s in scenes:
        fs = canon(s.get('first_sentence', ''))
        span = find_forward(content, fs, pos, len(content)) if fs else None
        anchors.append(span[0] if span else None)
        if span:
            pos = span[0]
    texts = []
    for i, s in enumerate(scenes):
        start = anchors[i]
        if start is None:
            start = anchors[i - 1] if i > 0 and anchors[i - 1] is not None else 0
        end = next((a for a in anchors[i + 1:] if a is not None), len(plot_text))
        texts.append(plot_text[start:end].strip())
    return texts


def two_pass(llm, model, title, author, plot, plot_ctx):
    """Pass 1: split scenes. Pass 2: extract trajectories per scene."""
    kc = build_key_characters(plot, plot_ctx)
    kc_str = '\n'.join(f"- {c['name']}" for c in kc) if kc else '(unknown)'
    split_prompt = SCENE_SPLIT_PROMPT.format(
        title=title, summary=plot.get('summary', ''),
        key_characters=kc_str, text=plot['text'],
    )
    split = llm.call_json(split_prompt, model=model)
    scenes = split.get('scenes', [])
    if not scenes:
        return {'plots': [{'plot_id': 0, 'chapter_title': plot.get('chapter_title'),
                           'scenes': []}]}

    scene_texts = slice_scene_texts(plot['text'], scenes)
    merged_scenes = []
    for i, (scene, scene_text) in enumerate(zip(scenes, scene_texts)):
        participants = scene.get('participants', [])
        scene_ctx = {c: plot_ctx[c] for c in participants if c in plot_ctx}
        prompt = build_extraction_prompt(
            title=title, author=author, text=scene_text,
            chapter_title=plot.get('chapter_title'),
            plot_summary=plot.get('summary'),
            key_characters=[{'name': c, 'experience': ''} for c in participants],
            character_context=scene_ctx,
        )
        try:
            data = llm.call_json(prompt, model=model)
            inner = (data.get('plots') or [{}])[0]
            scene_out = inner.get('scenes', [])
            if scene_out:
                merged = scene_out[0]
                merged['scene_id'] = i
                merged.setdefault('description', scene.get('description'))
                merged.setdefault('participants', participants)
                merged.setdefault('relationship_context',
                                  scene.get('relationship_context', {}))
            else:
                merged = {'scene_id': i, 'description': scene.get('description'),
                          'participants': participants,
                          'relationship_context': scene.get('relationship_context', {}),
                          'trajectories': []}
        except Exception as e:
            print(f'    scene {i} of plot {plot["plot_id"]} failed: {str(e)[:80]}')
            merged = {'scene_id': i, 'description': scene.get('description'),
                      'participants': participants,
                      'relationship_context': scene.get('relationship_context', {}),
                      'trajectories': [], 'extraction_error': str(e)[:200]}

        # Per-character fallback: a big scene with participants but no trajectory
        # is likely truncated; extract each participant separately.
        if (not merged.get('trajectories') and len(participants) > 0
                and len(scene_text) > 2000):
            print(f'    scene {i} of plot {plot["plot_id"]}: empty, '
                  f'falling back to per-character extraction')
            merged.pop('extraction_error', None)
            for char in participants:
                try:
                    char_ctx = {char: plot_ctx[char]} if char in plot_ctx else {}
                    char_prompt = build_extraction_prompt(
                        title=title, author=author, text=scene_text,
                        chapter_title=plot.get('chapter_title'),
                        plot_summary=plot.get('summary'),
                        key_characters=[{'name': char, 'experience': ''}],
                        character_context=char_ctx,
                        focus_character=char,
                    )
                    char_data = llm.call_json(char_prompt, model=model)
                    char_inner = (char_data.get('plots') or [{}])[0]
                    char_scenes = char_inner.get('scenes', [])
                    for cs in char_scenes:
                        for t in cs.get('trajectories', []):
                            if t.get('steps'):
                                merged['trajectories'].append(t)
                except Exception as e:
                    print(f'      character {char} failed: {str(e)[:80]}')
        merged_scenes.append(merged)

    return {'plots': [{'plot_id': 0, 'chapter_title': plot.get('chapter_title'),
                       'scenes': merged_scenes}]}


def extract_one(llm, model, title, author, plot, plot_ctx):
    try:
        data = single_pass(llm, model, title, author, plot, plot_ctx)
        if not is_valid_structure(data) or is_suspicious_empty(data, plot):
            print(f'  plot {plot["plot_id"]}: single-pass '
                  f'{"invalid structure" if not is_valid_structure(data) else "suspicious empty"}'
                  f', falling back to two-pass')
            data = two_pass(llm, model, title, author, plot, plot_ctx)
    except Exception:
        try:
            data = two_pass(llm, model, title, author, plot, plot_ctx)
        except Exception as e:
            return None, str(e)

    data['plot_id'] = plot['plot_id']
    data['chapter_title'] = plot.get('chapter_title')
    data['prominence'] = plot.get('prominence')
    data['summary'] = plot.get('summary')
    return data, None


def main():
    parser = argparse.ArgumentParser(description='Stage B: T-A-O trajectory extraction')
    parser.add_argument('--title', default='Pride and Prejudice')
    parser.add_argument('--plots_dir', default='data/plots')
    parser.add_argument('--contexts_dir', default='data/contexts')
    parser.add_argument('--output_dir', default='data/trajectories')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--model', default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--only', type=int, default=None, help='extract only this plot_id (debug)')
    parser.add_argument('--plots', type=str, default=None,
                        help='comma-separated plot_ids to (re)extract')
    args = parser.parse_args()

    plots_data, contexts, alias_map = load_inputs(args.title, args.plots_dir, args.contexts_dir)
    plots = [p for p in plots_data['plots'] if p.get('text')]
    if args.only is not None:
        plots = [p for p in plots if p['plot_id'] == args.only]
    if args.plots:
        ids = {int(x) for x in args.plots.split(',')}
        plots = [p for p in plots if p['plot_id'] in ids]
    print(f'{len(plots)} plots to extract')

    out_dir = os.path.join(args.output_dir, args.title)
    os.makedirs(out_dir, exist_ok=True)
    llm = LLM(config_path=args.config)
    model = args.model or llm.default_model

    def out_path(p):
        return os.path.join(out_dir, f'plot_{p["plot_id"]}.json')

    todo = [p for p in plots if not os.path.exists(out_path(p))]
    print(f'{len(todo)} not yet extracted (skipping {len(plots) - len(todo)} existing)')

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for p in todo:
            plot_ctx = contexts.get(str(p['plot_id']), {})
            futures[ex.submit(extract_one, llm, model, plots_data['book'],
                              plots_data['author'], p, plot_ctx)] = p
        for fut in as_completed(futures):
            p = futures[fut]
            data, err = fut.result()
            if err:
                failures.append((p['plot_id'], err))
                print(f'  plot {p["plot_id"]}: FAILED - {err[:100]}')
                continue
            with open(out_path(p), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f'  plot {p["plot_id"]}: {count_steps(data)} steps')

    print(f'\nDone. extracted={len(todo) - len(failures)}, failed={len(failures)}')
    if failures:
        print('failures:', failures)
    print(f'Cost: {llm.cost_summary()}')


if __name__ == '__main__':
    main()
