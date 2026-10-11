# Block-ID Stage A: detect mutually exclusive plot extraction segments.
# Chunks carry [B00012] block prefixes and
# the model cites first_block/last_block instead of verbatim sentences.
# Usage: cd repository-root && python3 -m actor.trajectory_extraction.detect_plots --title "Book" --src path.jsonl
from actor.shared.paths import ROOT_DIR

import argparse
import hashlib
import json
import os
import sys

_HERE = ROOT_DIR

from actor.shared.utils import load_book
from actor.trajectory_extraction.helpers import call_extraction_json
from actor.shared.llm import LLM
from actor.shared.utils import atomic_json_dump, safe_title

from actor.trajectory_extraction.blocks import (build_blocks, chunk_blocks, render_chunk, save_blocks)
from actor.trajectory_extraction.prompts import PLOT_DETECTION_PROMPT_BLOCKS

MAX_SPLIT_DEPTH = 3


def render_registry(registry):
    if not registry:
        return '(empty — no characters identified yet)'
    lines = []
    for c in registry:
        alias = f" (also called: {', '.join(c['aliases'])})" if c['aliases'] else ''
        lines.append(f"- {c['canonical']}{alias}")
    return '\n'.join(lines)


def plot_character_names(plot):
    names = list(plot.get('key_character_names') or [])
    for character in plot.get('key_characters') or []:
        if isinstance(character, dict) and character.get('name'):
            names.append(character['name'])
        elif isinstance(character, str):
            names.append(character)
    return names


def update_registry(registry, updates, plots):
    for update in updates:
        canonical = update.get('canonical')
        alias = update.get('new_alias')
        if not canonical:
            continue
        entry = next((item for item in registry if item['canonical'] == canonical), None)
        if entry is None:
            registry.append({'canonical': canonical, 'aliases': [alias] if alias else []})
        elif alias and alias not in entry['aliases']:
            entry['aliases'].append(alias)
    for plot in plots:
        for name in plot_character_names(plot):
            if name and not any(
                    name == entry['canonical'] or name in entry['aliases']
                    for entry in registry):
                registry.append({'canonical': name, 'aliases': []})


def detect_one_chunk(book, chunk_text, registry, truncated, llm, model):
    prompt = PLOT_DETECTION_PROMPT_BLOCKS.format(
        title=book['title'], author=book['author'],
        registry=render_registry(registry),
        truncated_plots=(json.dumps(truncated, ensure_ascii=False, indent=2)
                         if truncated else 'None'),
        chunk=chunk_text,
    )
    try:
        result = call_extraction_json(llm, prompt, model, max_tokens=65536)
    except (TypeError, AttributeError):
        result = llm.call_json(prompt, model=model, max_tokens=65536)
    plots = result.get('plots', [])
    update_registry(registry, result.get('character_registry_updates', []),
                       plots)
    return result, plots


def process_chunk(book, content, chunk, registry, truncated, finished_plots,
                  chapter_beginnings, llm, model, i_chunk, label, depth=0):
    """Detect plots in one block-chunk; on truncation, split in half."""
    try:
        result, plots = detect_one_chunk(
            book, render_chunk(content, chunk), registry, truncated,
            llm, model)
    except ValueError as error:
        if 'truncated' not in str(error).lower() or depth >= MAX_SPLIT_DEPTH:
            raise
        if len(chunk) < 2:
            raise error from None
        middle = len(chunk) // 2
        print(f'  {label}: truncated response, splitting into halves '
              f'(depth {depth + 1}/{MAX_SPLIT_DEPTH})')
        for half_index, half in enumerate((chunk[:middle], chunk[middle:])):
            truncated = process_chunk(
                book, content, half, registry, truncated, finished_plots,
                chapter_beginnings, llm, model, i_chunk,
                f'{label}.{half_index + 1}', depth=depth + 1)
        return truncated

    chapter_beginnings.extend(result.get('chapter_beginnings', []))
    new_truncated = [p for p in plots if p.get('state') == 'truncated']
    finished = [p for p in plots if p.get('state') != 'truncated']
    for plot in finished:
        plot['i_chunk'] = i_chunk
    finished_plots.extend(finished)
    print(f'{label}: +{len(finished)} finished, {len(new_truncated)} truncated')
    return new_truncated


def detect_plots(book, content, chunks, llm, model, state_path=None,
                 chunk_size=None):
    """Detect plots chunk by chunk, checkpointing after every chunk.

    state_path (logs/<book>/detect_state.json) lets a killed run resume
    from the next unfinished chunk instead of replaying via the LLM cache.
    The checkpoint carries a fingerprint (chunk layout + content hash);
    a stale checkpoint is discarded instead of resuming at a wrong offset.
    """
    registry = []
    truncated = []
    finished_plots = []
    chapter_beginnings = []
    start = 0
    fingerprint = {
        'chunk_count': len(chunks),
        'chunk_size': chunk_size,
        'content_sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(),
    }
    if state_path and os.path.exists(state_path):
        try:
            with open(state_path, encoding='utf-8') as f:
                state = json.load(f)
            if state.get('fingerprint') != fingerprint:
                print('detect checkpoint stale (chunk layout or content '
                      'changed); starting over')
            else:
                registry = state.get('registry') or []
                truncated = state.get('truncated') or []
                finished_plots = state.get('finished_plots') or []
                chapter_beginnings = state.get('chapter_beginnings') or []
                start = state.get('next_chunk') or 0
                if start > len(chunks):
                    raise ValueError('next_chunk out of range')
                if start:
                    print(f'resuming detect at chunk {start + 1}/{len(chunks)} '
                          f'({len(finished_plots)} plots already finished)')
        except Exception as error:
            print(f'detect checkpoint unreadable ({error}); starting over')
            registry, truncated, finished_plots, chapter_beginnings = [], [], [], []
            start = 0
    for i in range(start, len(chunks)):
        truncated = process_chunk(
            book, content, chunks[i], registry, truncated, finished_plots,
            chapter_beginnings, llm, model, i,
            label=f'chunk {i + 1}/{len(chunks)}')
        if state_path:
            atomic_json_dump(state_path, {
                'fingerprint': fingerprint,
                'next_chunk': i + 1, 'registry': registry,
                'truncated': truncated, 'finished_plots': finished_plots,
                'chapter_beginnings': chapter_beginnings,
            })
    for plot in truncated:
        plot['state'] = 'truncated_end'
        finished_plots.append(plot)
    for plot_id, plot in enumerate(finished_plots):
        plot['plot_id'] = plot_id
    if state_path and os.path.exists(state_path):
        os.remove(state_path)  # clean checkpoint on success
    return finished_plots, registry, chapter_beginnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(ROOT_DIR, 'data/src/all_books.jsonl'))
    ap.add_argument('--title', required=True)
    ap.add_argument('--chunk-size', type=int, default=30000)
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--output-dir', default=os.path.join(ROOT_DIR, 'data/plots_detected'))
    ap.add_argument('--fresh', action='store_true',
                    help='ignore any existing detect checkpoint and start over')
    args = ap.parse_args()

    book = load_book(args.src, args.title)
    content = book['content']
    blocks = build_blocks(content)
    chunks = chunk_blocks(blocks, chunk_size=args.chunk_size)
    print(f'{len(blocks)} blocks, {len(chunks)} chunks')

    run_dir = os.environ.get('AT_RUN_DIR', 'data/logs')
    state_path = os.path.join(run_dir, safe_title(args.title),
                              'detect_state.json')
    if args.fresh and os.path.exists(state_path):
        os.remove(state_path)

    llm = LLM(config_path=os.path.join(ROOT_DIR, 'config.json'), book=args.title)
    plots, registry, chapter_beginnings = detect_plots(
        book, content, chunks, llm, args.model, state_path=state_path,
        chunk_size=args.chunk_size)

    os.makedirs(args.output_dir, exist_ok=True)
    safe = safe_title(args.title)
    blocks_path = os.path.join(args.output_dir, f'{safe}.blocks.json')
    save_blocks(blocks, blocks_path)
    output = {'book': book['title'], 'author': book['author'],
              'chapter_beginnings': chapter_beginnings,
              'registry': registry, 'plots': plots,
              'blocks_file': os.path.basename(blocks_path),
              'detection_policy': 'mutually_exclusive_extraction_segments',
              'boundary_protocol': 'block_id'}
    path = os.path.join(args.output_dir, f'{safe}.plots.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'{len(plots)} detected plots -> {path}')
    print(f'block map -> {blocks_path}')


if __name__ == '__main__':
    main()
