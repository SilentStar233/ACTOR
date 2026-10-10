# v2 Stage A: detect mutually exclusive plot extraction segments.
# Does not modify the parent v1 detector.
# Usage: cd v2 && python detect_plots_v2.py --title "Book Title" --src path.jsonl
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from chunk import create_chunks
from detect_plots import render_registry
from extract import load_book
from extract_helpers_v2 import call_extraction_json
from llm import LLM
from plot_prompts_v2 import PLOT_DETECTION_PROMPT_V2

MAX_SPLIT_DEPTH = 3


def split_chunk_in_half(chunk):
    """Split chunk at a paragraph boundary nearest its midpoint."""
    midpoint = len(chunk) // 2
    candidates = []
    start = 0
    while True:
        idx = chunk.find('\n\n', start)
        if idx == -1:
            break
        candidates.append(idx + 2)
        start = idx + 2
    if not candidates:
        raise ValueError('cannot split chunk: no paragraph boundary found')
    cut = min(candidates, key=lambda pos: abs(pos - midpoint))
    left, right = chunk[:cut].strip(), chunk[cut:].strip()
    if not left or not right:
        raise ValueError('cannot split chunk into two non-empty halves')
    return left, right


def plot_character_names(plot):
    names = list(plot.get('key_character_names') or [])
    for character in plot.get('key_characters') or []:
        if isinstance(character, dict) and character.get('name'):
            names.append(character['name'])
        elif isinstance(character, str):
            names.append(character)
    return names


def update_registry_v2(registry, updates, plots):
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


def call_plot_detection(llm, model, prompt):
    try:
        return call_extraction_json(llm, prompt, model, max_tokens=65536)
    except (TypeError, AttributeError):
        return llm.call_json(prompt, model=model, max_tokens=65536)


def detect_one_chunk(book, chunk_text, registry, truncated, llm, model):
    prompt = PLOT_DETECTION_PROMPT_V2.format(
        title=book['title'], author=book['author'],
        registry=render_registry(registry),
        truncated_plots=(json.dumps(truncated, ensure_ascii=False, indent=2)
                         if truncated else 'None'),
        chunk=chunk_text,
    )
    result = call_plot_detection(llm, model, prompt)
    plots = result.get('plots', [])
    update_registry_v2(registry, result.get('character_registry_updates', []), plots)
    return result, plots


def process_chunk(book, chunk_text, registry, truncated, finished_plots,
                  chapter_beginnings, llm, model, i_chunk, label, depth=0):
    """Detect plots in one chunk; on truncation, split in half and retry."""
    try:
        result, plots = detect_one_chunk(
            book, chunk_text, registry, truncated, llm, model)
    except ValueError as error:
        if 'truncated' not in str(error).lower() or depth >= MAX_SPLIT_DEPTH:
            raise
        try:
            left, right = split_chunk_in_half(chunk_text)
        except ValueError:
            # Chunk can no longer be split at a paragraph boundary; report the
            # original truncation rather than the secondary split failure.
            raise error from None
        print(f'  {label}: truncated response, splitting into halves '
              f'(depth {depth + 1}/{MAX_SPLIT_DEPTH})')
        for half_index, half in enumerate((left, right)):
            truncated = process_chunk(
                book, half, registry, truncated, finished_plots,
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


def detect_plots(book, chunks, llm, model):
    registry = []
    truncated = []
    finished_plots = []
    chapter_beginnings = []
    for i, chunk in enumerate(chunks):
        truncated = process_chunk(
            book, chunk, registry, truncated, finished_plots,
            chapter_beginnings, llm, model, i,
            label=f'chunk {i + 1}/{len(chunks)}')
    for plot in truncated:
        plot['state'] = 'truncated_end'
        finished_plots.append(plot)
    for plot_id, plot in enumerate(finished_plots):
        plot['plot_id'] = plot_id
    return finished_plots, registry, chapter_beginnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/src/all_books.jsonl')
    ap.add_argument('--title', default='Pride and Prejudice')
    ap.add_argument('--chunk-size', type=int, default=30000)
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--output-dir', default='data/plots_detected')
    args = ap.parse_args()

    book = load_book(args.src, args.title)
    chunks = create_chunks(book['content'], chunk_size=args.chunk_size)
    llm = LLM(config_path='config.json', book=args.title)
    plots, registry, chapter_beginnings = detect_plots(
        book, chunks, llm, args.model)
    output = {'book': book['title'], 'author': book['author'],
              'chapter_beginnings': chapter_beginnings,
              'registry': registry, 'plots': plots,
              'detection_policy': 'mutually_exclusive_extraction_segments'}
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f'{args.title}.plots.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'{len(plots)} detected plots -> {path}')


if __name__ == '__main__':
    main()
