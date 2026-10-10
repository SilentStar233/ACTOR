# Stage A: detect plot boundaries chunk by chunk.
# - Rolling character registry (canonical names) passed between chunks
# - CoSER-style truncated-plot continuation across chunk boundaries
import argparse
import json
import os

from llm import LLM
from prompts import PLOT_DETECTION_PROMPT
from chunk import create_chunks
from extract import load_book


def render_registry(registry):
    if not registry:
        return '(empty — no characters identified yet)'
    lines = []
    for c in registry:
        alias = f" (also called: {', '.join(c['aliases'])})" if c['aliases'] else ''
        lines.append(f"- {c['canonical']}{alias}")
    return '\n'.join(lines)


def update_registry(registry, updates, plots):
    for upd in updates:
        canon = upd.get('canonical')
        alias = upd.get('new_alias')
        if not canon:
            continue
        entry = next((c for c in registry if c['canonical'] == canon), None)
        if entry is None:
            registry.append({'canonical': canon, 'aliases': [alias] if alias else []})
        elif alias and alias not in entry['aliases']:
            entry['aliases'].append(alias)
    # register key_characters names not yet seen
    for p in plots:
        for c in p.get('key_characters', []):
            name = c.get('name')
            if name and not any(r['canonical'] == name or name in r['aliases'] for r in registry):
                registry.append({'canonical': name, 'aliases': []})


def detect_plots(book, chunks, llm, model):
    registry = []
    truncated = []
    finished_plots = []
    chapter_beginnings = []

    for i, chunk in enumerate(chunks):
        prompt = PLOT_DETECTION_PROMPT.format(
            title=book['title'],
            author=book['author'],
            registry=render_registry(registry),
            truncated_plots=json.dumps(truncated, ensure_ascii=False, indent=2) if truncated else 'None',
            chunk=chunk,
        )
        result = llm.call_json(prompt, model=model)

        chapter_beginnings.extend(result.get('chapter_beginnings', []))
        plots = result.get('plots', [])
        update_registry(registry, result.get('character_registry_updates', []), plots)

        # The first len(truncated) plots are completions of the carried-over ones.
        n_carry = len(truncated)
        current = plots[:n_carry] + plots[n_carry:]

        truncated = [p for p in current if p.get('state') == 'truncated']
        finished = [p for p in current if p.get('state') != 'truncated']
        for p in finished:
            p['i_chunk'] = i
        finished_plots.extend(finished)
        print(f'  chunk {i + 1}/{len(chunks)}: +{len(finished)} finished, '
              f'{len(truncated)} truncated carried over')

    # Anything still truncated at the end of the book is kept as-is.
    for p in truncated:
        p['state'] = 'truncated_end'
        finished_plots.append(p)

    for pid, p in enumerate(finished_plots):
        p['plot_id'] = pid
    return finished_plots, registry, chapter_beginnings


def main():
    parser = argparse.ArgumentParser(description='Stage A: plot boundary detection')
    parser.add_argument('--src', default='data/src/all_books.jsonl')
    parser.add_argument('--title', default='Pride and Prejudice')
    parser.add_argument('--chunk_size', type=int, default=60000)
    parser.add_argument('--model', default=None, help='override default model')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output_dir', default='data/plots')
    args = parser.parse_args()

    book = load_book(args.src, args.title)
    print(f'Book: {book["title"]} by {book["author"]}, {len(book["content"])} chars')

    chunks = create_chunks(book['content'], chunk_size=args.chunk_size)
    print(f'{len(chunks)} chunks (~{args.chunk_size} chars each)')

    llm = LLM(config_path=args.config)
    if args.model:
        llm.default_model = args.model

    plots, registry, chapter_beginnings = detect_plots(book, chunks, llm, llm.default_model)

    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        'book': book['title'],
        'author': book['author'],
        'chapter_beginnings': chapter_beginnings,
        'registry': registry,
        'plots': plots,
    }
    out_path = os.path.join(args.output_dir, f'{book["title"]}.plots.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'\nDone. {len(plots)} plots, {len(registry)} characters in registry.')
    print(f'Output: {out_path}')
    print(f'Cost: {llm.cost_summary()}')


if __name__ == '__main__':
    main()
