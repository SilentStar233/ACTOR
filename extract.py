# Phase 1: Extract T-A-O behavioral trajectories from a single chapter of a novel.
import argparse
import json
import os
import re
import sys

import openai

from prompts import EXTRACTION_PROMPT


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_book(jsonl_path, title):
    """Load one book from a jsonl dataset.

    When the AT_BOOK_CACHE_DIR env var is set, per-book JSON extracts are
    cached there (keyed by safe title), so the many stage subprocesses of a
    batch do not each rescan the full multi-MB jsonl.
    """
    cache_dir = os.environ.get('AT_BOOK_CACHE_DIR')
    cache_path = None
    src_fp = None
    if cache_dir:
        from utils import atomic_json_dump, safe_title
        stat = os.stat(jsonl_path)
        src_fp = {'src': os.path.abspath(jsonl_path), 'size': stat.st_size,
                  'mtime': int(stat.st_mtime)}
        cache_path = os.path.join(cache_dir, safe_title(title) + '.json')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding='utf-8') as f:
                    cached = json.load(f)
                # Stale cache (dataset replaced or path changed) is ignored.
                if cached.get('_src_fp') == src_fp:
                    return cached['book']
            except Exception:
                pass  # corrupt/stale entry: rescan and rewrite
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            book = json.loads(line)
            if book.get('title') == title:
                if cache_path:
                    os.makedirs(cache_dir, exist_ok=True)
                    atomic_json_dump(cache_path,
                                     {'_src_fp': src_fp, 'book': book})
                return book
    raise ValueError(f'Book not found in {jsonl_path}: {title}')


def extract_chapter(content, chapter):
    """Slice the text of `chapter` out of the book content (between 'Chapter N.' and 'Chapter N+1.')."""
    # Chapter headings may use non-breaking spaces (U+00A0), e.g. 'Chapter\xa01.'
    start_pat = rf'Chapter\s+{chapter}\.\s*\n'
    next_pat = rf'Chapter\s+{chapter + 1}\.\s*\n'
    start = re.search(start_pat, content)
    if not start:
        raise ValueError(f'Chapter {chapter} not found')
    end = re.search(next_pat, content)
    s = start.start()
    e = end.start() if end else len(content)
    return content[s:e].strip()


def call_llm(config, model, prompt):
    client = openai.OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url'],
        timeout=300,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16384,
        temperature=0,
    )
    return completion.choices[0].message.content


def parse_json_response(text):
    """Best-effort JSON parsing: strip markdown fences, take the outermost braces."""
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError('No JSON object found in response')
    return json.loads(text[start:end + 1])


def main():
    parser = argparse.ArgumentParser(description='Extract T-A-O trajectories from one chapter')
    parser.add_argument('--src', default='data/src/all_books.jsonl',
                        help='Source jsonl containing raw book text')
    parser.add_argument('--title', default='Pride and Prejudice')
    parser.add_argument('--chapter', type=int, default=1)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--model', default='deepseek-chat')
    parser.add_argument('--output_dir', default='data/extracted')
    args = parser.parse_args()

    config = load_config(args.config)
    book = load_book(args.src, args.title)

    print(f'Book: {book["title"]} by {book["author"]}')
    print(f'Extracting Chapter {args.chapter}...')
    text = extract_chapter(book['content'], args.chapter)
    print(f'Chapter length: {len(text)} chars')

    prompt = EXTRACTION_PROMPT.format(
        title=book['title'],
        author=book['author'],
        text=text,
    )

    print(f'Calling {args.model}...')
    response = call_llm(config, args.model, prompt)
    if response is None:
        sys.exit('LLM call failed')

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, f'{book["title"]}_ch{args.chapter}.raw.txt')
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(response)

    try:
        data = parse_json_response(response)
    except Exception as e:
        print(f'JSON parsing failed: {e}')
        print(f'Raw response saved to: {raw_path}')
        sys.exit(1)

    out_path = os.path.join(args.output_dir, f'{book["title"]}_ch{args.chapter}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_plots = len(data.get('plots', []))
    n_scenes = sum(len(p.get('scenes', [])) for p in data.get('plots', []))
    n_traj = sum(len(s.get('trajectories', [])) for p in data.get('plots', []) for s in p.get('scenes', []))
    n_steps = sum(len(t.get('steps', [])) for p in data.get('plots', []) for s in p.get('scenes', []) for t in s.get('trajectories', []))
    print(f'Done. plots={n_plots}, scenes={n_scenes}, trajectories={n_traj}, steps={n_steps}')
    print(f'Output: {out_path}')


if __name__ == '__main__':
    main()
