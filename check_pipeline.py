#!/usr/bin/env python3
"""Read-only preflight and manifest check for the full pipeline."""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from utils import safe_title


def read_titles(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f
                if line.strip() and not line.lstrip().startswith('#')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default='data/src/all_books.jsonl')
    ap.add_argument('--books-root', default='data')
    ap.add_argument('--eval-only-books', required=True)
    ap.add_argument('--train-eval-books', required=True)
    ap.add_argument('--train-only-books', required=True)
    args = ap.parse_args()

    def absolute(path):
        return path if os.path.isabs(path) else os.path.join(_HERE, path)

    src = absolute(args.src)
    root = absolute(args.books_root)
    if not os.path.isfile(src):
        raise SystemExit(f'missing source JSONL: {src}')
    with open(src, encoding='utf-8') as f:
        books = [json.loads(line) for line in f if line.strip()]
    if not books:
        raise SystemExit(f'source JSONL is empty: {src}')
    for index, book in enumerate(books, 1):
        if (not isinstance(book, dict)
                or not isinstance(book.get('title'), str)
                or not book['title'].strip()
                or not isinstance(book.get('content'), str)
                or not book['content'].strip()):
            raise SystemExit(
                f'source row {index} needs non-empty title and content')
    titles = [book['title'] for book in books]
    source = set(titles)
    problems = []
    if len(titles) != len(source):
        problems.append('duplicate titles in source JSONL')

    groups = {
        'eval_only': read_titles(absolute(args.eval_only_books)),
        'train_eval': read_titles(absolute(args.train_eval_books)),
        'train_only': read_titles(absolute(args.train_only_books)),
    }
    listed = set()
    for name, group in groups.items():
        if len(group) != len(set(group)):
            problems.append(f'duplicate titles in {name} list')
        overlap = listed & set(group)
        if overlap:
            problems.append(f'{name} overlaps previous lists: {sorted(overlap)}')
        listed.update(group)
    if listed != source:
        problems.append(f'lists missing={sorted(source-listed)} unknown={sorted(listed-source)}')

    seen_safe = {}
    for title in titles:
        safe = safe_title(title)
        if safe in seen_safe and seen_safe[safe] != title:
            problems.append(f'safe_title collision: {seen_safe[safe]!r} vs {title!r}')
        seen_safe[safe] = title

    stages = {
        'blocks': ('character_timelines', source),
        'profiles': ('profile_trials', source),
        'rewash': ('character_timelines_rewashed', source),
        'dataset_eval': (os.path.join('dataset', 'eval'),
                         set(groups['eval_only']) | set(groups['train_eval'])),
        'dataset_train': (os.path.join('dataset', 'train'),
                          set(groups['train_only']) | set(groups['train_eval'])),
        'anon_eval': (os.path.join('dataset_anon', 'eval'),
                      set(groups['eval_only']) | set(groups['train_eval'])),
        'anon_train': (os.path.join('dataset_anon', 'train'),
                       set(groups['train_only']) | set(groups['train_eval'])),
        'final_eval': (os.path.join('dataset_final', 'eval'),
                       set(groups['eval_only']) | set(groups['train_eval'])),
        'final_train': (os.path.join('dataset_final', 'train'),
                        set(groups['train_only']) | set(groups['train_eval'])),
    }
    counts = {}
    for stage, (relative, expected_titles) in stages.items():
        path = os.path.join(root, relative)
        if not os.path.isdir(path):
            counts[stage] = 0
        elif stage in ('blocks', 'profiles', 'rewash'):
            counts[stage] = sum(os.path.isdir(os.path.join(path, safe_title(t)))
                                for t in expected_titles)
        else:
            counts[stage] = sum(
                os.path.isfile(os.path.join(path, safe_title(t) + '.jsonl'))
                for t in expected_titles)

    print(f'source books: {len(titles)}')
    print('title lists: ' + ', '.join(f'{k}={len(v)}' for k, v in groups.items()))
    print('artifacts: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
    if problems:
        print('PRECHECK FAILED:')
        for problem in problems:
            print(f'  - {problem}')
        raise SystemExit(1)
    print('precheck OK')


if __name__ == '__main__':
    main()
