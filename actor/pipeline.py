#!/usr/bin/env python3
"""Run the full tolerant trajectory pipeline, from blocks through final data.

Stages are deliberately serial. A failed stage is recorded but does not stop
later stages; every child receives resume flags so a later full rerun only
retries incomplete work.
"""
from actor.shared.paths import ROOT_DIR

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = ROOT_DIR
PY = sys.executable


def load_titles(path):
    if not os.path.isfile(path):
        raise SystemExit(f'missing titles file: {path}')
    with open(path, encoding='utf-8') as f:
        titles = [line.strip() for line in f
                  if line.strip() and not line.lstrip().startswith('#')]
    if len(titles) != len(set(titles)):
        raise SystemExit(f'duplicate title in {path}')
    return titles


def preflight(src, root, paths):
    """Validate inputs and report existing artifacts without API calls or writes."""
    from actor.shared.utils import safe_title

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
        'eval_only': load_titles(paths['eval_only']),
        'train_eval': load_titles(paths['train_eval']),
        'train_only': load_titles(paths['train_only']),
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
    return {name: len(group) for name, group in groups.items()}


def run_stage(name, command, cwd, log_dir, results):
    log_path = os.path.join(log_dir, f'{name}.log')
    started = time.time()
    print(f'\n===== {name} started: {time.ctime(started)} =====', flush=True)
    print(' '.join(command), flush=True)
    try:
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f'\n===== started {time.ctime(started)} =====\n')
            log.flush()
            completed = subprocess.run(command, cwd=cwd, stdout=log,
                                       stderr=subprocess.STDOUT)
        code = completed.returncode
        state = 'ok' if code == 0 else 'failed_continue'
    except Exception as error:
        code = None
        state = 'launch_error_continue'
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f'LAUNCH ERROR: {type(error).__name__}: {error}\n')
    results[name] = {
        'status': state,
        'returncode': code,
        'elapsed_sec': int(time.time() - started),
        'log': log_path,
    }
    print(f'===== {name}: {state} (returncode={code}); continuing =====',
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default=os.path.join(ROOT_DIR, 'data/src/all_books.jsonl'))
    ap.add_argument('--books-root', default=os.path.join(ROOT_DIR, 'data'))
    ap.add_argument('--eval-only-books', required=True)
    ap.add_argument('--train-eval-books', required=True)
    ap.add_argument('--train-only-books', required=True)
    ap.add_argument('--check-only', action='store_true',
                    help='Validate source/lists and report artifacts without running stages')
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--parallel-books', type=int, default=8)
    ap.add_argument('--profile-workers', type=int, default=5)
    ap.add_argument('--rewash-workers', type=int, default=8)
    ap.add_argument('--anonymize-workers', type=int, default=8)
    ap.add_argument('--parallel-anonymize-files', type=int, default=1)
    ap.add_argument('--max-rpm', type=int, default=2000)
    ap.add_argument('--max-tpm', type=int, default=4000000)
    args = ap.parse_args()

    def absolute(path):
        return path if os.path.isabs(path) else os.path.join(_HERE, path)

    src = absolute(args.src)
    books_root = absolute(args.books_root)
    lists = {
        'eval_only': absolute(args.eval_only_books),
        'train_eval': absolute(args.train_eval_books),
        'train_only': absolute(args.train_only_books),
    }
    counts = preflight(src, books_root, lists)
    if args.check_only:
        return

    expected_root = os.path.join(_HERE, 'data')
    if os.path.normcase(os.path.realpath(books_root)) != os.path.normcase(
            os.path.realpath(expected_root)):
        raise SystemExit(
            'This distribution requires --books-root data; the block and '
            'profile stages write to the repository data/ directory.')
    if not os.path.isfile(src):
        raise SystemExit(f'missing source JSONL: {src}')
    config_path = os.path.join(_HERE, 'config.json')
    if not os.path.isfile(config_path):
        raise SystemExit(f'missing API config: {config_path}')
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    if (not isinstance(config, dict)
            or not isinstance(config.get('api_key'), str)
            or not config['api_key'].strip()
            or config['api_key'].startswith('REPLACE_')
            or not isinstance(config.get('base_url'), str)
            or not config['base_url'].strip()):
        raise SystemExit('Set a real api_key and base_url in config.json')
    try:
        import openai  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            f'missing Python dependency {error.name!r}; run '
            f'"{PY} -m pip install -r requirements.txt" first') from error
    print(f'validated disjoint title lists: {counts}', flush=True)

    block_dir = _HERE
    # Empty title lists are valid. Create these directories so the downstream
    # batch stages can enumerate an empty split without raising FileNotFound.
    for split in ('eval', 'train'):
        os.makedirs(os.path.join(books_root, 'dataset', split), exist_ok=True)
        os.makedirs(os.path.join(books_root, 'dataset_anon', split), exist_ok=True)
    logs = os.path.join(books_root, 'logs', 'full_pipeline')
    os.makedirs(logs, exist_ok=True)
    results = {}

    common_rate = ['--max-rpm', str(args.max_rpm),
                   '--max-tpm', str(args.max_tpm)]
    stages = [
        ('01_blocks', [PY, '-u', '-m', 'actor.trajectory_extraction.batch', '--src', src,
         '--resume', '--best-effort', '--parallel-books',
         str(args.parallel_books), *common_rate, '--extract-model', args.model,
         '--judge-model', args.model], block_dir),
        ('02_profiles', [PY, '-u', '-m', 'actor.character_profiles.batch', '--src', src,
         '--resume', '--parallel-books', str(args.parallel_books), '--workers',
         str(args.profile_workers), '--top', '9999', '--model', args.model,
         *common_rate], block_dir),
        ('03_rewash', [PY, '-u', '-m', 'actor.action_refinement.batch', '--books-root',
         books_root, '--resume', '--parallel-books', str(args.parallel_books),
         '--workers', str(args.rewash_workers), '--model', args.model,
         *common_rate], _HERE),
        ('04a_build_eval_only', [PY, '-u', '-m', 'actor.dataset_building.batch',
         '--books-root', books_root, '--titles-file', lists['eval_only'],
         '--mode', 'all', '--eval-output-dir',
         os.path.join(books_root, 'dataset', 'eval'), '--resume'], _HERE),
        ('04b_build_train_eval', [PY, '-u', '-m', 'actor.dataset_building.batch',
         '--books-root', books_root, '--titles-file', lists['train_eval'],
         '--mode', 'split', '--split-ratio', '0.2', '--eval-output-dir',
         os.path.join(books_root, 'dataset', 'eval'), '--train-output-dir',
         os.path.join(books_root, 'dataset', 'train'), '--resume'], _HERE),
        ('04c_build_train_only', [PY, '-u', '-m', 'actor.dataset_building.batch',
         '--books-root', books_root, '--titles-file', lists['train_only'],
         '--mode', 'train-only', '--train-output-dir',
         os.path.join(books_root, 'dataset', 'train'), '--resume'], _HERE),
    ]
    for split in ('eval', 'train'):
        anon_dir = os.path.join(books_root, 'dataset_anon', split)
        stages.append((f'05_anonymize_{split}', [
            PY, '-u', '-m', 'actor.name_anonymization.batch', '--input-dir',
            os.path.join(books_root, 'dataset', split), '--output-dir', anon_dir,
            '--model', args.model, '--workers', str(args.anonymize_workers),
            '--parallel-files', str(args.parallel_anonymize_files),
            '--resume', *common_rate], _HERE))
    for split in ('eval', 'train'):
        anon_dir = os.path.join(books_root, 'dataset_anon', split)
        stages.append((f'06_fill_{split}', [
            PY, '-u', '-m', 'actor.name_anonymization.fill_batch', '--input-dir', anon_dir,
            '--audit-dir', anon_dir, '--output-dir',
            os.path.join(books_root, 'dataset_final', split), '--books-root',
            books_root, '--resume'], _HERE))

    for name, command, cwd in stages:
        run_stage(name, command, cwd, logs, results)
        summary_path = os.path.join(logs, 'summary.json')
        tmp = summary_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'time': time.time(), 'stages': results}, f,
                      ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, summary_path)

    failed = [name for name, result in results.items()
              if result['status'] != 'ok']
    print(f'\nPipeline finished: {len(results) - len(failed)} stages ok, '
          f'{len(failed)} stages reported failures. Summary: {summary_path}',
          flush=True)
    if failed:
        print('Stages with partial/fatal failures: ' + ', '.join(failed),
              flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
