#!/usr/bin/env python3
"""Run the full tolerant trajectory pipeline, from blocks through final data.

Stages are deliberately serial. A failed stage is recorded but does not stop
later stages; every child receives resume flags so a later full rerun only
retries incomplete work.
"""
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
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


def validate_title_lists(paths, src):
    groups = {name: set(load_titles(path)) for name, path in paths.items()}
    for left, right in (('eval_only', 'train_eval'),
                        ('eval_only', 'train_only'),
                        ('train_eval', 'train_only')):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            raise SystemExit(
                f'title lists overlap ({left} vs {right}): {overlap}')
    source_titles = set()
    with open(src, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            title = json.loads(line)['title']
            if title in source_titles:
                raise SystemExit(f'duplicate title in src: {title!r}')
            source_titles.add(title)
    listed = set().union(*groups.values())
    unknown = sorted(listed - source_titles)
    if unknown:
        raise SystemExit(f'titles not found in --src: {unknown}')
    missing = sorted(source_titles - listed)
    if missing:
        raise SystemExit(
            f'title lists do not cover every source book; missing: {missing}')
    from utils import safe_title
    seen_safe = {}
    for title in source_titles:
        safe = safe_title(title)
        if safe in seen_safe and seen_safe[safe] != title:
            raise SystemExit(
                f'safe_title collision in src: {seen_safe[safe]!r} vs {title!r}')
        seen_safe[safe] = title
    return {name: len(items) for name, items in groups.items()}


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
    ap.add_argument('--src', default='data/src/all_books.jsonl')
    ap.add_argument('--books-root', default='data')
    ap.add_argument('--eval-only-books', required=True)
    ap.add_argument('--train-eval-books', required=True)
    ap.add_argument('--train-only-books', required=True)
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
    expected_root = os.path.join(_HERE, 'data')
    if os.path.normcase(os.path.realpath(books_root)) != os.path.normcase(
            os.path.realpath(expected_root)):
        raise SystemExit(
            'This flat distribution requires --books-root data; the block and '
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
    lists = {
        'eval_only': absolute(args.eval_only_books),
        'train_eval': absolute(args.train_eval_books),
        'train_only': absolute(args.train_only_books),
    }
    counts = validate_title_lists(lists, src)
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
        ('01_blocks', [PY, '-u', 'run_batch_blocks.py', '--src', src,
         '--resume', '--best-effort', '--parallel-books',
         str(args.parallel_books), *common_rate, '--extract-model', args.model,
         '--judge-model', args.model], block_dir),
        ('02_profiles', [PY, '-u', 'run_batch_profiles.py', '--src', src,
         '--resume', '--parallel-books', str(args.parallel_books), '--workers',
         str(args.profile_workers), '--top', '9999', '--model', args.model,
         *common_rate], block_dir),
        ('03_rewash', [PY, '-u', 'run_batch_rewash.py', '--books-root',
         books_root, '--resume', '--parallel-books', str(args.parallel_books),
         '--workers', str(args.rewash_workers), '--model', args.model,
         *common_rate], _HERE),
        ('04a_build_eval_only', [PY, '-u', 'run_batch_build_dataset.py',
         '--books-root', books_root, '--titles-file', lists['eval_only'],
         '--mode', 'all', '--eval-output-dir',
         os.path.join(books_root, 'dataset', 'eval'), '--resume'], _HERE),
        ('04b_build_train_eval', [PY, '-u', 'run_batch_build_dataset.py',
         '--books-root', books_root, '--titles-file', lists['train_eval'],
         '--mode', 'split', '--split-ratio', '0.2', '--eval-output-dir',
         os.path.join(books_root, 'dataset', 'eval'), '--train-output-dir',
         os.path.join(books_root, 'dataset', 'train'), '--resume'], _HERE),
        ('04c_build_train_only', [PY, '-u', 'run_batch_build_dataset.py',
         '--books-root', books_root, '--titles-file', lists['train_only'],
         '--mode', 'train-only', '--train-output-dir',
         os.path.join(books_root, 'dataset', 'train'), '--resume'], _HERE),
    ]
    for split in ('eval', 'train'):
        anon_dir = os.path.join(books_root, 'dataset_anon', split)
        stages.append((f'05_anonymize_{split}', [
            PY, '-u', 'run_batch_anonymize.py', '--input-dir',
            os.path.join(books_root, 'dataset', split), '--output-dir', anon_dir,
            '--model', args.model, '--workers', str(args.anonymize_workers),
            '--parallel-files', str(args.parallel_anonymize_files),
            '--resume', *common_rate], _HERE))
    for split in ('eval', 'train'):
        anon_dir = os.path.join(books_root, 'dataset_anon', split)
        stages.append((f'06_fill_{split}', [
            PY, '-u', 'run_batch_fill_variants.py', '--input-dir', anon_dir,
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
