# Batch scheduler: run actor.action_refinement.actions for many books with a process pool,
# per-book completion tracking, resume, and a batch summary. Mirrors the
# actor.trajectory_extraction.batch architecture.
#
# Reliability rules:
# - a book counts as done ONLY when its _summary.json says complete=true
#   (every action of every entity carries counterfactual_action_rewashed);
# - a non-zero subprocess returncode always means failure;
# - watchdogs PER ATTEMPT: --book-timeout wall-clock cap, and a stall
#   detector (--stall-timeout) on the book's progress files (rewash.log /
#   cost.jsonl / character_timelines_rewashed dir mtime). Budgets restart on
#   every (re)run;
# - batch.json is rewritten after every finished book;
# - SIGINT/SIGTERM kill all child process groups and save partials.
#
# Usage:
#   cd repository-root
#   nohup python3 -m actor.action_refinement.batch \
#     --books-root data \
#     --resume --parallel-books 8 --max-rpm 2000 --max-tpm 4000000 \
#     --model gpt-5.6-luna \
#     > data/logs/rewash_batch.log 2>&1 &
from actor.shared.paths import ROOT_DIR

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = ROOT_DIR
from actor.shared.utils import atomic_json_dump, safe_title
from actor.shared.utils import list_book_titles, resolve_book_dirs, tree_signature

PY = sys.executable
TIMEOUT_CODE = 201
LAUNCH_ERROR_CODE = 202
STALL_CODE = 203
DEFAULT_BOOKS_ROOT = os.path.join(_HERE, 'data')


def list_books(books_root, titles_file):
    """All books under books_root (layout-aware via rewash_actions helpers)."""
    if titles_file:
        with open(titles_file, encoding='utf-8') as f:
            return [line.strip() for line in f
                    if line.strip() and not line.startswith('#')]
    return list_book_titles(books_root)


def summary_path(books_root, title):
    _, out_dir, _ = resolve_book_dirs(books_root, title)
    if out_dir is None:
        return None
    return os.path.join(out_dir, '_summary.json')


def book_summary(books_root, title):
    path = summary_path(books_root, title)
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def is_done(summary, books_root=None, title=None):
    if not summary or not summary.get('complete'):
        return False
    if books_root is None or title is None:
        return True
    ct_dir, _, _ = resolve_book_dirs(books_root, title)
    return bool(ct_dir and summary.get('source_signature')
                == tree_signature(ct_dir))


_LIVE = {}
_LIVE_LOCK = threading.Lock()
_INTERRUPTED = []


def kill_live(sig=signal.SIGTERM):
    with _LIVE_LOCK:
        procs = [proc for proc, _ in _LIVE.values()]
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            pass


def live_titles():
    with _LIVE_LOCK:
        return [title for _, title in _LIVE.values()]


def run_one_book(title, args, log_root, env):
    """Run one book; never raises — returns (title, returncode, elapsed)."""
    command = [PY, '-u', '-m', 'actor.action_refinement.actions',
               '--book', title, '--books-root', args.books_root,
               '--workers', str(args.workers), '--model', args.model]
    book_log_dir = os.path.join(log_root, safe_title(title))
    os.makedirs(book_log_dir, exist_ok=True)
    log_path = os.path.join(book_log_dir, 'rewash.log')
    _, out_dir, _ = resolve_book_dirs(args.books_root, title)
    watched = [log_path, os.path.join(book_log_dir, 'cost.jsonl')]
    started = time.time()
    deadline = started + args.book_timeout
    try:
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f'\n===== rewash run started {time.ctime(started)} '
                      f'(cap {args.book_timeout}s, stall {args.stall_timeout}s)'
                      ' =====\n')
            log.flush()
            proc = subprocess.Popen(
                command, cwd=_HERE, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            with _LIVE_LOCK:
                _LIVE[proc.pid] = (proc, title)
            last_progress = time.time()
            kill_reason = None
            try:
                while True:
                    code = proc.poll()
                    if code is not None:
                        break
                    now = time.time()
                    if now > deadline:
                        kill_reason = 'timeout'
                        break
                    latest = max(
                        (os.path.getmtime(p) for p in watched
                         if os.path.exists(p)),
                        default=last_progress)
                    # Entity files are replaced atomically on every flush, so
                    # the output dir mtime is also a progress signal.
                    if out_dir and os.path.isdir(out_dir):
                        latest = max(latest, os.path.getmtime(out_dir))
                    if latest > last_progress:
                        last_progress = latest
                    elif now - last_progress > args.stall_timeout:
                        kill_reason = 'stalled'
                        break
                    time.sleep(30)
                if kill_reason:
                    log.write(f'===== book {kill_reason}, '
                              'killing process group =====\n')
                    log.flush()
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()
                    return (title,
                            TIMEOUT_CODE if kill_reason == 'timeout'
                            else STALL_CODE,
                            time.time() - started)
            finally:
                with _LIVE_LOCK:
                    _LIVE.pop(proc.pid, None)
        return title, code, time.time() - started
    except Exception:
        return title, LAUNCH_ERROR_CODE, time.time() - started


def classify(title, code, elapsed, books_root):
    summary = book_summary(books_root, title) or {}
    if code == TIMEOUT_CODE:
        state = 'timeout'
    elif code == LAUNCH_ERROR_CODE:
        state = 'launch_error'
    elif code == STALL_CODE:
        state = 'stalled'
    elif code < 0:
        state = 'killed'
    elif code != 0:
        state = 'failed'
    elif is_done(summary, books_root, title):
        state = 'ok'
    else:
        state = 'incomplete'  # exited 0 but some actions still unrewashed
    return {'status': state, 'returncode': code, 'elapsed_sec': int(elapsed),
            'incomplete': summary.get('incomplete'),
            'total_actions': summary.get('total_actions')}


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    ap = argparse.ArgumentParser(description='batch scheduler for rewash_actions')
    ap.add_argument('--books-root', default=DEFAULT_BOOKS_ROOT,
                    help='root dir with <book>/character_timelines/')
    ap.add_argument('--titles-file', default=None,
                    help='one book title per line (default: all under books-root)')
    ap.add_argument('--parallel-books', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8,
                    help='thread workers per book subprocess')
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--resume', action='store_true',
                    help='skip books whose _summary.json says complete')
    ap.add_argument('--book-timeout', type=int, default=4 * 3600,
                    help='wall-clock cap per book PER ATTEMPT (default 4h)')
    ap.add_argument('--stall-timeout', type=int, default=1800,
                    help='kill a book with no progress for this many seconds '
                         '(default 30min)')
    ap.add_argument('--retry-failed', type=int, default=2,
                    help='retry rounds for failed/incomplete books at the end')
    ap.add_argument('--max-rpm', type=int, default=None,
                    help='account-level requests/min, divided among parallel books')
    ap.add_argument('--max-tpm', type=int, default=None,
                    help='account-level tokens/min, divided among parallel books')
    args = ap.parse_args()
    if args.parallel_books < 1 or args.retry_failed < 0:
        raise ValueError('parallel-books >= 1 and retry-failed >= 0 required')
    args.books_root = os.path.abspath(args.books_root)
    print(f'concurrency budget: {args.parallel_books} books x {args.workers} '
          f'workers = {args.parallel_books * args.workers} in-flight (max)')

    books = list_books(args.books_root, args.titles_file)
    seen = {}
    for title in books:
        safe = safe_title(title)
        if safe in seen and seen[safe] != title:
            raise SystemExit(f'safe_title collision: {seen[safe]!r} vs {title!r}')
        seen[safe] = title

    log_root = os.path.join(_HERE, 'data', 'logs', 'rewash_batch')
    os.makedirs(log_root, exist_ok=True)
    batch_path = os.path.join(log_root, 'batch.json')
    env = dict(os.environ)
    env['AT_RUN_DIR'] = log_root          # per-book LLM cache + cost.jsonl
    env['PYTHONUNBUFFERED'] = '1'
    if args.max_rpm:
        env['AT_MAX_RPM'] = str(max(1, args.max_rpm // args.parallel_books))
    if args.max_tpm:
        env['AT_MAX_TPM'] = str(max(1, args.max_tpm // args.parallel_books))

    todo, skipped = [], []
    for title in books:
        summary = book_summary(args.books_root, title) if args.resume else None
        if args.resume and is_done(summary, args.books_root, title):
            skipped.append(title)
        else:
            todo.append(title)
    # Biggest books first (by total actions if known, else by timeline dir size)
    def book_size(t):
        s = book_summary(args.books_root, t)
        if s and s.get('total_actions'):
            return s['total_actions']
        ct, _, _ = resolve_book_dirs(args.books_root, t)
        try:
            return sum(os.path.getsize(os.path.join(ct, f))
                       for f in os.listdir(ct) if f.endswith('.json'))
        except (OSError, TypeError):
            return 0
    todo.sort(key=lambda t: -book_size(t))

    print(f'books: {len(books)} total, {len(todo)} to run, '
          f'{len(skipped)} done', flush=True)

    results = {}

    def flush_batch():
        for title in skipped:
            results.setdefault(title, {'status': 'ok', 'skipped': True})
        atomic_json_dump(batch_path, {
            'time': time.time(), 'books_root': args.books_root,
            'model': args.model, 'books': results,
        })

    def run_round(titles, label):
        pool = ThreadPoolExecutor(max_workers=args.parallel_books)
        n = 0
        try:
            futures = {pool.submit(run_one_book, t, args, log_root, env): t
                       for t in titles}
            for future in as_completed(futures):
                title, code, elapsed = future.result()
                results[title] = classify(title, code, elapsed, args.books_root)
                n += 1
                print(f'[{label} {n}/{len(titles)}] {title}: '
                      f"{results[title]['status']} ({int(elapsed)}s, "
                      f"incomplete={results[title].get('incomplete')})",
                      flush=True)
                flush_batch()
        except KeyboardInterrupt:
            _INTERRUPTED.extend(live_titles())
            kill_live()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    children_interrupted = False
    try:
        run_round(todo, 'main')
        for attempt in range(1, args.retry_failed + 1):
            statuses = ('failed', 'incomplete', 'killed', 'launch_error') + (
                ('stalled',) if attempt == 1 else ())
            retry = [t for t in todo
                     if results.get(t, {}).get('status') in statuses]
            if not retry:
                break
            backoff = 60 * attempt
            print(f'\nretry round {attempt}: {len(retry)} books, '
                  f'sleeping {backoff}s first', flush=True)
            time.sleep(backoff)
            run_round(retry, f'retry-{attempt}')
    except KeyboardInterrupt:
        children_interrupted = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print('\ninterrupted: killed live process groups, saving partial '
              'batch (rerun with --resume)', flush=True)
        time.sleep(3)
        kill_live(signal.SIGKILL)
        for live in _INTERRUPTED:
            results.setdefault(live, {'status': 'interrupted'})
        flush_batch()

    ok = sum(1 for r in results.values() if r['status'] == 'ok')
    failed = [t for t, r in results.items()
              if r['status'] in ('failed', 'timeout', 'stalled', 'incomplete',
                                 'killed', 'launch_error', 'interrupted')]
    print(f'\n=== batch summary ===')
    print(f'ok={ok} failed={len(failed)} skipped={len(skipped)}')
    if failed:
        print('failed books:')
        for title in failed:
            print(f"  - {title} (incomplete={results[title].get('incomplete')})")
    print(f'summary: {batch_path}')
    if children_interrupted:
        sys.exit(130)
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
