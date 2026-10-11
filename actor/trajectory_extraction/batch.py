# Batch scheduler: run actor.trajectory_extraction.pipeline for many books with a process
# pool, per-book status tracking, resume, and a batch summary.
#
# Reliability rules:
# - resume counts a book done ONLY when stage == 'done' and status is
#   ok/ok_with_warnings with a matching scope (mid-stage statuses never count);
# - a non-zero subprocess returncode always means failure, regardless of
#   what status.json says;
# - watchdogs, PER ATTEMPT: each book subprocess gets --book-timeout
#   (default 16h) wall-clock and is killed when no progress file
#   (pipeline.log/status.json/detect_state.json/cost.jsonl) updates for
#   --stall-timeout (default 1h). The budget restarts on every (re)run —
#   this is a circuit breaker for hung attempts, NOT a cumulative budget
#   across resume/retry;
# - batch.json is rewritten incrementally after every finished book;
# - SIGINT/SIGTERM kill all child process groups immediately and save
#   partials (interrupted books are marked 'interrupted').
# Usage: cd repository-root && python3 -m actor.trajectory_extraction.batch [--parallel-books 3] [--resume]
from actor.shared.paths import ROOT_DIR

import argparse
import hashlib
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

PY = sys.executable
LOCKED_EXIT_CODE = 17
TIMEOUT_CODE = 201
LAUNCH_ERROR_CODE = 202
STALL_CODE = 203
DONE_STATUSES = ('ok', 'ok_with_warnings')


def load_books(src, titles_file):
    books = {}
    with open(src, encoding='utf-8') as f:
        for line in f:
            book = json.loads(line)
            title = book['title']
            if title in books:
                raise SystemExit(f'duplicate title in src: {title!r}')
            books[title] = book
    if titles_file:
        with open(titles_file, encoding='utf-8') as f:
            wanted = [line.strip() for line in f if line.strip()]
        missing = [t for t in wanted if t not in books]
        if missing:
            raise SystemExit(f'titles not found in src: {missing}')
        return {t: books[t] for t in wanted}
    return books


def status_path(log_root, title):
    return os.path.join(log_root, safe_title(title), 'status.json')


def book_status(log_root, title):
    path = status_path(log_root, title)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


_LIVE = {}
_LIVE_LOCK = threading.Lock()
_INTERRUPTED = []  # titles snapshotted at interrupt time (before kill)


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


def run_one_book(title, args, env):
    """Run one book; never raises — returns (title, returncode, elapsed).

    Two per-attempt watchdogs: a flat wall-clock cap (--book-timeout) and a
    stall detector (--stall-timeout): if none of the book's progress files
    (pipeline.log / status.json / detect_state.json / cost.jsonl) change
    for that long, the book is considered hung and its process group killed.
    """
    command = [PY, '-u', '-m', 'actor.trajectory_extraction.pipeline', '--src', args.src,
               '--title', title, '--chunk-size', str(args.chunk_size),
               '--workers', str(args.workers),
               '--extract-model', args.extract_model,
               '--judge-model', args.judge_model]
    if args.best_effort:
        command.append('--best-effort')
    if args.plot_stage_only:
        command.append('--plot-stage-only')
    book_dir = os.path.join(_HERE, 'data', 'logs', safe_title(title))
    log_path = os.path.join(book_dir, 'pipeline.log')
    os.makedirs(book_dir, exist_ok=True)
    watched = [log_path,
               os.path.join(book_dir, 'status.json'),
               os.path.join(book_dir, 'detect_state.json'),
               os.path.join(book_dir, 'cost.jsonl')]
    started = time.time()
    deadline = started + args.book_timeout
    try:
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f'\n===== batch run started {time.ctime(started)} '
                      f'(cap {args.book_timeout}s, stall {args.stall_timeout}s)'
                      ' =====\n')
            log.flush()
            proc = subprocess.Popen(
                command, cwd=_HERE, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)  # own process group
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


def scope_matches(status_scope, want_scope):
    """A full run satisfies any want; a plot_stage run only satisfies
    plot_stage. Missing scope means 'full' (legacy status files)."""
    status_scope = status_scope or 'full'
    if want_scope == 'plot_stage':
        return status_scope in ('plot_stage', 'full')
    return status_scope == 'full'


def is_done(status, want_scope):
    return (status.get('stage') == 'done'
            and status.get('status') in DONE_STATUSES
            and scope_matches(status.get('scope'), want_scope))


def classify(title, code, elapsed, log_root, want_scope):
    status = book_status(log_root, title) or {}
    if code == LOCKED_EXIT_CODE:
        state = 'locked'
    elif code == TIMEOUT_CODE:
        state = 'timeout'
    elif code == LAUNCH_ERROR_CODE:
        state = 'launch_error'
    elif code == STALL_CODE:
        state = 'stalled'
    elif code < 0:
        state = 'killed'  # external signal, e.g. OOM killer (-9)
    elif code != 0:
        state = 'failed'  # returncode veto: status.json never overrides this
    elif is_done(status, want_scope):
        state = status['status']
    elif status.get('status') in DONE_STATUSES:
        state = 'incomplete'  # exited 0 but not done in this scope
    else:
        state = 'failed'
    return {'status': state, 'returncode': code,
            'elapsed_sec': int(elapsed), 'stage': status.get('stage'),
            'scope': status.get('scope'), 'error': status.get('error')}


def _raise_interrupt(signum, frame):
    # Route SIGTERM (the standard `kill <pid>` for background jobs) through
    # the same cleanup path as Ctrl-C.
    raise KeyboardInterrupt(f'signal {signum}')


def main():
    # Background shells (nohup ... &) start us with SIGINT ignored; restore
    # Python's KeyboardInterrupt handler so kill -INT / Ctrl-C runs cleanup.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    ap = argparse.ArgumentParser(description='batch scheduler for block-ID pipeline')
    ap.add_argument('--src', default=os.path.join(ROOT_DIR, 'data/src/all_books.jsonl'))
    ap.add_argument('--titles-file', default=None,
                    help='file with one book title per line (default: all books)')
    ap.add_argument('--parallel-books', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk-size', type=int, default=30000)
    ap.add_argument('--extract-model', default='gpt-5.6-luna')
    ap.add_argument('--judge-model', default='gpt-5.6-luna')
    ap.add_argument('--best-effort', action='store_true')
    ap.add_argument('--plot-stage-only', action='store_true')
    ap.add_argument('--resume', action='store_true',
                    help='skip books already done with matching content hash')
    ap.add_argument('--book-timeout', type=int, default=16 * 3600,
                    help='wall-clock cap per book PER ATTEMPT in seconds '
                         '(default 16h); restarts on every resume/retry')
    ap.add_argument('--stall-timeout', type=int, default=3600,
                    help='kill a book with no progress-file update for this '
                         'many seconds (default 1h)')
    ap.add_argument('--retry-failed', type=int, default=1,
                    help='retry failed books K times at the end of the batch')
    ap.add_argument('--max-rpm', type=int, default=None,
                    help='account-level requests/minute; divided among parallel books')
    ap.add_argument('--max-tpm', type=int, default=None,
                    help='account-level tokens/minute; divided among parallel books')
    args = ap.parse_args()
    if args.parallel_books < 1 or args.retry_failed < 0:
        raise ValueError('parallel-books >= 1 and retry-failed >= 0 required')
    args.src = os.path.abspath(args.src)
    print(f'concurrency budget: {args.parallel_books} books x '
          f'{args.workers} workers = {args.parallel_books * args.workers} '
          'in-flight requests (max)')

    books = load_books(args.src, args.titles_file)
    seen = {}
    for title in books:
        safe = safe_title(title)
        if safe in seen and seen[safe] != title:
            raise SystemExit(
                f'safe_title collision: {seen[safe]!r} vs {title!r}')
        seen[safe] = title

    log_root = os.path.join(_HERE, 'data', 'logs')
    batch_path = os.path.join(log_root, 'batch.json')
    env = dict(os.environ)
    env['AT_RUN_DIR'] = log_root
    env['AT_BOOK_CACHE_DIR'] = os.path.join(_HERE, 'data', 'book_cache')
    env['PYTHONUNBUFFERED'] = '1'
    if args.max_rpm:
        env['AT_MAX_RPM'] = str(max(1, args.max_rpm // args.parallel_books))
    if args.max_tpm:
        env['AT_MAX_TPM'] = str(max(1, args.max_tpm // args.parallel_books))

    want_scope = 'plot_stage' if args.plot_stage_only else 'full'
    todo, skipped, changed = [], [], []
    for title, book in books.items():
        content_sha = hashlib.sha256(
            book['content'].encode('utf-8')).hexdigest()
        status = book_status(log_root, title) if args.resume else None
        if status is None:
            todo.append(title)
            continue
        if status.get('content_sha256') and \
                status['content_sha256'] != content_sha:
            changed.append(title)
            continue
        if is_done(status, want_scope):
            skipped.append((title, status['status']))
        else:
            todo.append(title)
    # Longest books first: avoid ending the batch with one giant book alone.
    todo.sort(key=lambda t: -len(books[t]['content']))

    print(f'books: {len(books)} total, {len(todo)} to run, '
          f'{len(skipped)} done, {len(changed)} content-changed')
    if changed:
        print('content changed (not rerun automatically):')
        for title in changed:
            print(f'  - {title}')

    results = {}

    def flush_batch():
        for title, state in skipped:
            results.setdefault(title, {'status': state, 'skipped': True})
        for title in changed:
            results.setdefault(title, {'status': 'content_changed',
                                       'skipped': True})
        atomic_json_dump(batch_path, {
            'time': time.time(), 'src': args.src,
            'best_effort': args.best_effort, 'books': results,
        })

    def run_round(titles, round_label):
        pool = ThreadPoolExecutor(max_workers=args.parallel_books)
        done_in_round = 0
        try:
            futures = {pool.submit(run_one_book, title, args, env): title
                       for title in titles}
            for future in as_completed(futures):
                title, code, elapsed = future.result()
                results[title] = classify(title, code, elapsed, log_root,
                                          want_scope)
                done_in_round += 1
                print(f"[{round_label} {done_in_round}/{len(titles)}] {title}: "
                      f"{results[title]['status']} "
                      f"({int(elapsed)}s, stage={results[title].get('stage')})",
                      flush=True)
                flush_batch()
        except KeyboardInterrupt:
            _INTERRUPTED.extend(live_titles())  # snapshot BEFORE killing;
            kill_live()  # workers' finally would empty _LIVE within ms
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    children_interrupted = False
    try:
        run_round(todo, 'main')
        for attempt in range(1, args.retry_failed + 1):
            # 'timeout' books are NOT retried: the per-attempt budget is a
            # circuit breaker, and re-running a book that just burned 16h
            # doubles the damage. 'stalled' (1h zero-progress) is cheap and
            # usually transient, so it gets exactly one retry.
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
        # A second SIGINT/SIGTERM during cleanup must not interrupt the
        # cleanup itself (would orphan children and lose the summary).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print('\ninterrupted: killed live process groups, saving partial '
              'batch (rerun with --resume)', flush=True)
        # Escalate in case a child ignores SIGTERM.
        time.sleep(3)
        kill_live(signal.SIGKILL)
        # Books still in flight have no result row; mark them so the
        # summary and batch.json show what was cut short.
        for live in _INTERRUPTED:
            results.setdefault(live, {'status': 'interrupted'})
        flush_batch()

    ok = sum(1 for r in results.values() if r['status'] == 'ok')
    warned = sum(1 for r in results.values()
                 if r['status'] == 'ok_with_warnings')
    failed = [t for t, r in results.items()
              if r['status'] in ('failed', 'timeout', 'stalled', 'incomplete',
                                 'killed', 'launch_error', 'interrupted')]
    locked = [t for t, r in results.items() if r['status'] == 'locked']
    print(f'\n=== batch summary ===')
    print(f'ok={ok} ok_with_warnings={warned} failed={len(failed)} '
          f'locked={len(locked)} skipped={len(skipped)} '
          f'content_changed={len(changed)}')
    if failed:
        print('failed books:')
        for title in failed:
            print(f"  - {title} (stage={results[title].get('stage')}, "
                  f"error={str(results[title].get('error'))[:120]})")
    # Aggregate needs_review gates so humans know where to look.
    gate_counts = {}
    for title in books:
        path = os.path.join(log_root, safe_title(title), 'needs_review.json')
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                items = json.load(f).get('items', [])
        except Exception:
            continue
        for item in items:
            gate_counts.setdefault(item.get('gate', '?'), []).append(title)
    if gate_counts:
        print('needs_review by gate:')
        for gate, titles in sorted(gate_counts.items()):
            print(f'  {gate}: {len(titles)} book(s)')
    print('summary: data/logs/batch.json')
    if children_interrupted:
        sys.exit(130)
    if failed or changed or locked:
        sys.exit(1)


if __name__ == '__main__':
    main()
