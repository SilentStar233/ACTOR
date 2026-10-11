# Batch scheduler: run actor.character_profiles.generate for many books with a process
# pool, per-book completion tracking, resume, and a batch summary.
#
# Mirrors actor.trajectory_extraction.batch but only runs the profile stage.
# Books whose profile_trials/<safe>/index.json already exists are skipped.
#
# Reliability rules:
# - a book counts as done ONLY when its profile_trials/<safe>/index.json
#   exists and lists every entity from entities/<safe>.entities.json;
# - a non-zero subprocess returncode always means failure;
# - watchdogs PER ATTEMPT: --book-timeout wall-clock cap, and a stall
#   detector (--stall-timeout) on the book's progress files;
# - batch.json is rewritten after every finished book;
# - SIGINT/SIGTERM kill all child process groups immediately and save
#   partials (interrupted books are marked 'interrupted').
#
# Usage:
#   cd repository-root
#   nohup python3 -m actor.character_profiles.batch \
#     --src data/src/all_books.jsonl \
#     --resume --parallel-books 8 --max-rpm 2000 --max-tpm 4000000 \
#     --model gpt-5.6-luna --workers 5 --top 9999 \
#     > data/logs/profile_batch.log 2>&1 &
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

PY = sys.executable
LOCKED_EXIT_CODE = 17
TIMEOUT_CODE = 201
LAUNCH_ERROR_CODE = 202
STALL_CODE = 203


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


def profile_dir(safe):
    return os.path.join(_HERE, 'data', 'profile_trials', safe)


def profile_index_path(safe):
    return os.path.join(profile_dir(safe), 'index.json')


def entities_path(safe):
    return os.path.join(_HERE, 'data', 'entities', f'{safe}.entities.json')


def plots_path(safe):
    return os.path.join(_HERE, 'data', 'plots_final', f'{safe}.plots.json')


def character_timelines_dir(safe):
    return os.path.join(_HERE, 'data', 'character_timelines', safe)


def is_profile_done(safe):
    """Check that every entity with actions has a valid profile file."""
    idx_path = profile_index_path(safe)
    if not os.path.exists(idx_path):
        return False
    ent_path = entities_path(safe)
    if not os.path.exists(ent_path):
        return False
    try:
        idx = json.load(open(idx_path, encoding='utf-8'))
        ent = json.load(open(ent_path, encoding='utf-8'))
        idx_ids = {e['entity_id'] for e in idx.get('entities', [])}
        expected = {
            e['entity_id']: int(e.get('source_action_count') or 0)
            for e in ent.get('entities', [])
            if int(e.get('source_action_count') or 0) > 0
        }
        ent_ids = set(expected)
        if not ent_ids <= idx_ids:
            return False
        for entity_id in ent_ids:
            path = os.path.join(profile_dir(safe), f'{entity_id}.json')
            with open(path, encoding='utf-8') as f:
                profile = json.load(f)
            if not profile.get('profile_versions'):
                return False
            if int(profile.get('action_count') or -1) != expected[entity_id]:
                return False
        return True
    except Exception:
        return False


def upstream_status(safe):
    """Return (ready, reason) without launching a doomed profile process."""
    required = [
        ('entities', entities_path(safe)),
        ('plots_final', plots_path(safe)),
        ('character_timelines', character_timelines_dir(safe)),
        ('character_timelines/index.json',
         os.path.join(character_timelines_dir(safe), 'index.json')),
    ]
    missing = [label for label, path in required
               if not (os.path.isdir(path) if label == 'character_timelines'
                       else os.path.isfile(path))]
    if missing:
        return False, 'missing upstream: ' + ', '.join(missing)
    return True, None


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


def run_one_book(title, args, env):
    """Run actor.character_profiles.generate for one book; never raises.

    Returns (title, returncode, elapsed).
    """
    safe = safe_title(title)
    char_dir = character_timelines_dir(safe)
    ent_file = entities_path(safe)
    plots_file = plots_path(safe)
    out_dir = profile_dir(safe)

    command = [PY, '-u', '-m', 'actor.character_profiles.generate',
               '--title', title,
               '--character-dir', char_dir,
               '--entities', ent_file,
               '--plots-file', plots_file,
               '--output-dir', out_dir,
               '--top', str(args.top),
               '--model', args.model,
               '--workers', str(args.workers)]
    if args.resume:
        command.append('--resume')

    book_log_dir = os.path.join(_HERE, 'data', 'logs', safe)
    os.makedirs(book_log_dir, exist_ok=True)
    log_path = os.path.join(book_log_dir, 'profile.log')
    cost_path = os.path.join(book_log_dir, 'cost.jsonl')
    watched = [log_path, cost_path, out_dir]
    started = time.time()
    deadline = started + args.book_timeout
    try:
        with open(log_path, 'a', encoding='utf-8') as log:
            log.write(f'\n===== profile batch run started {time.ctime(started)} '
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


def classify(title, code, elapsed):
    safe = safe_title(title)
    if code == LOCKED_EXIT_CODE:
        state = 'locked'
    elif code == TIMEOUT_CODE:
        state = 'timeout'
    elif code == LAUNCH_ERROR_CODE:
        state = 'launch_error'
    elif code == STALL_CODE:
        state = 'stalled'
    elif code < 0:
        state = 'killed'
    elif code != 0:
        state = 'failed'
    elif is_profile_done(safe):
        state = 'ok'
    else:
        state = 'incomplete'
    return {'status': state, 'returncode': code,
            'elapsed_sec': int(elapsed)}


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    ap = argparse.ArgumentParser(
        description='batch scheduler for actor.character_profiles.generate')
    ap.add_argument('--src', default=os.path.join(ROOT_DIR, 'data/src/all_books.jsonl'))
    ap.add_argument('--titles-file', default=None,
                    help='file with one book title per line (default: all books)')
    ap.add_argument('--parallel-books', type=int, default=2)
    ap.add_argument('--workers', type=int, default=5,
                    help='thread workers per book (passed to generate_profiles)')
    ap.add_argument('--top', type=int, default=9999,
                    help='top N entities by action_count (default: all)')
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--resume', action='store_true',
                    help='skip books whose profile index covers all entities; '
                         'also passed to generate_profiles for per-entity resume')
    ap.add_argument('--book-timeout', type=int, default=8 * 3600,
                    help='wall-clock cap per book PER ATTEMPT (default 8h)')
    ap.add_argument('--stall-timeout', type=int, default=3600,
                    help='kill a book with no progress for this many seconds '
                         '(default 1h)')
    ap.add_argument('--retry-failed', type=int, default=2,
                    help='retry rounds for failed/incomplete books')
    ap.add_argument('--max-rpm', type=int, default=None,
                    help='account-level requests/min, divided among parallel books')
    ap.add_argument('--max-tpm', type=int, default=None,
                    help='account-level tokens/min, divided among parallel books')
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
    os.makedirs(log_root, exist_ok=True)
    batch_path = os.path.join(log_root, 'profile_batch.json')
    env = dict(os.environ)
    env['AT_RUN_DIR'] = log_root
    env['AT_BOOK_CACHE_DIR'] = os.path.join(_HERE, 'data', 'book_cache')
    env['PYTHONUNBUFFERED'] = '1'
    if args.max_rpm:
        env['AT_MAX_RPM'] = str(max(1, args.max_rpm // args.parallel_books))
    if args.max_tpm:
        env['AT_MAX_TPM'] = str(max(1, args.max_tpm // args.parallel_books))

    todo, skipped, skipped_upstream = [], [], {}
    for title in books:
        safe = safe_title(title)
        if args.resume and is_profile_done(safe):
            skipped.append(title)
        else:
            ready, reason = upstream_status(safe)
            if ready:
                todo.append(title)
            else:
                skipped_upstream[title] = reason

    # Sort by entity count descending (more entities = longer, run first)
    def book_entity_count(t):
        safe = safe_title(t)
        ent_path = entities_path(safe)
        if not os.path.exists(ent_path):
            return 0
        try:
            return len(json.load(open(ent_path, encoding='utf-8')).get('entities', []))
        except Exception:
            return 0
    todo.sort(key=lambda t: -book_entity_count(t))

    print(f'books: {len(books)} total, {len(todo)} to run, '
          f'{len(skipped)} done, {len(skipped_upstream)} skipped upstream',
          flush=True)
    for title, reason in skipped_upstream.items():
        print(f'  SKIP {title}: {reason}', flush=True)

    results = {}

    def flush_batch():
        for title in skipped:
            results.setdefault(title, {'status': 'ok', 'skipped': True})
        for title, reason in skipped_upstream.items():
            results.setdefault(title, {
                'status': 'skipped_upstream', 'reason': reason,
            })
        atomic_json_dump(batch_path, {
            'time': time.time(), 'src': args.src,
            'model': args.model, 'books': results,
        })

    def run_round(titles, label):
        pool = ThreadPoolExecutor(max_workers=args.parallel_books)
        n = 0
        try:
            futures = {pool.submit(run_one_book, t, args, env): t
                       for t in titles}
            for future in as_completed(futures):
                title, code, elapsed = future.result()
                results[title] = classify(title, code, elapsed)
                n += 1
                print(f'[{label} {n}/{len(titles)}] {title}: '
                      f"{results[title]['status']} ({int(elapsed)}s)",
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
            print(f'  - {title} ({results[title]["status"]})')
    print(f'summary: {batch_path}')
    if children_interrupted:
        sys.exit(130)
    if failed or skipped_upstream:
        sys.exit(1)


if __name__ == '__main__':
    main()
