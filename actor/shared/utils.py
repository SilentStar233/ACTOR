# Shared utilities for the action_trajectory pipeline.
import json
import hashlib
import os
import re
import threading

_ILLEGAL_PATH_CHARS_RE = re.compile(r'[/\\\x00-\x1f]')


def atomic_json_dump(path, payload, indent=2):
    """Write JSON atomically: unique tmp file + fsync + os.replace.

    The tmp name carries pid/thread-id so concurrent writers never share
    a tmp file; fsync guards against power-loss empty files.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_title(title):
    """Make a book title safe for use in file/directory names.

    Only filesystem-illegal characters ('/', '\\', control chars) are
    replaced with '_'; spaces and unicode punctuation are preserved so
    existing artifacts (e.g. 'King Lear.plots.json') keep their paths.
    """
    return _ILLEGAL_PATH_CHARS_RE.sub('_', title).strip()


def resolve_book_dirs(books_root, book):
    """Resolve timeline/output/profile directories for supported layouts."""
    safe = safe_title(book)
    cand_a = os.path.join(books_root, safe, 'character_timelines')
    cand_b = os.path.join(books_root, 'character_timelines', safe)
    if os.path.isdir(cand_a):
        return (cand_a,
                os.path.join(books_root, safe, 'character_timelines_rewashed'),
                os.path.join(books_root, safe, 'profiles'))
    if os.path.isdir(cand_b):
        return (cand_b,
                os.path.join(books_root, 'character_timelines_rewashed', safe),
                os.path.join(books_root, 'profile_trials', safe))
    return None, None, None


def list_book_titles(books_root):
    """List original book titles from either supported timeline layout."""
    layout_b = os.path.join(books_root, 'character_timelines')
    if os.path.isdir(layout_b):
        titles = []
        for dirname in sorted(os.listdir(layout_b)):
            book_dir = os.path.join(layout_b, dirname)
            if not os.path.isdir(book_dir):
                continue
            title = dirname
            try:
                with open(os.path.join(book_dir, 'index.json'),
                          encoding='utf-8') as f:
                    index = json.load(f)
                title = index.get('book') or index.get('title') or dirname
            except Exception:
                pass
            titles.append(title)
        return titles
    return sorted(
        dirname for dirname in os.listdir(books_root)
        if os.path.isdir(os.path.join(
            books_root, dirname, 'character_timelines')))


def tree_signature(path):
    """Stable metadata signature for detecting changes in an artifact tree."""
    digest = hashlib.sha256()
    if not os.path.exists(path):
        digest.update(b'MISSING')
        return digest.hexdigest()
    if os.path.isfile(path):
        paths = [path]
        root = os.path.dirname(path)
    else:
        root = path
        paths = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames.sort()
            paths.extend(os.path.join(dirpath, name)
                         for name in sorted(filenames))
    for child in paths:
        stat = os.stat(child)
        rel = os.path.relpath(child, root)
        digest.update(
            f'{rel}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode('utf-8'))
    return digest.hexdigest()


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
