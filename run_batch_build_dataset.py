#!/usr/bin/env python3
"""Batch scheduler for build_dataset.py.

Mirrors run_batch_rewash.py / run_batch_profiles.py architecture.
Since build_dataset.py is pure-local (no LLM calls), this is a simple
parallel wrapper with resume support.

Usage:
  cd pipeline_flat_github

  # Pure eval books
  python3 run_batch_build_dataset.py \
    --books-root data \
    --titles-file eval_only_books.txt \
    --mode all \
    --eval-output-dir data/dataset/eval \
    --resume

  # Train+eval books (split)
  python3 run_batch_build_dataset.py \
    --books-root data \
    --titles-file train_eval_books.txt \
    --mode split --split-ratio 0.2 \
    --eval-output-dir data/dataset/eval \
    --train-output-dir data/dataset/train \
    --resume

  # Pure train books
  python3 run_batch_build_dataset.py \
    --books-root data \
    --titles-file train_only_books.txt \
    --mode train-only \
    --train-output-dir data/dataset/train \
    --resume
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from utils import atomic_json_dump, list_book_titles, safe_title
from build_dataset import (BUILD_META_VERSION, build_input_signature,
                           output_meta_path)

PY = sys.executable


def load_titles(titles_file):
    with open(titles_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def list_book_titles(books_root):
    from utils import list_book_titles as _lbt
    return _lbt(books_root)


def expected_config(args):
    return {
        'split_ratio': args.split_ratio,
        'min_actions': args.min_actions,
        'min_train_actions': args.min_train_actions,
        'max_critical_per_char': args.max_critical_per_char,
        'max_noncritical_per_char': args.max_noncritical_per_char,
        'include_first_action': args.include_first_action,
        'use_rewashed': not args.no_rewashed,
        'seed': args.seed,
    }


def is_done(output_dir, title, output_kind, args, source_signature):
    """Accept empty outputs only when source/config metadata also matches."""
    path = os.path.join(output_dir, safe_title(title) + ".jsonl")
    meta_path = output_meta_path(output_dir, title)
    if not os.path.isfile(path) or not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        with open(path, encoding='utf-8') as f:
            record_count = sum(1 for line in f if line.strip())
        return (
            meta.get('meta_version') == BUILD_META_VERSION
            and meta.get('book') == title
            and meta.get('mode') == args.mode
            and meta.get('output_kind') == output_kind
            and meta.get('source_signature') == source_signature
            and meta.get('config') == expected_config(args)
            and meta.get('record_count') == record_count
        )
    except Exception:
        return False


def run_one_book(title, args):
    """Run build_dataset.py for one book. Returns (title, returncode, elapsed)."""
    command = [PY, "-u", os.path.join(_HERE, "build_dataset.py"),
               "--books-root", args.books_root,
               "--title", title,
               "--mode", args.mode,
               "--seed", str(args.seed),
               "--min-actions", str(args.min_actions),
               "--min-train-actions", str(args.min_train_actions),
               "--max-critical-per-char", str(args.max_critical_per_char),
               "--max-noncritical-per-char", str(args.max_noncritical_per_char)]
    if args.mode == "split":
        command += ["--split-ratio", str(args.split_ratio)]
    command += ["--eval-output-dir", args.eval_output_dir]
    if args.mode in ("split", "train-only"):
        command += ["--train-output-dir", args.train_output_dir]
    if args.no_rewashed:
        command.append("--no-rewashed")
    if args.include_first_action:
        command.append("--include-first-action")

    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}", flush=True)
    else:
        # Print stdout for progress
        for line in result.stdout.splitlines():
            print(f"  {line}", flush=True)
    return title, result.returncode, elapsed


def main():
    ap = argparse.ArgumentParser(
        description="Batch scheduler for build_dataset.py")
    ap.add_argument("--books-root", required=True)
    ap.add_argument("--titles-file", default=None,
                    help="File with one title per line (default: all books)")
    ap.add_argument("--mode", default="all", choices=["all", "split", "train-only"])
    ap.add_argument("--split-ratio", type=float, default=0.2)
    ap.add_argument("--eval-output-dir", default="dataset/eval")
    ap.add_argument("--train-output-dir", default="dataset/train")
    ap.add_argument("--min-actions", type=int, default=20)
    ap.add_argument("--min-train-actions", type=int, default=5)
    ap.add_argument("--max-critical-per-char", type=int, default=15)
    ap.add_argument("--max-noncritical-per-char", type=int, default=5)
    ap.add_argument("--include-first-action", action="store_true")
    ap.add_argument("--no-rewashed", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="Skip books whose output already exists")
    ap.add_argument("--parallel-books", type=int, default=1,
                    help="Parallel books (default 1; build is CPU-only)")
    args = ap.parse_args()
    args.books_root = os.path.abspath(args.books_root)

    # Determine book list
    if args.titles_file:
        books = load_titles(args.titles_file)
    else:
        books = list_book_titles(args.books_root)

    seen_safe_titles = {}
    for title in books:
        safe = safe_title(title)
        if safe in seen_safe_titles and seen_safe_titles[safe] != title:
            raise SystemExit(
                f"safe_title collision: {seen_safe_titles[safe]!r} vs {title!r}")
        seen_safe_titles[safe] = title

    todo, skipped = [], []
    for title in books:
        signature = build_input_signature(args.books_root, title)
        checks = []
        if args.mode in ('all', 'split'):
            checks.append(is_done(args.eval_output_dir, title, 'eval',
                                  args, signature))
        if args.mode in ('split', 'train-only'):
            checks.append(is_done(args.train_output_dir, title, 'train',
                                  args, signature))
        if args.resume and checks and all(checks):
            skipped.append(title)
        else:
            todo.append(title)

    print(f"books: {len(books)} total, {len(todo)} to run, "
          f"{len(skipped)} done", flush=True)

    results = {}
    batch_path = os.path.join(
        args.books_root, "logs", f"build_dataset_batch_{args.mode}.json")
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)

    def flush_batch():
        atomic_json_dump(batch_path, {
            "time": time.time(), "mode": args.mode, "books": results,
        })

    # Run (parallelizable but build is fast and CPU-only, default 1)
    pool = ThreadPoolExecutor(max_workers=max(1, args.parallel_books))
    n = 0
    try:
        futures = {pool.submit(run_one_book, t, args): t for t in todo}
        for future in as_completed(futures):
            title, code, elapsed = future.result()
            state = "ok" if code == 0 else "failed"
            results[title] = {"status": state, "elapsed_sec": int(elapsed)}
            n += 1
            print(f"[{n}/{len(todo)}] {title}: {state} ({int(elapsed)}s)", flush=True)
            flush_batch()
    except KeyboardInterrupt:
        print("\ninterrupted, saving partial batch", flush=True)
    pool.shutdown(wait=True)

    for title in skipped:
        results.setdefault(title, {"status": "ok", "skipped": True})
    flush_batch()

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    failed = [t for t, r in results.items() if r["status"] == "failed"]
    print(f"\n=== batch summary ===")
    print(f"ok={ok} failed={len(failed)} skipped={len(skipped)}")
    if failed:
        print("failed books:")
        for title in failed:
            print(f"  - {title}")
    print(f"summary: {batch_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
