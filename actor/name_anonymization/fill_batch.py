#!/usr/bin/env python3
"""Batch scheduler for actor.name_anonymization.fill_variants.

Processes all .jsonl files in an input directory, running actor.name_anonymization.fill_variants
on each with its corresponding .audit.jsonl file.

Usage:
  cd repository-root

  # Fill variants for all anonymized eval files
  python3 -m actor.name_anonymization.fill_batch \
    --input-dir data/dataset_anon/eval \
    --audit-dir data/dataset_anon/eval \
    --output-dir data/dataset_final/eval \
    --books-root data

  # Fill variants for all anonymized train files
  python3 -m actor.name_anonymization.fill_batch \
    --input-dir data/dataset_anon/train \
    --audit-dir data/dataset_anon/train \
    --output-dir data/dataset_final/train \
    --books-root data
"""
from actor.shared.paths import ROOT_DIR

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = ROOT_DIR

PY = sys.executable


def find_audit_file(audit_dir, filename):
    """Find the audit file for a given input filename.

    If input is "A Christmas Carol.jsonl", audit is
    "A Christmas Carol.audit.jsonl" (or ".audit.jsonl" suffix variant).
    """
    base, ext = os.path.splitext(filename)
    if base.endswith(".audit"):
        base = base[:-6]
    candidates = [
        base + ".audit" + ext,
        base + ".audit.jsonl",
        filename.replace(".jsonl", ".audit.jsonl"),
    ]
    for c in candidates:
        path = os.path.join(audit_dir, c)
        if os.path.exists(path):
            return path
    return None


def count_valid_lines(path):
    with open(path, encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_done(input_dir, audit_dir, output_dir, filename):
    """Require output metadata to match the current input and audit files."""
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    audit_path = find_audit_file(audit_dir, filename)
    meta_path = output_path + '.meta.json'
    if not (audit_path and os.path.isfile(output_path)
            and os.path.isfile(meta_path)):
        return False
    try:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        return (
            count_valid_lines(input_path) == count_valid_lines(output_path)
            and meta.get('input_sha256') == file_sha256(input_path)
            and meta.get('audit_sha256') == file_sha256(audit_path)
        )
    except Exception:
        return False


def run_one_file(filename, args):
    """Run actor.name_anonymization.fill_variants for one file. Returns (filename, returncode, elapsed)."""
    input_path = os.path.join(args.input_dir, filename)
    output_path = os.path.join(args.output_dir, filename)
    audit_path = find_audit_file(args.audit_dir, filename)

    if audit_path is None:
        print(f"  SKIP: no audit file for {filename}", flush=True)
        return filename, 1, 0

    command = [
        PY, "-u", '-m', 'actor.name_anonymization.fill_variants',
        "--input", input_path,
        "--audit", audit_path,
        "--output", output_path,
        "--books-root", args.books_root,
    ]

    started = time.time()
    result = subprocess.run(command, cwd=_HERE, capture_output=True, text=True)
    elapsed = time.time() - started

    for line in result.stdout.splitlines():
        print(f"  {line}", flush=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}", flush=True)

    return filename, result.returncode, elapsed


def main():
    ap = argparse.ArgumentParser(
        description="Batch scheduler for actor.name_anonymization.fill_variants")
    ap.add_argument("--input-dir", required=True,
                    help="Directory with anonymized .jsonl files")
    ap.add_argument("--audit-dir", default=None,
                    help="Directory with .audit.jsonl files (default: same as input-dir)")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for filled output")
    ap.add_argument("--books-root", required=True,
                    help="Root dir with contexts/{book}/alias_map.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip files already done in output-dir")
    ap.add_argument("--parallel-files", type=int, default=1)
    args = ap.parse_args()

    if args.audit_dir is None:
        args.audit_dir = args.input_dir

    args.input_dir = os.path.abspath(args.input_dir)
    args.audit_dir = os.path.abspath(args.audit_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.books_root = os.path.abspath(args.books_root)
    os.makedirs(args.output_dir, exist_ok=True)

    # List input files (exclude .audit.jsonl files themselves)
    files = sorted(f for f in os.listdir(args.input_dir)
                   if (f.endswith(".jsonl") and ".audit" not in f
                       and not f.startswith("excluded_")))
    print(f"Files: {len(files)} in {args.input_dir}")

    todo, skipped = [], []
    for f in files:
        if args.resume and is_done(args.input_dir, args.audit_dir,
                                   args.output_dir, f):
            skipped.append(f)
        else:
            todo.append(f)

    print(f"To process: {len(todo)}, skipped: {len(skipped)}")

    results = {}
    pool = ThreadPoolExecutor(max_workers=max(1, args.parallel_files))
    n = 0
    try:
        futures = {pool.submit(run_one_file, f, args): f for f in todo}
        for future in as_completed(futures):
            filename, code, elapsed = future.result()
            state = "ok" if code == 0 else "failed"
            results[filename] = {"status": state, "elapsed_sec": int(elapsed)}
            n += 1
            print(f"[{n}/{len(todo)}] {filename}: {state} ({int(elapsed)}s)", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    pool.shutdown(wait=True)

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    failed = [f for f, r in results.items() if r["status"] == "failed"]
    print(f"\n=== batch summary ===")
    print(f"ok={ok} failed={len(failed)} skipped={len(skipped)}")
    if failed:
        print("failed files:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
