#!/usr/bin/env python3
"""Batch scheduler for actor.name_anonymization.anonymize.

Processes all .jsonl files in an input directory, running actor.name_anonymization.anonymize
on each and writing results to an output directory.

Usage:
  cd repository-root

  # Anonymize all eval files
  python3 -m actor.name_anonymization.batch \
    --input-dir data/dataset/eval \
    --output-dir data/dataset_anon/eval \
    --model gpt-5.6-luna --workers 8 --resume

  # Anonymize all train files
  python3 -m actor.name_anonymization.batch \
    --input-dir data/dataset/train \
    --output-dir data/dataset_anon/train \
    --model gpt-5.6-luna --workers 8 --resume
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


def record_id(record):
    return record.get('question_id') or record.get('sample_id') or ''


def record_hash(record):
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def load_jsonl(path, audit=False):
    records = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = obj.get('record_id') if audit else record_id(obj)
            if not rid or rid in records:
                raise ValueError(f'missing/duplicate record ID in {path}')
            records[rid] = obj
    return records


def is_done(input_dir, output_dir, filename):
    """Require exact, current input/output/audit record pairing."""
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    base, ext = os.path.splitext(output_path)
    audit_path = base + '.audit' + ext
    if not (os.path.isfile(output_path) and os.path.isfile(audit_path)):
        return False
    try:
        inputs = load_jsonl(input_path)
        outputs = load_jsonl(output_path)
        audits = load_jsonl(audit_path, audit=True)
        ids = set(inputs)
        return (
            set(outputs) == ids == set(audits)
            and all(audits[rid].get('source_hash') == record_hash(inputs[rid])
                    for rid in ids)
        )
    except Exception:
        return False


def count_input_records(input_path):
    """Count lines in input JSONL."""
    try:
        with open(input_path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def run_one_file(filename, args):
    """Run actor.name_anonymization.anonymize for one file. Returns (filename, returncode, elapsed)."""
    input_path = os.path.join(args.input_dir, filename)
    output_path = os.path.join(args.output_dir, filename)
    base, ext = os.path.splitext(output_path)
    audit_path = base + ".audit" + ext

    command = [
        PY, "-u", '-m', 'actor.name_anonymization.anonymize',
        "--input", input_path,
        "--output", output_path,
        "--audit-out", audit_path,
        "--config", args.config,
        "--model", args.model,
        "--max-tokens", str(args.max_tokens),
        "--workers", str(args.workers),
    ]
    if args.max_rpm:
        command += ["--max-rpm",
                    str(max(1, args.max_rpm // args.parallel_files))]
    if args.max_tpm:
        command += ["--max-tpm",
                    str(max(1, args.max_tpm // args.parallel_files))]
    if args.resume:
        command.append("--resume")
    if args.no_cache:
        command.append("--no-cache")

    started = time.time()
    result = subprocess.run(command, cwd=_HERE, capture_output=True, text=True)
    elapsed = time.time() - started

    # Print stdout for progress
    for line in result.stdout.splitlines():
        print(f"  {line}", flush=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}", flush=True)

    return filename, result.returncode, elapsed


def main():
    ap = argparse.ArgumentParser(
        description="Batch scheduler for actor.name_anonymization.anonymize")
    ap.add_argument("--input-dir", required=True,
                    help="Directory with .jsonl files to anonymize")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for anonymized output")
    ap.add_argument("--config", default=os.path.join(_HERE, "config.json"))
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent LLM calls per file")
    ap.add_argument("--max-rpm", type=int, default=None,
                    help="account requests/min, divided among parallel files")
    ap.add_argument("--max-tpm", type=int, default=None,
                    help="account tokens/min, divided among parallel files")
    ap.add_argument("--resume", action="store_true",
                    help="skip records already in output files")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--parallel-files", type=int, default=1,
                    help="process N files in parallel (default 1; "
                         "each file spawns --workers LLM calls)")
    args = ap.parse_args()

    args.input_dir = os.path.abspath(args.input_dir)
    args.config = os.path.abspath(args.config)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # List input files
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".jsonl"))
    print(f"Files: {len(files)} in {args.input_dir}")

    todo, skipped = [], []
    for f in files:
        if args.resume and is_done(args.input_dir, args.output_dir, f):
            skipped.append(f)
        else:
            n = count_input_records(os.path.join(args.input_dir, f))
            todo.append((f, n))

    total_records = sum(n for _, n in todo)
    print(f"To process: {len(todo)} files, {total_records} records total")
    print(f"Skipped (already done): {len(skipped)}")

    results = {}
    pool = ThreadPoolExecutor(max_workers=max(1, args.parallel_files))
    n = 0
    try:
        futures = {pool.submit(run_one_file, f, args): f for f, _ in todo}
        for future in as_completed(futures):
            filename, code, elapsed = future.result()
            state = "ok" if code == 0 else "failed"
            results[filename] = {"status": state, "elapsed_sec": int(elapsed)}
            n += 1
            print(f"[{n}/{len(todo)}] {filename}: {state} ({int(elapsed)}s)", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted, saving partial results", flush=True)
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
