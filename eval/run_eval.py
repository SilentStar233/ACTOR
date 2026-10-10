#!/usr/bin/env python3
"""Unified letter evaluation using an existing OpenAI-compatible service.

The service loads the model and adapter; no model names are guessed here.
Results, a credential-free protocol manifest, scores and partitions are saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset_prompts import CONTEXT_VARIANTS, build_letter_prompt
from score_eval import compute_stats, load_jsonl, score_results
from score_eval_partitions import load_questions, print_summary, score_one


def build_messages(question, ablation="no_book", context_variant="baseline"):
    return [{"role": "user", "content": build_letter_prompt(
        question, ablation, context_variant or "baseline")}]


def call_model(client, model, messages, max_tokens=4, temperature=0.0,
               seed=0, guided_choice=None, disable_thinking=False, max_retries=3):
    """Retry transient errors without silently dropping protocol parameters."""
    params = {"model": model, "messages": messages,
              "max_completion_tokens": max_tokens}
    if temperature is not None:
        params["temperature"] = temperature
    if seed is not None:
        params["seed"] = seed
    extra = {}
    if guided_choice:
        extra["guided_choice"] = guided_choice
    if disable_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if extra:
        params["extra_body"] = extra
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(**params)
            if not completion.choices:
                raise ValueError("Model returned no choices")
            content = completion.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Model returned an empty response")
            return content.strip()
        except (openai.APIError, ValueError) as error:
            status = getattr(error, "status_code", None)
            if attempt == max_retries - 1 or status in (400, 401, 403, 404, 422):
                raise
            wait = (5 if isinstance(error, openai.RateLimitError) else 1) * 2 ** attempt
            # Exception bodies may echo a whole prompt; do not dump them.
            print(f"Request retry {attempt + 1}/{max_retries}: "
                  f"{type(error).__name__}; waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Request exhausted all retries")


def load_existing(output_path):
    """Keep the latest row; tolerate only a truncated final JSONL fragment."""
    existing = {}
    if not os.path.exists(output_path):
        return existing
    with open(output_path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    print(f"Ignoring truncated final result line {number}", file=sys.stderr)
                    continue
                raise ValueError(f"Corrupt result JSON at line {number}")
            if not isinstance(row, dict) or not row.get("question_id"):
                raise ValueError(f"Invalid result row at line {number}")
            existing[row["question_id"]] = row
    return existing


def atomic_write_results(output_path, rows):
    temporary = f"{output_path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_protocol(args, base_url):
    endpoint = urlsplit(base_url)
    if (endpoint.scheme not in ("http", "https") or not endpoint.netloc
            or endpoint.username is not None or endpoint.password is not None
            or endpoint.query or endpoint.fragment):
        raise ValueError("Use an HTTP(S) base URL without credentials, query or fragment; "
                         "pass credentials through --api-key or OPENAI_API_KEY.")
    return {
        "version": 1,
        "questions_sha256": file_sha256(args.questions),
        "prompt_sha256": file_sha256(_HERE / "dataset_prompts.py"),
        "backend": args.backend,
        "base_url": base_url.rstrip("/"),
        "model": args.model,
        "ablation": args.ablation,
        "context_variant": args.context_variant,
        "max_output_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "guided_choice": args.guided_choice,
        "thinking_enabled": False if args.disable_thinking else None,
    }


def verify_run_config(path, protocol, output_exists, resume):
    path = Path(path)
    if output_exists and not resume:
        raise ValueError("Output already exists. Use --resume or a new --output path.")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("protocol") != protocol:
            changed = [key for key in protocol
                       if saved.get("protocol", {}).get(key) != protocol[key]]
            raise ValueError("Run configuration mismatch: " + ", ".join(changed)
                             + ". Use a new output path.")
    elif output_exists:
        raise ValueError("Existing results lack a verified run configuration. "
                         "Use a new output path; score historical files separately.")
    else:
        atomic_write_json(path, {"protocol": protocol})


def validate_questions(questions):
    if not questions:
        raise ValueError("Question file is empty")
    seen = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Question rows must be objects")
        qid = question.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError("Missing string question_id")
        if qid in seen:
            raise ValueError(f"Duplicate question_id: {qid}")
        seen.add(qid)
        if (not isinstance(question.get("book"), str)
                or not isinstance(question.get("character"), str)
                or not isinstance(question.get("is_critical"), bool)
                or question.get("correct_answer") not in ("A", "B", "C", "D")):
            raise ValueError(f"{qid}: missing or invalid evaluation metadata")
        options = question.get("options")
        if (not isinstance(options, list) or len(options) != 4
                or any(not isinstance(option, dict)
                       or option.get("id") not in ("A", "B", "C", "D")
                       or not isinstance(option.get("text"), str)
                       or not option["text"].strip() for option in options)
                or {option["id"] for option in options} != set("ABCD")):
            raise ValueError(f"{qid}: expected four non-empty A/B/C/D options")


def partitions_available(questions, mode):
    if mode == "off":
        return False
    if all(q.get("split") in ("all", "tail") for q in questions):
        return True
    if mode == "required" or any(q.get("split") is not None for q in questions):
        raise ValueError("Partition scoring requires split='all'/'tail' on every "
                         "question; fix metadata or explicitly use --partition-mode off.")
    print("Partition scoring skipped: no split metadata. Overall/Key/Non-key "
          "scores will still be generated.", file=sys.stderr)
    return False


def score_output(args, questions, include_partitions):
    """Preserve the original parser, joining authoritative question metadata."""
    results, result_ids = load_jsonl(args.output)
    expected = {q["question_id"]: q for q in questions}
    if result_ids != set(expected):
        raise ValueError("Cannot score output with missing or extra question IDs")
    joined = []
    for row in results:
        question = expected[row["question_id"]]
        if row.get("correct_answer") != question["correct_answer"]:
            raise ValueError(f"Gold answer mismatch: {row['question_id']}")
        joined.append({**row, **{key: question.get(key) for key in
                      ("book", "character", "entity_id", "is_critical")}})
    stats = compute_stats(score_results(joined))
    stats.update(model=args.model, ablation=args.ablation,
                 context_variant=args.context_variant,
                 error_rows=sum(bool(row.get("error")) for row in results))
    label = f"{args.model} [{args.ablation}]"
    if args.context_variant != "baseline":
        label += f" [{args.context_variant}]"
    atomic_write_json(args.scores, {label: stats})
    print(f"Overall: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%}); "
          f"No-answer: {stats['no_answer_count']}; Request errors: {stats['error_rows']}")
    print(f"Options: {stats['option_distribution']}")
    print(f"Scores: {args.scores}")
    if include_partitions:
        metadata, counts = load_questions(Path(args.questions))
        label, report, _ = score_one(Path(args.output), metadata, counts,
                                     allow_errors=bool(stats["error_rows"]))
        atomic_write_json(args.partition_scores, {
            "questions": str(Path(args.questions).resolve()),
            "question_counts": {"overall": len(metadata), **counts},
            "models": {label: report},
        })
        print_summary({label: report})
        print(f"Partition scores: {args.partition_scores}")
    if stats["error_rows"]:
        print("WARNING: diagnostic scores contain failed requests; resume before "
              "using these as final results.", file=sys.stderr)
    return stats["error_rows"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", default="openai", choices=["openai"],
                    help="Existing OpenAI-compatible API, including vLLM")
    ap.add_argument("--model", required=True, help="Model/adapter name exposed by the server")
    ap.add_argument("--api-key", default=None, help="Default: OPENAI_API_KEY")
    ap.add_argument("--base-url", default=None, help="Default: OPENAI_BASE_URL")
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--omit-temperature", action="store_true",
                    help="Explicitly omit temperature for APIs that reject it")
    ap.add_argument("--omit-seed", action="store_true",
                    help="Explicitly omit seed for APIs that reject it")
    ap.add_argument("--guided-choice", nargs="+", default=None,
                    help="Opt-in vLLM extension: --guided-choice A B C D")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Send chat_template_kwargs.enable_thinking=false; requires server support")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ablation", default="no_book", choices=["full", "no_book",
                    "no_long_memory", "no_memory", "no_profile", "no_both"],
                    help="Book naming protocol; legacy context values remain supported")
    ap.add_argument("--context-variant", default="baseline", choices=CONTEXT_VARIANTS)
    ap.add_argument("--scores", help="Default: <output stem>.score.json")
    ap.add_argument("--partition-scores", help="Default: <output stem>.partition-score.json")
    ap.add_argument("--partition-mode", default="auto", choices=["auto", "required", "off"])
    args = ap.parse_args()
    if args.max_tokens < 1 or args.workers < 1:
        ap.error("--max-tokens and --workers must be positive")
    if args.temperature < 0:
        ap.error("--temperature must be non-negative")
    if args.omit_temperature:
        args.temperature = None
    if args.omit_seed:
        args.seed = None
    if args.guided_choice is not None:
        if len(args.guided_choice) != 4 or set(args.guided_choice) != set("ABCD"):
            ap.error("Letter evaluation requires --guided-choice A B C D")
    if args.ablation not in ("full", "no_book"):
        warnings.warn("Legacy --ablation context semantics: no_memory removes only "
                      "recent memory. Prefer --context-variant for new experiments.")
        if args.context_variant != "baseline":
            ap.error("Do not combine legacy ablations with --context-variant")
    stem = str(Path(args.output).with_suffix(""))
    args.scores = args.scores or stem + ".score.json"
    args.partition_scores = args.partition_scores or stem + ".partition-score.json"
    targets = [args.questions, args.output, args.scores, args.partition_scores,
               args.output + ".eval-config.json", args.output + ".lock"]
    if len({os.path.normcase(str(Path(p).resolve())) for p in targets}) != len(targets):
        ap.error("Question, result, score, manifest and lock paths must be distinct")
    return args


def run(args):
    with open(args.questions, encoding="utf-8") as stream:
        questions = [json.loads(line) for line in stream if line.strip()]
    validate_questions(questions)
    include_partitions = partitions_available(questions, args.partition_mode)
    for question in questions:
        build_messages(question, args.ablation, args.context_variant)
    print(f"Loaded {len(questions)} questions from {args.questions}")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    protocol = make_protocol(args, base_url)
    verify_run_config(args.output + ".eval-config.json", protocol,
                      os.path.exists(args.output), args.resume)
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    existing = load_existing(args.output) if args.resume else {}
    question_map = {q["question_id"]: q for q in questions}
    for qid, row in existing.items():
        if (qid not in question_map or row.get("model") != args.model
                or row.get("ablation") != args.ablation
                or row.get("context_variant") != args.context_variant
                or row.get("correct_answer") != question_map[qid]["correct_answer"]
                or row.get("run_id") != run_id):
            raise ValueError("Incompatible result rows; use a new output path")
    completed = {qid for qid, row in existing.items()
                 if not row.get("error") and row.get("raw_response")}
    pending = [q for q in questions if q["question_id"] not in completed]
    print(f"Resume: {len(completed)} answered; pending: {len(pending)}")
    if pending:
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY or pass --api-key (EMPTY for local vLLM)")
        client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

        def process_one(question):
            try:
                response = call_model(client, args.model,
                    build_messages(question, args.ablation, args.context_variant),
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    seed=args.seed, guided_choice=args.guided_choice,
                    disable_thinking=args.disable_thinking)
                return question, response, None
            except Exception as error:
                status = getattr(error, "status_code", None)
                detail = type(error).__name__
                if status is not None:
                    detail += f" (HTTP {status})"
                if isinstance(error, ValueError):
                    detail += f": {error}"
                return question, None, detail

        try:
            # Repair truncated final lines BEFORE appending and deduplicate.
            atomic_write_results(args.output, existing.values())
            with open(args.output, "a", encoding="utf-8", newline="\n") as output:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(process_one, q) for q in pending]
                    failed = 0
                    for done, future in enumerate(as_completed(futures), 1):
                        question, response, error = future.result()
                        failed += bool(error)
                        row = {key: question.get(key) for key in
                               ("question_id", "book", "character", "entity_id",
                                "is_critical", "correct_answer")}
                        row.update(model=args.model, ablation=args.ablation,
                                   context_variant=args.context_variant, run_id=run_id,
                                   max_output_tokens=args.max_tokens,
                                   temperature=args.temperature, seed=args.seed,
                                   guided_choice=args.guided_choice,
                                   thinking_enabled=protocol["thinking_enabled"],
                                   raw_response=response, error=error)
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        if done % 50 == 0 or done == len(pending):
                            print(f"[{done}/{len(pending)}] done; {failed} failed", flush=True)
        finally:
            client.close()
    latest = load_existing(args.output)
    atomic_write_results(args.output, [latest[q["question_id"]] for q in questions])
    # A fully resumed run also regenerates scores.
    return score_output(args, questions, include_partitions)


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    lock = Path(args.output + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit(f"Output is locked: {lock}. If a job was killed, confirm "
                         "no writer is running before removing this lock file.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        failed = run(args)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
    finally:
        lock.unlink()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
