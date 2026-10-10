#!/usr/bin/env python3
"""Validate and prepare character-action preference pairs for RPO training.

The script deliberately refuses to truncate examples.  With the default
``--overlength-policy error`` it writes an audit report and exits non-zero if
any chosen/rejected sequence exceeds the requested token limits.  Re-run with
``--overlength-policy drop`` only after reviewing ``overlength.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from transformers import AutoTokenizer


REQUIRED_FIELDS = {
    "sample_id",
    "book",
    "character",
    "entity_id",
    "action_uid",
    "is_critical",
    "scene_uid",
    "character_action_order",
    "split",
    "profile",
    "recent_memory",
    "long_term_memory",
    "scene",
    "chosen",
    "rejected",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt-module", type=Path, required=True)
    parser.add_argument("--ablation", choices=["no_book"], default="no_book")
    parser.add_argument(
        "--context-variant",
        default=None,
        help=(
            "Optional prompt-module-specific context variant. When supplied, "
            "build_context_prompt must accept a context_variant keyword argument."
        ),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=7168)
    parser.add_argument(
        "--overlength-policy",
        choices=["error", "drop"],
        default="error",
        help="Never truncates. 'error' audits and stops; 'drop' excludes listed rows.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="Optional exact source-row count check; 0 disables it.",
    )
    parser.add_argument(
        "--expected-file-count",
        type=int,
        default=0,
        help="Optional exact source-file count check; 0 disables it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Prepare only the first N rows for a dry run; disables expected-count.",
    )
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    if not 0.0 <= args.validation_ratio < 1.0:
        parser.error("--validation-ratio must be in [0, 1)")
    if args.max_prompt_length > args.max_length:
        parser.error("--max-prompt-length cannot exceed --max-length")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompt_builder(path: Path) -> Callable[..., str]:
    spec = importlib.util.spec_from_file_location("authoritative_dataset_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_context_prompt", None)
    if not callable(builder):
        raise RuntimeError(f"build_context_prompt is missing from {path}")
    return builder


def build_prompt(
    builder: Callable[..., str],
    row: dict[str, Any],
    ablation: str,
    context_variant: str | None,
) -> str:
    """Call old and new prompt modules without changing legacy behavior."""
    if context_variant is None:
        return builder(row, ablation=ablation)
    parameters = inspect.signature(builder).parameters
    supports_variant = (
        "context_variant" in parameters
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    )
    if not supports_variant:
        raise RuntimeError(
            f"Prompt module does not support --context-variant={context_variant!r}"
        )
    return builder(
        row,
        ablation=ablation,
        context_variant=context_variant,
    )


def stable_is_validation(sample_id: str, seed: int, ratio: float) -> bool:
    if ratio == 0.0:
        return False
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < ratio


def percentile(values: list[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def length_summary(values: list[int]) -> dict[str, int | None]:
    return {
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "p99_9": percentile(values, 0.999),
        "max": max(values) if values else None,
    }


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def dump_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def write_jsonl(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")


def iter_source_rows(files: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                yield path, line_number, row


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    prompt_module = args.prompt_module.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not model_path.is_dir():
        raise SystemExit(f"Model directory not found: {model_path}")
    if not prompt_module.is_file():
        raise SystemExit(f"Prompt module not found: {prompt_module}")

    source_files = sorted(input_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not source_files:
        raise SystemExit(f"No JSONL files found under {input_dir}")
    if args.expected_file_count and len(source_files) != args.expected_file_count:
        raise SystemExit(
            f"Expected {args.expected_file_count} JSONL files but found "
            f"{len(source_files)} under {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    overlength_path = output_dir / "overlength.jsonl"
    manifest_path = output_dir / "manifest.json"
    train_tmp = output_dir / "train.jsonl.tmp"
    validation_tmp = output_dir / "validation.jsonl.tmp"
    overlength_tmp = output_dir / "overlength.jsonl.tmp"

    build_context_prompt = load_prompt_builder(prompt_module)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.chat_template is None:
        raise SystemExit("The selected tokenizer has no chat_template")

    source_hashes = []
    combined_hash = hashlib.sha256()
    print(f"Hashing {len(source_files)} source files...", flush=True)
    for path in source_files:
        file_hash = sha256_file(path)
        relative_name = path.relative_to(input_dir).as_posix()
        source_hashes.append({"file": relative_name, "sha256": file_hash})
        combined_hash.update(relative_name.encode("utf-8"))
        combined_hash.update(b"\0")
        combined_hash.update(file_hash.encode("ascii"))
        combined_hash.update(b"\n")

    seen_ids: set[str] = set()
    split_values: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    chosen_total_lengths: list[int] = []
    rejected_total_lengths: list[int] = []
    chosen_completion_lengths: list[int] = []
    rejected_completion_lengths: list[int] = []
    source_count = train_count = validation_count = overlength_count = 0
    dropped_count = 0

    try:
        with (
            train_tmp.open("w", encoding="utf-8", newline="\n") as train_stream,
            validation_tmp.open("w", encoding="utf-8", newline="\n") as validation_stream,
            overlength_tmp.open("w", encoding="utf-8", newline="\n") as over_stream,
        ):
            for path, line_number, row in iter_source_rows(source_files):
                if args.limit is not None and source_count >= args.limit:
                    break
                source_count += 1

                missing = sorted(REQUIRED_FIELDS - row.keys())
                if missing:
                    raise ValueError(f"Missing fields {missing} at {path}:{line_number}")
                sample_id = row["sample_id"]
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"Invalid sample_id at {path}:{line_number}")
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample_id {sample_id!r} at {path}:{line_number}")
                seen_ids.add(sample_id)
                # The authoritative directory is already the final training
                # partition.  Its records currently carry split="head";
                # do not reject them merely because the metadata label differs
                # from the directory name.  Record observed labels in manifest
                # so an accidental mixture is still visible.
                split_values[str(row.get("split"))] += 1
                chosen = row["chosen"]
                rejected = row["rejected"]
                if not isinstance(chosen, str) or not chosen.strip():
                    raise ValueError(f"Empty/non-string chosen for {sample_id!r}")
                if not isinstance(rejected, str) or not rejected.strip():
                    raise ValueError(f"Empty/non-string rejected for {sample_id!r}")
                if chosen.strip() == rejected.strip():
                    raise ValueError(f"chosen equals rejected for {sample_id!r}")

                prompt_text = build_prompt(
                    build_context_prompt,
                    row,
                    ablation=args.ablation,
                    context_variant=args.context_variant,
                )
                if not isinstance(prompt_text, str) or not prompt_text.strip():
                    raise ValueError(f"Prompt builder returned an empty prompt for {sample_id!r}")
                if args.ablation == "no_book" and 'in the novel "' in prompt_text:
                    raise ValueError(f"no_book prompt leaked the book clause for {sample_id!r}")

                prompt_messages = [{"role": "user", "content": prompt_text}]
                chosen_messages = [{"role": "assistant", "content": chosen}]
                rejected_messages = [{"role": "assistant", "content": rejected}]
                prompt_ids = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True
                )
                chosen_ids = tokenizer.apply_chat_template(
                    prompt_messages + chosen_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                rejected_ids = tokenizer.apply_chat_template(
                    prompt_messages + rejected_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                chosen_prefix = common_prefix_length(prompt_ids, chosen_ids)
                rejected_prefix = common_prefix_length(prompt_ids, rejected_ids)
                if chosen_prefix != len(prompt_ids) or rejected_prefix != len(prompt_ids):
                    raise ValueError(
                        "Chat-template prompt is not an exact prefix of both completions "
                        f"for {sample_id!r}; cannot audit preference-pair lengths safely"
                    )

                prompt_tokens = len(prompt_ids)
                chosen_total_tokens = len(chosen_ids)
                rejected_total_tokens = len(rejected_ids)
                chosen_completion_tokens = chosen_total_tokens - prompt_tokens
                rejected_completion_tokens = rejected_total_tokens - prompt_tokens
                prompt_lengths.append(prompt_tokens)
                chosen_total_lengths.append(chosen_total_tokens)
                rejected_total_lengths.append(rejected_total_tokens)
                chosen_completion_lengths.append(chosen_completion_tokens)
                rejected_completion_lengths.append(rejected_completion_tokens)

                reasons = []
                if prompt_tokens > args.max_prompt_length:
                    reasons.append("prompt_exceeds_max_prompt_length")
                if chosen_total_tokens > args.max_length:
                    reasons.append("chosen_exceeds_max_length")
                if rejected_total_tokens > args.max_length:
                    reasons.append("rejected_exceeds_max_length")
                is_overlength = bool(reasons)
                if is_overlength:
                    overlength_count += 1
                    write_jsonl(
                        over_stream,
                        {
                            "sample_id": sample_id,
                            "source_file": path.relative_to(input_dir).as_posix(),
                            "source_line": line_number,
                            "book": row["book"],
                            "character": row["character"],
                            "reasons": reasons,
                            "prompt_tokens": prompt_tokens,
                            "chosen_completion_tokens": chosen_completion_tokens,
                            "rejected_completion_tokens": rejected_completion_tokens,
                            "chosen_total_tokens": chosen_total_tokens,
                            "rejected_total_tokens": rejected_total_tokens,
                            "chosen_chars": len(chosen),
                            "rejected_chars": len(rejected),
                        },
                    )
                    if args.overlength_policy in {"error", "drop"}:
                        dropped_count += 1
                        continue

                prepared = {
                    "prompt": prompt_messages,
                    "chosen": chosen_messages,
                    "rejected": rejected_messages,
                    "sample_id": sample_id,
                    "book": row["book"],
                    "character": row["character"],
                    "is_critical": bool(row["is_critical"]),
                    "prompt_tokens": prompt_tokens,
                    "chosen_completion_tokens": chosen_completion_tokens,
                    "rejected_completion_tokens": rejected_completion_tokens,
                    "chosen_total_tokens": chosen_total_tokens,
                    "rejected_total_tokens": rejected_total_tokens,
                }
                if stable_is_validation(sample_id, args.split_seed, args.validation_ratio):
                    write_jsonl(validation_stream, prepared)
                    validation_count += 1
                else:
                    write_jsonl(train_stream, prepared)
                    train_count += 1

                if args.progress_every and source_count % args.progress_every == 0:
                    print(
                        f"Checked {source_count:,} rows; overlength={overlength_count:,}",
                        flush=True,
                    )

        os.replace(overlength_tmp, overlength_path)

        expected_count = 0 if args.limit is not None else args.expected_count
        if expected_count and source_count != expected_count:
            raise ValueError(
                f"Expected {expected_count:,} source rows but found {source_count:,}"
            )

        failed_for_overlength = (
            args.overlength_policy == "error" and overlength_count > 0
        )
        status = "failed_overlength" if failed_for_overlength else "ready"
        manifest = {
            "status": status,
            "ablation": args.ablation,
            "context_variant": args.context_variant,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "model_path": str(model_path),
            "tokenizer_class": tokenizer.__class__.__name__,
            "prompt_module": str(prompt_module),
            "prompt_module_sha256": sha256_file(prompt_module),
            "source_file_count": len(source_files),
            "source_row_count": source_count,
            "source_files": source_hashes,
            "source_dataset_fingerprint": combined_hash.hexdigest(),
            "source_split_values": dict(sorted(split_values.items())),
            "validation_ratio": args.validation_ratio,
            "split_seed": args.split_seed,
            "limits": {
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "overlength_policy": args.overlength_policy,
            },
            "counts": {
                "train": train_count if not failed_for_overlength else 0,
                "validation": validation_count if not failed_for_overlength else 0,
                "overlength": overlength_count,
                "dropped": dropped_count if args.overlength_policy == "drop" else 0,
            },
            "token_lengths": {
                "prompt": length_summary(prompt_lengths),
                "chosen_completion": length_summary(chosen_completion_lengths),
                "rejected_completion": length_summary(rejected_completion_lengths),
                "chosen_total": length_summary(chosen_total_lengths),
                "rejected_total": length_summary(rejected_total_lengths),
            },
            "limit": args.limit,
        }
        dump_json_atomic(manifest_path, manifest)

        if failed_for_overlength:
            train_tmp.unlink(missing_ok=True)
            validation_tmp.unlink(missing_ok=True)
            print(
                f"AUDIT FAILED: {overlength_count:,} rows exceed the explicit token limits.",
                file=sys.stderr,
            )
            print(f"Review: {overlength_path}", file=sys.stderr)
            print(
                "No train/validation files were replaced. Choose new limits or re-run "
                "with --overlength-policy drop after reviewing the rows.",
                file=sys.stderr,
            )
            return 2

        os.replace(train_tmp, train_path)
        os.replace(validation_tmp, validation_path)
        print(
            f"READY: source={source_count:,}, train={train_count:,}, "
            f"validation={validation_count:,}, dropped={dropped_count:,}"
        )
        print(f"Manifest: {manifest_path}")
        return 0
    except Exception:
        train_tmp.unlink(missing_ok=True)
        validation_tmp.unlink(missing_ok=True)
        overlength_tmp.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
