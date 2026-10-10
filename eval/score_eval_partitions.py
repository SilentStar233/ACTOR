#!/usr/bin/env python3
"""Score existing evaluation results by dataset construction partition.

The final evaluation set contains two semantically different partitions:

* ``split == "all"``: Eval Only books.  No action from these books is used
  for DPO training.
* ``split == "tail"``: Train Eval books.  Earlier actions (the head split)
  are used for training and later actions (the tail split) are used for
  evaluation.

This script joins result rows to the source questions by ``question_id`` and
reports Overall, Eval Only, and Train Eval metrics without running inference
again.  It deliberately fails on incomplete coverage, duplicate IDs, unknown
split labels, or result/question answer mismatches unless ``--allow-errors``
is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from score_eval import compute_stats, load_jsonl, score_results


PARTITION_BY_SPLIT = {
    "all": "eval_only",
    "tail": "train_eval",
}


def load_questions(path: Path) -> tuple[dict[str, dict], dict[str, int]]:
    """Load question metadata and validate the final-dataset split labels."""
    questions: dict[str, dict] = {}
    counts = {partition: 0 for partition in PARTITION_BY_SPLIT.values()}

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row.get("question_id")
            if not question_id:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if question_id in questions:
                raise ValueError(
                    f"{path}:{line_number}: duplicate question_id {question_id!r}"
                )

            split = row.get("split")
            if split not in PARTITION_BY_SPLIT:
                raise ValueError(
                    f"{path}:{line_number}: unsupported split {split!r}; "
                    "expected 'all' (Eval Only) or 'tail' (Train Eval)"
                )
            if row.get("correct_answer") not in {"A", "B", "C", "D"}:
                raise ValueError(
                    f"{path}:{line_number}: invalid correct_answer "
                    f"{row.get('correct_answer')!r}"
                )
            if not isinstance(row.get("is_critical"), bool):
                raise ValueError(f"{path}:{line_number}: is_critical must be boolean")

            partition = PARTITION_BY_SPLIT[split]
            counts[partition] += 1
            questions[question_id] = {
                "question_id": question_id,
                "split": split,
                "dataset_partition": partition,
                "correct_answer": row["correct_answer"],
                "book": row.get("book"),
                "character": row.get("character"),
                "entity_id": row.get("entity_id"),
                "is_critical": bool(row.get("is_critical", False)),
            }

    if not questions:
        raise ValueError(f"Question file is empty: {path}")
    return questions, counts


def join_results(
    results_path: Path,
    questions: dict[str, dict],
    allow_errors: bool,
) -> tuple[list[dict], dict[str, int]]:
    """Join result rows with authoritative question metadata by question_id."""
    results, result_ids = load_jsonl(str(results_path))
    expected_ids = set(questions)
    missing = expected_ids - result_ids
    extra = result_ids - expected_ids
    error_rows = sum(1 for row in results if row.get("error"))

    if missing or extra or error_rows:
        message = (
            f"{results_path}: coverage/errors mismatch: missing={len(missing)}, "
            f"extra={len(extra)}, error_rows={error_rows}"
        )
        if not allow_errors:
            raise ValueError(message)
        print(f"WARNING: {message}", file=sys.stderr)

    joined = []
    answer_mismatches = 0
    for result in results:
        question_id = result["question_id"]
        question = questions.get(question_id)
        if question is None:
            # Extra rows cannot be assigned to either dataset partition.
            continue
        if result.get("correct_answer") != question["correct_answer"]:
            answer_mismatches += 1
            if not allow_errors:
                raise ValueError(
                    f"{results_path}: correct_answer mismatch for {question_id!r}: "
                    f"result={result.get('correct_answer')!r}, "
                    f"question={question['correct_answer']!r}"
                )

        # The question file is authoritative for grouping metadata.  This also
        # makes old result files usable even if they did not store split.
        joined.append(
            {
                **result,
                "correct_answer": question["correct_answer"],
                "book": question["book"],
                "character": question["character"],
                "entity_id": question["entity_id"],
                "is_critical": question["is_critical"],
                "question_split": question["split"],
                "dataset_partition": question["dataset_partition"],
            }
        )

    if answer_mismatches:
        print(
            f"WARNING: {results_path}: {answer_mismatches} answer mismatches "
            "were replaced from the authoritative question file",
            file=sys.stderr,
        )

    diagnostics = {
        "missing": len(missing),
        "extra": len(extra),
        "error_rows": error_rows,
        "answer_mismatches": answer_mismatches,
    }
    return joined, diagnostics


def compact_stats(scored: list[dict]) -> dict:
    """Return the existing full metric set for one group."""
    if not scored:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "valid_answer_count": 0,
            "valid_answer_accuracy": 0.0,
            "no_answer_count": 0,
            "book_count": 0,
            "critical": {"total": 0, "correct": 0, "accuracy": 0.0},
            "non_critical": {"total": 0, "correct": 0, "accuracy": 0.0},
            "per_character": {"mean_accuracy": 0.0, "count": 0},
            "per_character_critical": {"mean_accuracy": 0.0, "count": 0},
            "per_character_noncritical": {"mean_accuracy": 0.0, "count": 0},
            "option_distribution": {},
            "correct_answer_distribution": {},
        }
    stats = compute_stats(scored)
    valid_answer_count = stats["total"] - stats["no_answer_count"]
    stats["valid_answer_count"] = valid_answer_count
    stats["valid_answer_accuracy"] = (
        stats["correct"] / valid_answer_count if valid_answer_count else 0.0
    )
    stats["book_count"] = len({row.get("book") for row in scored})
    return stats


def score_one(
    results_path: Path,
    questions: dict[str, dict],
    question_counts: dict[str, int],
    allow_errors: bool,
) -> tuple[str, dict, list[dict]]:
    joined, diagnostics = join_results(results_path, questions, allow_errors)
    scored = score_results(joined)

    eval_only = [
        row for row in scored if row["dataset_partition"] == "eval_only"
    ]
    train_eval = [
        row for row in scored if row["dataset_partition"] == "train_eval"
    ]

    model_name = (
        (joined[0].get("model") if joined else None)
        or results_path.name.removesuffix(".jsonl")
    )
    ablation = (joined[0].get("ablation") if joined else None) or "full"
    context_variant = (joined[0].get("context_variant") if joined else None)
    label = f"{model_name} [{ablation}]"
    if context_variant and context_variant != "baseline":
        label += f" [{context_variant}]"

    report = {
        "model": model_name,
        "ablation": ablation,
        "context_variant": context_variant,
        "source_results": str(results_path.resolve()),
        "partition_definition": {
            "eval_only": "question split='all'; books never used for DPO training",
            "train_eval": (
                "question split='tail'; earlier actions from the same books "
                "form the DPO training head split"
            ),
        },
        "expected_question_counts": question_counts,
        "coverage": diagnostics,
        "overall": compact_stats(scored),
        "eval_only": compact_stats(eval_only),
        "train_eval": compact_stats(train_eval),
    }
    report["train_eval_minus_eval_only"] = (
        report["train_eval"]["accuracy"] - report["eval_only"]["accuracy"]
    )
    return label, report, scored


def write_scored(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def print_summary(reports: dict[str, dict]) -> None:
    print(
        f"{'Model':<46} | {'Overall':>8} | {'Temporal':>8} | "
        f"{'Cross-work':>10} | {'Key':>8} | {'Non-key':>8} | {'NoAns':>6}"
    )
    print("-" * 120)
    for label, report in reports.items():
        overall = report["overall"]
        print(
            f"{label:<46} | {overall['accuracy']:>7.2%} | "
            f"{report['train_eval']['accuracy']:>7.2%} | "
            f"{report['eval_only']['accuracy']:>9.2%} | "
            f"{overall['critical']['accuracy']:>7.2%} | "
            f"{overall['non_critical']['accuracy']:>7.2%} | "
            f"{overall['no_answer_count']:>6}"
        )
        if not report["train_eval"]["total"] or not report["eval_only"]["total"]:
            print("  Note: an empty partition is represented by total=0, accuracy=0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="Authoritative final question JSONL containing split=all/tail",
    )
    parser.add_argument(
        "--results",
        required=True,
        nargs="+",
        type=Path,
        help="One or more existing model result JSONL files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Combined partition score JSON",
    )
    parser.add_argument(
        "--scored-output-dir",
        type=Path,
        default=None,
        help="Optionally write per-question scored JSONL files with partition labels",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Allow incomplete coverage/error rows (not recommended for final reporting)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions, question_counts = load_questions(args.questions.resolve())
    reports: dict[str, dict] = {}

    for results_path in args.results:
        results_path = results_path.resolve()
        if not results_path.is_file():
            raise FileNotFoundError(f"Result file not found: {results_path}")
        label, report, scored = score_one(
            results_path, questions, question_counts, args.allow_errors
        )
        if label in reports:
            label = f"{label} ({results_path.name})"
        reports[label] = report

        if args.scored_output_dir is not None:
            scored_name = results_path.name.removesuffix(".jsonl") + ".scored.jsonl"
            write_scored(args.scored_output_dir / scored_name, scored)

    output = {
        "questions": str(args.questions.resolve()),
        "question_counts": {
            "overall": len(questions),
            **question_counts,
        },
        "models": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, args.output)

    print_summary(reports)
    print(f"\nPartition scores written to: {args.output}")
    if args.scored_output_dir is not None:
        print(f"Per-question scored files: {args.scored_output_dir}")


if __name__ == "__main__":
    main()
