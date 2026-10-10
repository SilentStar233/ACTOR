#!/usr/bin/env python3
"""Score model responses against correct answers and produce summary tables.

Extracts A/B/C/D from raw model output via regex, compares to ground truth,
and reports accuracy broken down by:
  - Overall
  - Critical vs non-critical
  - Per-character (mean accuracy)
  - Profile freshness (fresh / moderate / stale)
  - Option distribution (check for position bias)

Usage:
  python score_eval.py --results results/model.jsonl --output scores/model.json
  python score_eval.py --results results/*.jsonl --summary summary.json
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(raw_response, strict=False):
    """Extract A/B/C/D from model output.

    Tries several strategies in order:
      1. Exact single letter
      2. "The answer is X" / "I choose X" / "X." patterns
      3. First standalone A/B/C/D in the text
    Returns the letter or None if no answer found.
    """
    if not raw_response:
        return None

    stripped = raw_response.strip()
    if strict:
        return stripped if re.fullmatch(r'[ABCD]', stripped) else None

    text = stripped.upper()

    # Strategy 1: exact single letter
    if re.match(r'^[ABCD]$', text):
        return text[0]

    # Strategy 2: common answer patterns
    patterns = [
        r'(?:ANSWER|ANSWER IS|I CHOOSE|I SELECT|OPTION)\s*(?:IS|:)?\s*([ABCD])',
        r'\b([ABCD])\b\s*[.\)]?\s*$',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)

    # Strategy 3: first standalone A/B/C/D
    m = re.search(r'\b([ABCD])\b', text)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results, strict=False):
    """Score a list of result dicts. Returns scored list + aggregate stats."""
    scored = []
    for r in results:
        if r.get("error"):
            scored.append({**r, "extracted_answer": None, "correct": False, "no_answer": True})
            continue

        extracted = extract_answer(r.get("raw_response"), strict=strict)
        correct = (extracted == r["correct_answer"]) if extracted else False
        scored.append({**r, "extracted_answer": extracted, "correct": correct, "no_answer": extracted is None})

    return scored


def compute_stats(scored):
    """Compute aggregate statistics from scored results."""
    total = len(scored)
    if total == 0:
        return {"total": 0}

    correct = sum(1 for s in scored if s["correct"])
    no_answer = sum(1 for s in scored if s["no_answer"])

    # Critical vs non-critical
    crit = [s for s in scored if s["is_critical"]]
    noncrit = [s for s in scored if not s["is_critical"]]

    crit_acc = sum(1 for s in crit if s["correct"]) / len(crit) if crit else 0
    noncrit_acc = sum(1 for s in noncrit if s["correct"]) / len(noncrit) if noncrit else 0

    # Per-character accuracy
    char_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    for s in scored:
        char_key = (s.get("book"), s.get("entity_id") or s["character"])
        char_scores[char_key]["total"] += 1
        if s["correct"]:
            char_scores[char_key]["correct"] += 1
    char_accs = [v["correct"] / v["total"] for v in char_scores.values() if v["total"] > 0]
    char_mean_acc = sum(char_accs) / len(char_accs) if char_accs else 0

    # Per-character critical accuracy
    char_crit = defaultdict(lambda: {"correct": 0, "total": 0})
    for s in crit:
        char_key = (s.get("book"), s.get("entity_id") or s["character"])
        char_crit[char_key]["total"] += 1
        if s["correct"]:
            char_crit[char_key]["correct"] += 1
    char_crit_accs = [v["correct"] / v["total"] for v in char_crit.values() if v["total"] > 0]
    char_crit_mean = sum(char_crit_accs) / len(char_crit_accs) if char_crit_accs else 0

    # Per-character non-critical accuracy
    char_noncrit = defaultdict(lambda: {"correct": 0, "total": 0})
    for s in noncrit:
        char_key = (s.get("book"), s.get("entity_id") or s["character"])
        char_noncrit[char_key]["total"] += 1
        if s["correct"]:
            char_noncrit[char_key]["correct"] += 1
    char_noncrit_accs = [v["correct"] / v["total"] for v in char_noncrit.values() if v["total"] > 0]
    char_noncrit_mean = sum(char_noncrit_accs) / len(char_noncrit_accs) if char_noncrit_accs else 0

    # Option distribution (model's predicted answers)
    option_dist = defaultdict(int)
    for s in scored:
        if s["extracted_answer"]:
            option_dist[s["extracted_answer"]] += 1
        else:
            option_dist["no_answer"] += 1

    # Correct answer distribution
    correct_dist = defaultdict(int)
    for s in scored:
        correct_dist[s["correct_answer"]] += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "no_answer_count": no_answer,
        "critical": {
            "total": len(crit),
            "correct": sum(1 for s in crit if s["correct"]),
            "accuracy": crit_acc,
        },
        "non_critical": {
            "total": len(noncrit),
            "correct": sum(1 for s in noncrit if s["correct"]),
            "accuracy": noncrit_acc,
        },
        "per_character": {
            "mean_accuracy": char_mean_acc,
            "count": len(char_accs),
        },
        "per_character_critical": {
            "mean_accuracy": char_crit_mean,
            "count": len(char_crit_accs),
        },
        "per_character_noncritical": {
            "mean_accuracy": char_noncrit_mean,
            "count": len(char_noncrit_accs),
        },
        "option_distribution": dict(option_dist),
        "correct_answer_distribution": dict(correct_dist),
    }


def load_jsonl(path):
    """Load result rows and reject duplicate question IDs."""
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if not qid:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if qid in seen:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {qid!r}")
            seen.add(qid)
            rows.append(row)
    return rows, seen


def load_expected_ids(path):
    expected = set()
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            qid = json.loads(line).get("question_id")
            if not qid:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if qid in expected:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {qid!r}")
            expected.add(qid)
    return expected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Score model evaluation results")
    ap.add_argument("--results", required=True, nargs="+",
                    help="One or more result JSONL files")
    ap.add_argument("--output", default=None,
                    help="Output JSON file with scores (default: print to stdout)")
    ap.add_argument("--scored-output", default=None,
                    help="Optional: write per-question scored JSONL")
    ap.add_argument("--questions", default=None,
                    help="Optional source questions JSONL; verify complete coverage")
    ap.add_argument("--allow-errors", action="store_true",
                    help="Return success even when results are incomplete or contain errors")
    ap.add_argument(
        "--strict-answer", action="store_true",
        help="Accept only an exact uppercase A/B/C/D response",
    )
    ap.add_argument("--summary", action="store_true",
                    help="Print a summary table comparing all result files")
    args = ap.parse_args()

    all_stats = {}
    expected_ids = load_expected_ids(args.questions) if args.questions else None
    had_incomplete = False

    for results_path in args.results:
        results, result_ids = load_jsonl(results_path)
        if expected_ids is not None:
            missing = expected_ids - result_ids
            extra = result_ids - expected_ids
            if missing or extra:
                had_incomplete = True
                print(
                    f"WARNING: {results_path} coverage mismatch; "
                    f"missing={len(missing)} extra={len(extra)}",
                    file=sys.stderr,
                )
        errors = sum(1 for r in results if r.get("error"))
        if errors:
            had_incomplete = True
            print(f"WARNING: {results_path} contains {errors} error rows",
                  file=sys.stderr)

        scored = score_results(results, strict=args.strict_answer)
        stats = compute_stats(scored)
        model_name = (results[0].get("model") if results else None) or \
            os.path.basename(results_path).replace(".jsonl", "")
        ablation = (results[0].get("ablation") if results else None) or "full"
        context_variant = results[0].get("context_variant") if results else None
        label = f"{model_name} [{ablation}]"
        if context_variant and context_variant != "baseline":
            label += f" [{context_variant}]"
        if label in all_stats:
            label = f"{label} ({os.path.basename(results_path)})"
        stats["model"] = model_name
        stats["ablation"] = ablation
        stats["context_variant"] = context_variant
        all_stats[label] = stats

        # Write per-question scored output
        if args.scored_output:
            base = os.path.basename(results_path).replace(".jsonl", "")
            if "{model}" in args.scored_output:
                scored_path = args.scored_output.replace("{model}", base)
            elif len(args.results) == 1:
                scored_path = args.scored_output
            else:
                stem, ext = os.path.splitext(args.scored_output)
                scored_path = f"{stem}.{base}{ext or '.jsonl'}"
            os.makedirs(os.path.dirname(os.path.abspath(scored_path)) or ".", exist_ok=True)
            with open(scored_path, "w", encoding="utf-8") as f:
                for s in scored:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Output
    output_data = all_stats
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Scores written to {args.output}")

    # Summary table
    if args.summary or not args.output:
        print(f"\n{'Model':<30} | {'Total':>5} | {'Acc':>6} | {'Crit':>6} | {'NonCrit':>7} | {'CharMean':>8} | {'NoAns':>5}")
        print("-" * 90)
        for model, s in sorted(all_stats.items()):
            print(f"{model:<30} | {s['total']:>5} | {s['accuracy']:>5.1%} | "
                  f"{s['critical']['accuracy']:>5.1%} | {s['non_critical']['accuracy']:>6.1%} | "
                  f"{s['per_character']['mean_accuracy']:>7.1%} | {s['no_answer_count']:>5}")

        # Option distribution
        print(f"\n{'Model':<30} | {'A':>5} | {'B':>5} | {'C':>5} | {'D':>5} | {'NoAns':>5}")
        print("-" * 70)
        for model, s in sorted(all_stats.items()):
            od = s["option_distribution"]
            print(f"{model:<30} | {od.get('A',0):>5} | {od.get('B',0):>5} | "
                  f"{od.get('C',0):>5} | {od.get('D',0):>5} | {od.get('no_answer',0):>5}")

    if had_incomplete and not args.allow_errors:
        raise SystemExit(
            "Incomplete or failed result rows detected; pass --allow-errors "
            "only if you intentionally want partial scores.")


if __name__ == "__main__":
    main()
