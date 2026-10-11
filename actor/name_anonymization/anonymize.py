#!/usr/bin/env python3
"""LLM-based anonymization of eval questions and/or preference training samples.

Each record is processed independently: the LLM receives the original
record text and is asked to identify ALL person names (full names,
name+patronymic, single names, nicknames, possessives, truncated fragments)
and replace them with "Character 1", "Character 2", ... in order of first
appearance within that record. Same character -> same number within a
record. Place names, animal names, and relational titles (Father/Mother/
brother) are preserved.

Source files are never modified; output is written to a brand-new file.

Reuses action_trajectory/actor.shared.llm for API calls, caching, retry, cost logging.

Works on both eval JSONL (question_id field) and train JSONL (sample_id field).
"""
from actor.shared.paths import ROOT_DIR

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time

_HERE = ROOT_DIR

from actor.shared.llm import LLM  # noqa: E402
from actor.name_anonymization.replacement import (  # noqa: E402
    apply_name_map,
    get_record_id,
    source_hash,
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Identify ALL person names in a character-role-playing dataset record from "{book_title}" and output a name_map.

Output: {{"name_map": {{"<exact name form as it appears in text>": "Character N", ...}}}}

CRITICAL RULES:
1. You MUST list EVERY distinct name form that appears in the text for each character. A single character may appear as: full name ("Alexey Fyodorovitch Karamazov"), name+patronymic ("Alexey Fyodorovitch"), single name ("Alexey"), nickname ("Alyosha"), possessive ("Alyosha's"), or title+name ("Father Zossima"). List EACH form separately as a key, all mapping to the same "Character N".
2. Keys MUST be copied VERBATIM from the text (same spelling, spacing, apostrophes). The script replaces complete name tokens only -- a misspelled key won't match.
3. "Character N": N = order of first appearance; same person -> same N. The "character" field's person is usually Character 1.
4. Do NOT include: place names (Moscow, Tula, Mokroe), animal names (Zhutchka, Perezvon), relational words (father, mother, brother, son, daughter, wife), "Mamma"/"Maman".
5. Include truncated fragments that are clearly part of a name: "Iva" (from "Ivan").

Example output for a record mentioning Alyosha and his father:
{{"name_map": {{"Alexey Fyodorovitch Karamazov": "Character 1", "Alexey Fyodorovitch": "Character 1", "Alexey": "Character 1", "Alyosha": "Character 1", "Alyosha's": "Character 1", "Fyodor Pavlovitch Karamazov": "Character 2", "Fyodor Pavlovitch": "Character 2", "Fyodor": "Character 2"}}}}

After any reasoning, output the final JSON on a new line starting with '{{'. Keep it compact. Do not use markdown fences."""


def build_prompt(record):
    """Build the user prompt for one record."""
    book_title = record.get("book", "the novel")
    system = SYSTEM_PROMPT.format(book_title=book_title)
    return system, (
        "List ALL person name forms (full names, name+patronymic, single names, "
        "nicknames, possessives, truncated fragments) that appear in the "
        "following dataset record. Map each form to 'Character N' "
        "(N = order of first appearance, same person -> same N).\n\n"
        "Return: {\"name_map\": {\"exact name form\": \"Character N\", ...}}\n\n"
        "=== RECORD (original, JSON) ===\n"
        + json.dumps(record, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------

def _scan_json_object(text, start):
    """Scan a JSON object starting at text[start] == '{'. Return parsed dict or None."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start: i + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        return None
    return None


def extract_name_map_json(text):
    """Robustly extract the name map from LLM output.

    Handles chain-of-thought, markdown fences, {"name_map": {...}} wrapper,
    and bare mapping objects.
    """
    # 1. Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 2. Try the wrapped form: {"name_map": {...}}
    for m in re.finditer(r'\{\s*"name_map"\s*:', text):
        obj = _scan_json_object(text, m.start())
        if obj is not None and isinstance(obj.get("name_map"), dict):
            return obj

    # 3. Fallback: LLM returned the mapping object directly.
    for m in re.finditer(r"\{", text):
        obj = _scan_json_object(text, m.start())
        if not isinstance(obj, dict) or not obj:
            continue
        vals = list(obj.values())
        if any(isinstance(v, str) and "Character" in v for v in vals):
            return {"name_map": obj}

    raise ValueError("No name_map JSON object found in LLM response")


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def load_jsonl_by_id(path):
    records = {}
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rid = (record.get("record_id") or get_record_id(record))
                if rid:
                    records[rid] = record
            except Exception:
                pass
    return records


def rewrite_jsonl_atomic(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def process_record(llm, record, model, max_tokens, use_cache=True):
    """Anonymize one record via LLM.

    Returns (anonymized_record, name_map).
    """
    system, user_prompt = build_prompt(record)
    # llm.call() takes a single prompt string; prepend system instructions.
    full_prompt = system + "\n\n" + user_prompt
    raw = llm.call(full_prompt, model=model,
                   max_tokens=max_tokens, json_mode=True,
                   use_cache=use_cache)
    if not raw:
        raise ValueError("Empty LLM response")

    obj = extract_name_map_json(raw)
    name_map = obj.get("name_map") or obj.get("name_replacements") or {}
    if not isinstance(name_map, dict) or not name_map:
        raise ValueError(f"Empty/invalid name_map in LLM response: {raw[:300]!r}")

    anon = apply_name_map(record, name_map)
    return anon, name_map


def process_one_safe(llm, record, model, max_tokens, idx, use_cache=True):
    """Wrap process_record with error handling + logging. Returns dict."""
    rid = get_record_id(record) or f"row_{idx}"
    t0 = time.time()
    try:
        try:
            anon, replacements = process_record(
                llm, record, model, max_tokens, use_cache=use_cache)
        except Exception:
            if not use_cache:
                raise
            # A syntactically/semantically bad response may already have been
            # cached by the transport layer. Retry once without reading or
            # writing that cache so later resume runs can actually recover.
            anon, replacements = process_record(
                llm, record, model, max_tokens, use_cache=False)
        dt = time.time() - t0
        return {
            "idx": idx,
            "record_id": rid,
            "ok": True,
            "anonymized": anon,
            "name_replacements": replacements,
            "elapsed": dt,
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            "idx": idx,
            "record_id": rid,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "elapsed": dt,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Input JSONL (eval or train)")
    ap.add_argument("--output", required=True,
                    help="Output anonymized JSONL")
    ap.add_argument("--audit-out", default=None,
                    help="Audit JSONL (default: {output}.audit.jsonl)")
    ap.add_argument("--config", default=os.path.join(_HERE, "config.json"))
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent LLM calls")
    ap.add_argument("--max-rpm", type=int, default=None)
    ap.add_argument("--max-tpm", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only first N records (0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="skip record IDs already in --output")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass LLM cache (force fresh calls)")
    args = ap.parse_args()

    if args.audit_out is None:
        base, ext = os.path.splitext(args.output)
        args.audit_out = base + ".audit" + ext

    # Load records
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"Loaded {len(records)} records from {args.input}")

    input_by_id = {}
    for record in records:
        rid = get_record_id(record)
        if not rid:
            raise SystemExit("every input record must have question_id or sample_id")
        if rid in input_by_id:
            raise SystemExit(f"duplicate input record ID: {rid}")
        input_by_id[rid] = record

    # A record is resumable only when output and audit both exist and the audit
    # fingerprint matches the current input. Compact first to remove duplicate,
    # orphaned, removed, or stale records left by an interrupted/older run.
    output_by_id = load_jsonl_by_id(args.output) if args.resume else {}
    audit_by_id = load_jsonl_by_id(args.audit_out) if args.resume else {}
    done_ids = {
        rid for rid, record in input_by_id.items()
        if rid in output_by_id and rid in audit_by_id
        and audit_by_id[rid].get("source_hash") == source_hash(record)
    }
    rewrite_jsonl_atomic(args.output,
                         [output_by_id[rid] for rid in input_by_id
                          if rid in done_ids])
    rewrite_jsonl_atomic(args.audit_out,
                         [audit_by_id[rid] for rid in input_by_id
                          if rid in done_ids])
    if args.resume:
        print(f"Resume: {len(done_ids)} complete output+audit pairs; "
              "all other records will be retried.")

    if not records:
        print('Input is empty; wrote empty output and audit files.')
        return

    todo = [(i, r) for i, r in enumerate(records)
            if get_record_id(r) not in done_ids]
    print(f"To process: {len(todo)}")

    if not todo:
        print('All records already complete; no LLM client needed.')
        return

    # Init LLM
    llm = LLM(
        config_path=args.config,
        book=os.path.splitext(os.path.basename(args.input))[0],
        default_model=args.model,
        max_retries=4,
        max_rpm=args.max_rpm,
        max_tpm=args.max_tpm,
    )

    ok_count = 0
    fail_count = 0
    t_start = time.time()

    with open(args.output, "a", encoding="utf-8") as out_f, \
         open(args.audit_out, "a", encoding="utf-8") as audit_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_one_safe, llm, r, args.model,
                            args.max_tokens, i, not args.no_cache): i
                for i, r in todo
            }
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                rid = res["record_id"]
                if res["ok"]:
                    anon = res["anonymized"]
                    orig = records[res["idx"]]
                    # Ensure protected fields are preserved.
                    for k in ("question_id", "sample_id", "book", "entity_id",
                              "action_uid", "scene_uid", "is_critical",
                              "character_action_order", "correct_answer", "split"):
                        if k in orig and k not in anon:
                            anon[k] = orig[k]
                    if "profile" in orig and "profile" in anon:
                        for k in ("version", "freshness", "distance"):
                            if (k in orig["profile"]
                                    and k not in anon.get("profile", {})):
                                anon["profile"][k] = orig["profile"][k]
                    out_f.write(json.dumps(anon, ensure_ascii=False) + "\n")
                    out_f.flush()
                    os.fsync(out_f.fileno())
                    audit_f.write(json.dumps({
                        "record_id": rid,
                        "source_hash": source_hash(orig),
                        "name_replacements": res["name_replacements"],
                        "elapsed": round(res["elapsed"], 2),
                    }, ensure_ascii=False) + "\n")
                    audit_f.flush()
                    os.fsync(audit_f.fileno())
                    ok_count += 1
                else:
                    fail_count += 1
                    print(f"  FAIL [{rid}] {res['error']}", flush=True)
                done = ok_count + fail_count
                if done % 10 == 0 or done == len(todo):
                    elapsed = time.time() - t_start
                    print(f"  progress {done}/{len(todo)}  ok={ok_count} "
                          f"fail={fail_count} elapsed={elapsed:.0f}s  "
                          f"({done/max(elapsed,1):.1f} r/s)", flush=True)

    print(f"\nDone. ok={ok_count} fail={fail_count}")
    print(f"Output: {args.output}")
    print(f"Audit:  {args.audit_out}")
    summary = llm.cost_summary()
    print(f"Cost summary: {summary}")
    if fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
