#!/usr/bin/env python3
"""Post-process LLM-anonymized dataset: fill in name variants the LLM missed.

Works for ANY book. For each record, the book name is read from the
record's "book" field, and the corresponding alias_map.json is loaded
from {books_root}/contexts/{book}/alias_map.json (Layout B).

Works on both eval JSONL (question_id) and train JSONL (sample_id).

Strategy:
  1. Group records by book; load each book's alias_map.json.
  2. For each anonymized record, build a reverse map: canonical name ->
     "Character N" by matching the LLM's name_map keys against known aliases.
  3. For canonical names that got a Character N, collect ALL their aliases
     (from alias_map) plus derived forms (2-part, single name, possessive)
     and do a second-pass literal replacement on the record.
  4. Fix half-replaced leftovers ("Character N <dangling surname/patronymic>")
     using fragments extracted dynamically from the book's alias_map -- no
     hardcoded language-specific suffixes.

Protected fields (question_id, sample_id, entity_id, etc.) are never touched.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

from actor.shared.utils import safe_title
from actor.name_anonymization.replacement import compile_replacer, replace_record_strings


def norm_quotes(s):
    """Normalize curly/straight apostrophes and quotes for matching."""
    return (s.replace("\u2019", "'").replace("\u2018", "'")
             .replace("\u201c", '"').replace("\u201d", '"'))


def get_record_id(record):
    """Get the unique ID of a record (question_id for eval, sample_id for train)."""
    return record.get("question_id") or record.get("sample_id") or ""


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_jsonl(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Per-book alias data (loaded lazily, cached)
# ---------------------------------------------------------------------------

_book_cache = {}


def artifact_title_candidates(book_name):
    """Possible artifact names across Linux and Windows/legacy exports."""
    candidates = [safe_title(book_name)]
    portable = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', book_name).strip()
    if portable not in candidates:
        candidates.append(portable)
    return candidates


def find_support_path(books_root, book_name, kind):
    """Resolve alias/entity support files without assuming one OS layout."""
    candidates = artifact_title_candidates(book_name)
    paths = []
    if kind == 'aliases':
        for title in candidates:
            paths.extend([
                os.path.join(books_root, 'contexts', title, 'alias_map.json'),
                os.path.join(books_root, title, 'contexts', 'alias_map.json'),
            ])
    elif kind == 'entities':
        paths = [os.path.join(books_root, 'entities',
                              f'{title}.entities.json')
                 for title in candidates]
    else:
        raise ValueError(f'unknown support kind: {kind}')
    return next((path for path in paths if os.path.isfile(path)), None), paths


def load_book_aliases(books_root, book_name):
    """Load alias_map for a book. Returns (alias_map, canonical_to_aliases).

    canonical_to_aliases maps each canonical name to a list of all its aliases
    (including the canonical name itself).
    """
    if book_name in _book_cache:
        return _book_cache[book_name]

    # Layout B: {books_root}/contexts/{book}/alias_map.json
    # Layout A fallback: {books_root}/{book}/contexts/alias_map.json
    alias_path, tried = find_support_path(books_root, book_name, 'aliases')
    if alias_path is None:
        raise FileNotFoundError(
            f"no alias_map.json for book {book_name!r}; tried {tried}")

    with open(alias_path, encoding="utf-8") as f:
        alias_map = json.load(f).get("alias_map", {})

    # canonical -> list of aliases
    canonical_to_aliases = defaultdict(list)
    for alias, canon in alias_map.items():
        canonical_to_aliases[canon].append(alias)
    # Add canonical name itself as an alias
    for canon in list(canonical_to_aliases.keys()):
        if canon not in canonical_to_aliases[canon]:
            canonical_to_aliases[canon].append(canon)

    result = (alias_map, dict(canonical_to_aliases))
    _book_cache[book_name] = result
    return result


def extract_surname_fragments(canonical_to_aliases):
    """Extract all non-first-word tokens from canonical names.

    These are patronymics, surnames, middle names etc. that may dangle after
    a half-replacement (e.g. "Character N Pavlovitch"). Used to build a
    language-agnostic cleanup regex.
    """
    fragments = set()
    for canon in canonical_to_aliases:
        parts = canon.split()
        if len(parts) >= 2:
            for p in parts[1:]:
                if len(p) >= 3:  # skip short tokens like "de", "von"
                    fragments.add(p)
    return fragments


# ---------------------------------------------------------------------------
# Reverse mapping: LLM name_map -> canonical -> Character N
# ---------------------------------------------------------------------------

def build_char_map_from_audit(name_replacements, alias_map, canonical_to_aliases):
    """Given LLM's {name_form -> Character N}, return {canonical -> Character N}."""
    canon_to_char = {}
    for name_form, char_n in name_replacements.items():
        # Audit replay already validates this, but keep this stage robust when
        # it is run directly on older artifacts.
        if not isinstance(char_n, str) or not re.fullmatch(
                r"Character [1-9]\d*", char_n):
            continue
        canon = alias_map.get(name_form)
        if canon is None:
            # Try quote-normalized match against alias_map
            for a, c in alias_map.items():
                if norm_quotes(a) == norm_quotes(name_form):
                    canon = c
                    break
        if canon is None:
            # Maybe the name_form IS a canonical name
            if name_form in canonical_to_aliases:
                canon = name_form
        if canon is None:
            # Maybe name_form is a single first name of some canonical.
            # Match by first token, only if unambiguous.
            tokens = name_form.split()
            first_token = tokens[0] if tokens else ""
            # Strip possessive: "Fyodor's" -> "Fyodor"
            if first_token.endswith("'s"):
                first_token = first_token[:-2]
            if first_token:
                candidates = [c for c in canonical_to_aliases
                             if c.split() and c.split()[0] == first_token]
                if len(candidates) == 1:
                    canon = candidates[0]
        if canon is not None:
            canon_to_char.setdefault(canon, char_n)
    return canon_to_char


# ---------------------------------------------------------------------------
# Fill pair construction
# ---------------------------------------------------------------------------

def build_fill_pairs(canon_to_char, canonical_to_aliases):
    """Build (alias, Character N) pairs for second-pass replacement, longest first.

    For each canonical name, derive ALL possible name forms:
    - All aliases from alias_map
    - 2-part (first + second token) from 3+ part canonicals
    - Single first name + its possessive ('s) and truncated (')
    """
    pairs = []
    for canon, char_n in canon_to_char.items():
        aliases = list(canonical_to_aliases.get(canon, []))
        parts = canon.split()
        # Derive 2-part (first + second) from 3+ part canonicals
        if len(parts) >= 3:
            two_part = parts[0] + " " + parts[1]
            if two_part not in aliases:
                aliases.append(two_part)
        # Derive single first name + possessive
        if parts:
            first = parts[0]
            if first not in aliases:
                aliases.append(first)
            aliases.append(first + "'s")   # possessive
            aliases.append(first + "'")    # truncated possessive
        for alias in aliases:
            pairs.append((alias, char_n))
    # Dedupe and sort longest first
    seen = {}
    for alias, char_n in pairs:
        seen.setdefault(alias, char_n)
    return sorted(seen.items(), key=lambda kv: -len(kv[0]))


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------

def make_fill_replacer(pairs, surname_fragments):
    """Build a safe alias replacer plus half-replaced surname cleanup."""
    replace_alias = compile_replacer(pairs)
    cleanup_possessive = cleanup_plain = None
    if surname_fragments:
        fragments = "|".join(
            re.escape(fragment) for fragment in sorted(
                surname_fragments, key=lambda value: -len(value)))
        cleanup_possessive = re.compile(
            r"(Character [1-9]\d*)\s+(?:" + fragments + r")['’]s(?!\w)")
        cleanup_plain = re.compile(
            r"(Character [1-9]\d*)\s+(?:" + fragments + r")(?!\w)")

    def replace(text):
        text = replace_alias(text)
        if cleanup_possessive:
            text = cleanup_possessive.sub(r"\1's", text)
            text = cleanup_plain.sub(r"\1", text)
        return text

    return replace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Input JSONL (LLM-anonymized records)")
    ap.add_argument("--audit", required=True,
                    help="Audit JSONL (name_replacements per record)")
    ap.add_argument("--output", required=True,
                    help="Output JSONL (filled records)")
    ap.add_argument("--books-root", required=True,
                    help="Root dir with contexts/{book}/alias_map.json")
    args = ap.parse_args()

    args.books_root = os.path.abspath(args.books_root)

    # Load audit (record_id -> name_replacements)
    audit = {}
    with open(args.audit, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
                rid = a.get("record_id") or a.get("question_id") or a.get("sample_id")
                if not rid:
                    raise ValueError('audit record missing ID')
                if rid in audit:
                    raise ValueError(f'duplicate audit record ID: {rid}')
                audit[rid] = a.get("name_replacements", {})
            except Exception as error:
                raise SystemExit(f'invalid audit line: {error}') from error

    input_records = []
    input_ids = set()
    with open(args.input, encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = get_record_id(rec)
            if not rid:
                raise SystemExit('input record missing question_id/sample_id')
            if rid in input_ids:
                raise SystemExit(f'duplicate input record ID: {rid}')
            input_ids.add(rid)
            input_records.append(rec)
    if input_ids != set(audit):
        missing = sorted(input_ids - set(audit))[:5]
        extra = sorted(set(audit) - input_ids)[:5]
        raise SystemExit(
            f'input/audit ID mismatch: missing_audit={missing}, extra_audit={extra}')

    # Process: group by book to load alias_map once per book
    filled = 0
    total = 0
    books_seen = set()

    output_records = []
    for rec in input_records:
            total += 1
            rid = get_record_id(rec)
            book = rec.get("book", "")
            books_seen.add(book)

            alias_map, canonical_to_aliases = load_book_aliases(
                args.books_root, book)
            surname_fragments = extract_surname_fragments(canonical_to_aliases)

            name_repl = audit.get(rid, {})
            canon_to_char = build_char_map_from_audit(
                name_repl, alias_map, canonical_to_aliases)

            # Supplement: if the record's "character" field was already
            # anonymized to "Character N", but the original canonical name
            # is not in canon_to_char (LLM missed it), derive it from the
            # character field's mapping.  The character field maps to
            # Character 1 (or whatever N) in the anonymized record.
            anon_char = rec.get("character", "")
            # A value such as "Character 1's mother" only contains an
            # anonymized mention; it is not itself the character placeholder.
            # Treating the whole phrase as a mapping target later breaks the
            # numeric Character N allocation.
            char_match = re.fullmatch(r"Character ([1-9]\d*)", anon_char)
            if char_match:
                char_n = anon_char
                # Find which canonical this entity maps to: check if any
                # canonical's aliases appear in the original name_repl
                # values (they should, since character=Character 1 means
                # the LLM replaced this character's name).
                # If not found, try the record's entity_id -> canonical.
                entity_id = rec.get("entity_id", "")
                if entity_id:
                    # Look up canonical by entity_id in the entities file
                    entities_path, _ = find_support_path(
                        args.books_root, book, 'entities')
                    if entities_path:
                        try:
                            ents = json.load(open(entities_path, encoding="utf-8"))
                            for e in ents.get("entities", []):
                                if e.get("entity_id") == entity_id:
                                    canon = e.get("canonical_name", "")
                                    if canon and canon not in canon_to_char:
                                        canon_to_char[canon] = char_n
                                    break
                        except Exception:
                            pass

            # Check for residual names in participant_descriptions keys that
            # the LLM missed entirely.  This must also run when none of the
            # audit names matched alias_map; otherwise records containing only
            # role-like entities ("Priest", "the master", etc.) are skipped.
            pds = rec.get("scene", {}).get("participant_descriptions", {})
            if isinstance(pds, dict):
                existing_targets = list(canon_to_char.values()) + list(
                    name_repl.values())
                max_n = max((int(match.group(1)) for value in existing_targets
                             if isinstance(value, str)
                             for match in [re.fullmatch(
                                 r"Character ([1-9]\d*)", value)]
                             if match), default=0)
                for pd_key in list(pds.keys()):
                    if pd_key.startswith("Character "):
                        continue
                    # Check if this key maps to a known canonical.
                    canon = alias_map.get(pd_key)
                    if canon is None:
                        for alias, candidate in alias_map.items():
                            if norm_quotes(alias) == norm_quotes(pd_key):
                                canon = candidate
                                break
                    if canon and canon not in canon_to_char:
                        max_n += 1
                        canon_to_char[canon] = f"Character {max_n}"

            if canon_to_char:
                pairs = build_fill_pairs(canon_to_char, canonical_to_aliases)
                rec = replace_record_strings(
                    rec, make_fill_replacer(pairs, surname_fragments))
                filled += 1
            output_records.append(rec)

    atomic_write_jsonl(args.output, output_records)
    meta = {
        'input_sha256': file_sha256(args.input),
        'audit_sha256': file_sha256(args.audit),
        'record_count': total,
    }
    meta_path = args.output + '.meta.json'
    tmp_meta = f'{meta_path}.{os.getpid()}.tmp'
    with open(tmp_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_meta, meta_path)

    print(f"Processed {total} records, filled variants for {filled}")
    print(f"Books seen: {sorted(books_seen)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
