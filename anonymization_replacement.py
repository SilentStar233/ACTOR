#!/usr/bin/env python3
"""Safe, deterministic text replacement for anonymized dataset records.

The anonymization pipeline receives name mappings from an LLM and later adds
aliases from entity metadata.  Both stages must obey the same safety rules:

* replacements match complete tokens, never substrings inside other words;
* JSON schema keys are immutable, except the dynamic person-name keys under
  ``scene.participant_descriptions``;
* identifiers, labels, split metadata, and option IDs are immutable; and
* replacement output is not scanned again by later replacement pairs.

This module contains no API or third-party dependencies, so stored audit maps
can be replayed locally without importing the LLM client.
"""

from __future__ import annotations

import hashlib
import json
import re


_PLACEHOLDER_RE = re.compile(r"Character [1-9]\d*")

# Metadata values that must remain byte-for-byte equivalent to the source.
# Matching by field name protects nested memory action/scene identifiers too.
_PROTECTED_FIELD_NAMES = {
    "question_id",
    "sample_id",
    "book",
    "entity_id",
    "action_uid",
    "scene_uid",
    "is_critical",
    "character_action_order",
    "correct_answer",
    "split",
    "id",  # eval option labels A/B/C/D
}

_PROTECTED_PATHS = {
    ("profile", "version"),
    ("profile", "freshness"),
    ("profile", "distance"),
}

_DYNAMIC_KEY_CONTAINER = ("scene", "participant_descriptions")
_DESCRIPTION_SEPARATOR = "\n\n"


class ReplacementError(ValueError):
    """Raised when a replacement map would damage record structure."""


def source_hash(record):
    """Return the stable fingerprint used by anonymization audit records."""
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_record_id(record):
    return record.get("question_id") or record.get("sample_id") or ""


def _quote_flexible_pattern(text):
    """Escape a literal while accepting straight/curly quote variants."""
    pieces = []
    for char in text:
        if char in "'\u2018\u2019":
            pieces.append("['\u2018\u2019]")
        elif char in '\"\u201c\u201d':
            pieces.append('[\"\u201c\u201d]')
        else:
            pieces.append(re.escape(char))
    return "".join(pieces)


def compile_replacer(pairs):
    """Compile ``(literal name, Character N)`` pairs into one safe pass.

    A single regex pass is important: placeholders inserted by one mapping
    can never become input to a later mapping.  Exact duplicate aliases keep
    the first mapping, matching the pipeline's historical longest-first rule.
    """
    normalized = []
    seen = set()
    for literal, placeholder in pairs:
        if not isinstance(literal, str) or not literal:
            continue
        if not isinstance(placeholder, str) or not _PLACEHOLDER_RE.fullmatch(
                placeholder):
            raise ReplacementError(
                f"invalid replacement target for {literal!r}: {placeholder!r}")
        quote_key = (literal.replace("\u2018", "'").replace("\u2019", "'")
                     .replace("\u201c", '\"').replace("\u201d", '\"'))
        if quote_key in seen:
            continue
        seen.add(quote_key)
        normalized.append((literal, quote_key, placeholder))

    # Longest first resolves overlapping forms such as full name vs first name.
    normalized.sort(key=lambda item: -len(item[0]))
    if not normalized:
        return lambda text: text

    by_key = {item[1]: item[2] for item in normalized}
    alternatives = "|".join(_quote_flexible_pattern(item[0])
                            for item in normalized)
    # Unicode-aware \w boundaries prevent Tim->Time, H->The, etc.
    matcher = re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)")

    def replace(text):
        if not isinstance(text, str) or not text:
            return text

        def substitution(match):
            key = (match.group(0).replace("\u2018", "'")
                   .replace("\u2019", "'").replace("\u201c", '\"')
                   .replace("\u201d", '\"'))
            return by_key[key]

        return matcher.sub(substitution, text)

    return replace


def _is_protected_value(path):
    return (bool(path) and path[-1] in _PROTECTED_FIELD_NAMES
            or tuple(path) in _PROTECTED_PATHS)


def replace_record_strings(record, replace_text):
    """Apply ``replace_text`` without allowing JSON schema corruption."""

    def merge_participant_descriptions(existing, incoming, key, path):
        """Preserve descriptions when multiple aliases collapse to one person.

        ``participant_descriptions`` is the sole dictionary whose keys are
        data rather than schema.  A scene can therefore contain, for example,
        both a full name and a surname for the same person.  Once both aliases
        become ``Character N`` their descriptions must be combined instead of
        treating the expected key collision as schema corruption.
        """
        if existing == incoming:
            return existing
        if not isinstance(existing, str) or not isinstance(incoming, str):
            raise ReplacementError(
                f"cannot merge non-string participant descriptions at "
                f"{path}: {key!r}")

        # Avoid duplicating a shorter description already contained verbatim
        # in the longer one, while retaining both genuinely different texts.
        if existing and existing in incoming:
            return incoming
        if incoming and incoming in existing:
            return existing
        if not existing:
            return incoming
        if not incoming:
            return existing
        return existing + _DESCRIPTION_SEPARATOR + incoming

    def walk(obj, path=()):
        if isinstance(obj, str):
            return obj if _is_protected_value(path) else replace_text(obj)
        if isinstance(obj, list):
            # List indices are intentionally omitted so paths such as
            # ("options", "id") are stable for every item.
            return [walk(value, path) for value in obj]
        if isinstance(obj, dict):
            output = {}
            dynamic_keys = tuple(path) == _DYNAMIC_KEY_CONTAINER
            for key, value in obj.items():
                new_key = replace_text(key) if dynamic_keys else key
                new_value = walk(value, path + (key,))
                if new_key in output:
                    if not dynamic_keys:
                        raise ReplacementError(
                            f"replacement key collision at {path}: "
                            f"{new_key!r}")
                    output[new_key] = merge_participant_descriptions(
                        output[new_key], new_value, new_key, path)
                else:
                    output[new_key] = new_value
            return output
        return obj

    before = schema_signature(record)
    output = walk(record)
    after = schema_signature(output)
    if before != after:
        raise ReplacementError("record schema changed during replacement")
    return output


def schema_signature(obj, path=()):
    """Describe all fixed dictionary keys, excluding dynamic participant keys."""
    signature = []
    if isinstance(obj, dict):
        if tuple(path) != _DYNAMIC_KEY_CONTAINER:
            signature.append((tuple(path), tuple(obj.keys())))
        for key, value in obj.items():
            child = path + (("<participant>") if tuple(path)
                            == _DYNAMIC_KEY_CONTAINER else key,)
            signature.extend(schema_signature(value, child))
    elif isinstance(obj, list):
        for value in obj:
            signature.extend(schema_signature(value, path + ("[]",)))
    return signature


def apply_name_map(record, name_map):
    """Apply an audit/LLM name map to a clean source record."""
    if not isinstance(name_map, dict) or not name_map:
        raise ReplacementError("name map is empty or invalid")
    replacer = compile_replacer(name_map.items())
    return replace_record_strings(record, replacer)
