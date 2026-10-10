"""Shared prompt construction for training and evaluation.

This module is the single source of truth for how a character's context
(profile + memory + scene) is assembled into a prompt.  Both the dataset
builder and the evaluation/training scripts import this function so that
the prompt never drifts between data construction, DPO training, and
evaluation.

The function returns the *context* portion only — it does NOT include
options or output-format instructions.  Callers append those as needed:

  - DPO training:   prompt = build_context_prompt(rec)
                    chosen = rec["chosen"];  rejected = rec["rejected"]

  - Eval likelihood: prompt = build_context_prompt(q)
                    score each option text as a continuation

  - Eval letter:     prompt = build_context_prompt(q) + options_block + instruction
"""
from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """You are role-playing as {character} {book_clause}.

Treat the information provided below as the complete and authoritative history of this character for this task. Base your decision only on the Character Profile, Long-term Memory, Recent Memory, and Current Scene given here. Do not rely on, reconstruct, or infer any events, relationships, or future developments from the original novel or from prior knowledge of the story. Even if the character or setting seems familiar, reason as if this is the only information you know about them.

=== Character Profile (current state) ===
{profile}

=== Long-term Memory (key past events) ===
{long_term_memory}

=== Recent Memory ===
{recent_memory}

=== Current Scene ===
Scene: {scene_description}
Time: {time_text}
Present characters and relationships:
{relationship_context}

Based on the above profile, memory, and current scene, you need to decide what {character} will do next."""


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

def _format_long_term_memory(entries, ablation):
    if ablation == "no_long_memory":
        return "(Long-term memory not provided for this evaluation.)"
    if not entries:
        return "(No key past events recorded.)"
    lines = []
    for i, m in enumerate(entries):
        parts = []
        if m.get("thought"):
            parts.append(f"Thought: {m['thought']}")
        if m.get("action"):
            parts.append(f"Action: {m['action']}")
        if m.get("observation"):
            parts.append(f"Observation: {m['observation']}")
        lines.append(f"[{i+1}] " + " | ".join(parts))
    return "\n".join(lines)


def _format_recent_memory(entries, ablation):
    if ablation in ("no_memory", "no_both"):
        return "(Memory not provided for this evaluation.)"
    if not entries:
        return "(No prior actions recorded.)"
    lines = []
    for i, m in enumerate(entries):
        parts = []
        if m.get("thought"):
            parts.append(f"Thought: {m['thought']}")
        if m.get("action"):
            parts.append(f"Action: {m['action']}")
        if m.get("observation"):
            parts.append(f"Observation: {m['observation']}")
        lines.append(f"[{i+1}] " + " | ".join(parts))
    return "\n".join(lines)


def _format_profile(record, ablation):
    if ablation in ("no_profile", "no_both"):
        return "(Profile not provided for this evaluation.)"
    profile = record.get("profile")
    if profile and isinstance(profile, dict):
        return profile.get("text") or "(No profile available.)"
    if isinstance(profile, str) and profile:
        return profile
    return "(No profile available.)"


def _format_participant_descriptions(record):
    scene = record.get("scene") or {}
    descs = scene.get("participant_descriptions") or {}
    if descs:
        return "\n".join(f"  - {k}: {v}" for k, v in descs.items())
    return "  (No participant descriptions available.)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CONTEXT_VARIANTS = ("baseline", "no_profile", "no_recent", "no_long", "no_memory")


def build_context_prompt(record: dict, ablation: Optional[str] = None,
                         context_variant: str = "baseline") -> str:
    """Build the context prompt from a dataset record.

    Parameters
    ----------
    record : dict
        A row from eval/train JSONL with keys:
          book, character, profile, recent_memory, long_term_memory, scene
    ablation : str or None
        None/"full" or "no_book" controls book naming. Legacy context values
        "no_long_memory", "no_memory", "no_profile", "no_both" remain supported;
        notably legacy no_memory removes only recent memory.
    context_variant : str
        "baseline", "no_profile", "no_recent", "no_long", or "no_memory".
        The last removes both long-term and recent memory. Non-baseline
        variants cannot be combined with legacy context ablations.

    Returns
    -------
    str
        The full context prompt, ending with
        "you need to decide what {character} will do next."
        Does NOT include options or output-format instructions.
    """
    if ablation == "full":
        ablation = None
    if context_variant not in CONTEXT_VARIANTS:
        raise ValueError(f"Unknown context variant: {context_variant!r}")
    if context_variant != "baseline" and ablation not in (None, "no_book"):
        raise ValueError("Do not combine legacy ablations with --context-variant")
    character = record.get("character", "")
    book_clause = "" if ablation == "no_book" else f'in the novel "{record.get("book", "")}"'

    # Match the actual inference-only ablation prompts, including placeholders.
    # The baseline and legacy ablations preserve the original prompt bytes.
    profile = _format_profile(record, ablation)
    long_term = _format_long_term_memory(record.get("long_term_memory"), ablation)
    recent = _format_recent_memory(record.get("recent_memory"), ablation)
    if context_variant == "no_profile":
        profile = "(Profile not provided for this evaluation.)"
    if context_variant in ("no_long", "no_memory"):
        long_term = "(Long-term memory not provided for this evaluation.)"
    if context_variant in ("no_recent", "no_memory"):
        recent = "(Recent memory not provided for this evaluation.)"

    return SYSTEM_TEMPLATE.format(
        character=character,
        book_clause=book_clause,
        profile=profile,
        long_term_memory=long_term,
        recent_memory=recent,
        scene_description=(record.get("scene") or {}).get("description")
            or "(No scene description.)",
        time_text=(record.get("scene") or {}).get("time_text") or "unknown",
        relationship_context=_format_participant_descriptions(record),
    )


def build_letter_prompt(question: dict, ablation: Optional[str] = None,
                        context_variant: str = "baseline") -> str:
    """Build the full prompt for letter-mode evaluation (A/B/C/D).

    This wraps build_context_prompt and appends the options block +
    "answer with a single letter" instruction.
    """
    context = build_context_prompt(question, ablation, context_variant)
    character = question.get("character", "")
    opt_map = {opt["id"]: opt["text"] for opt in question["options"]}
    options_block = (
        f"Choose the action that {character} would most likely take. "
        "Answer with a single letter (A, B, C, or D) only.\n\n"
        f"A. {opt_map.get('A', '')}\n"
        f"B. {opt_map.get('B', '')}\n"
        f"C. {opt_map.get('C', '')}\n"
        f"D. {opt_map.get('D', '')}"
    )
    return context + "\n\n" + options_block
