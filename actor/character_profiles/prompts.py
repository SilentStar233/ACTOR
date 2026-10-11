"""Prompts for dynamic character-profile generation."""

# Dynamic character profiles

# Prompts for dynamic character-profile generation.
# Profile is a single compact text: current character state + relevant lived
# experience. The action window never includes future information.

PROFILE_INITIALIZATION_PROMPT = """Create the initial dynamic profile for one character in the novel "{title}".

This profile exists IMMEDIATELY BEFORE the character's first recorded action. Write only what is supported by the supplied PRE-ACTION character description. Do not use outside knowledge of the novel, later events, future names/titles, or the first action itself.

The profile is one compact text of AT MOST 120 words (same language as the source) covering:
1. current identity/social position when evidenced;
2. stable or emerging temperament, values, and interaction style when evidenced;
3. current relationship orientation, goals, or relevant lived experience only when evidenced.

Be explicitly cautious when evidence is limited. Do not invent biographical facts, future relationships, or later character development.

Your entire response must fit within 800 tokens. Output exactly the JSON object and no commentary.

Return strict JSON only:
{{
  "profile": "...",
  "evidence_assessment": "brief statement of what the pre-action evidence supports and does not support"
}}

==Persistent entity==
entity_id: {entity_id}
current source designation: {source_name}
first recorded timeline position: {timeline_order}
first recorded time: {time_text}

==Pre-action character description==
{pre_action_description}
"""


PROFILE_UPDATE_PROMPT = """Update one character's dynamic profile in the novel "{title}".

You are positioned immediately AFTER the current trigger action. The action window contains every action that happened since the last saved profile version, in chronological order. All listed actions are past or current; you must not infer any later event or use outside knowledge of the novel.

The profile is one compact text of AT MOST 140 words (same language as the source) that jointly captures:
1. current identity/social position when evidenced;
2. stable temperament, values, interaction style, and important attitudes;
3. relevant lived experience that still shapes current judgment or behavior.

Decide whether the accumulated action window MATERIALLY changes this profile.

Set "changed" to true only when the interval produces a durable change in identity/social position, stable attitude, important relationship stance, goal orientation, self-understanding, or lived experience worth retaining in the compact profile.

Set "changed" to false when the actions merely repeat, confirm, or temporarily express the existing profile. A plot-important action does not automatically require a profile update.

If changed=false, return profile=null. If changed=true, rewrite the COMPLETE current profile: preserve still-relevant prior information, integrate the newly material experience, and compress details that no longer matter. Never make the profile a chronological action log. Do NOT quote, enumerate, or explain individual input actions in the profile.

Your entire response must fit within 800 tokens. Output exactly the JSON object and no commentary.

Do not use or mention counterfactual alternatives, criticality/audit explanations, future aliases/titles, future actions, or future plot outcomes.

Return strict JSON only:
{{
  "changed": true or false,
  "profile": "..." or null,
  "change_summary": "one concise evidence-grounded explanation"
}}

==Persistent entity==
entity_id: {entity_id}
current source designation: {source_name}
current timeline position: {timeline_order}
current time: {time_text}

==Previous saved profile (before this action window)==
profile_version: {profile_version}
{previous_profile}

==Action window since that profile==
{action_window}
"""
