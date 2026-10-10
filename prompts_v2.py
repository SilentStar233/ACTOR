# v2 extraction prompt (experiment): full-coverage extraction with individuality
# threshold, verbatim dialogue recording, counterfactual distractors, criticality
# labels, and scene-level time anchors. Does NOT modify prompts.py.
#
# Aligned decisions (2026-07-29):
# - Record ALL speech; record physical actions only if they express personality.
# - Dialogue action format: "<manner> <speech verb>: \"<verbatim quote>\""
# - One utterance = one step; long speeches are NOT split.
# - thought: brief (evidence optional) for mundane steps; full + mandatory
#   evidence for complex reasoning; critical actions MUST have evidence.
# - counterfactual_action: 3 alternatives, situationally plausible but
#   OUT OF CHARACTER; no mere rewording.
# - is_critical judged at extraction time as "real choice that changes a
#   persistent state or future options" (NOT "name a specific later event";
#   downstream-consequence verification belongs to the audit stage), with
#   default-false list and decision-chain dedup, plus criticality_reason.
# - Scene time_text/time_evidence only when the text mentions time, else null.

EXTRACTION_PROMPT_V2 = """Based on the provided plot text from a novel, extract the behavioral trajectories of the characters as Thought-Action-Observation (T-A-O) loops.

===Background===

We study how characters in novels behave and make decisions. Record each character's behavior as a sequence of loops:
- Thought: the character's first-person inner state or reasoning behind the behavior
- Action: what the character does or says
- Observation: the feedback the character perceives after acting

===What to record (CRITICAL)===

1. Record ALL speech acts — every line of dialogue a character speaks, including small talk and pleasantries. Speech always carries personal style.
2. Record physical (non-speech) actions ONLY when they express THIS character's personality, choice, or style. Apply the individuality test: if any ordinary person in this situation would do exactly the same thing, SKIP it.
   - RECORD examples: refusing a proposal; eavesdropping on purpose; taking a window seat to observe someone; a deliberate cold smile while listening; pointedly changing the subject.
   - SKIP examples: waking up, stretching, yawning, drinking water; standing up or walking across a room with no particular purpose; mechanically opening a door; eating or dressing with no characterization.
3. A meaningful silence or avoidance (when it clearly carries intent) counts as behavior — record it.

===Action format (CRITICAL)===

- Dialogue steps: "<manner/tone> <speech verb>: \\"<verbatim quote>\\""
  * The speech verb carries the function (refuses, teases, deflects, presses).
  * The manner captures HOW it is said (coldly, with a sigh, teasingly) when the text shows it.
  * The quote must be EXACTLY the character's words from the text, verbatim. Only DIRECT speech (inside the text's own quotation marks) may be quoted. For REPORTED/indirect speech (e.g., "He replied that he had not"), do NOT fabricate a quote — use the functional form instead (e.g., "Replies that he had not").
  * ONE utterance = ONE step. Do NOT split a long speech into multiple steps even if it performs several functions; keep the quote whole.
- Non-dialogue steps: a functional description of what the behavior accomplishes (e.g., "Silently signals a companion to follow upstairs").
- Atomicity for non-dialogue: if the text describes several distinct physical decisions in a row, split them into consecutive steps.

===Thought rules (CRITICAL)===

- For mundane, everyday steps: write ONE brief first-person sentence of inner state (e.g., "I am bored."). "thought_evidence" may be null for these.
- For steps involving real deliberation or motivation: write the full first-person reasoning, and "thought_evidence" is REQUIRED — a short VERBATIM quote from the text it is grounded in. NEVER invent inner states with no basis in the text.
- For every step where "is_critical" is true: full thought + non-null "thought_evidence" are MANDATORY.

===Counterfactual actions (CRITICAL)===

For EVERY step, generate exactly 3 "counterfactual_action" alternatives. Each must be:
1. Situationally plausible — something a person COULD reasonably do or say in this exact situation;
2. OUT OF CHARACTER for THIS character — it violates their personality, values, or style (an action they would not take, or a tone they would not use);
3. NOT a mere rewording — do not keep the same meaning, tone, and manner while only changing word order or phrasing.
For dialogue steps, write the alternatives as utterances in the same format: "<manner> <verb>: \\"<quote>\\"".

===Criticality (CRITICAL)===

For every step, judge "is_critical". A step is critical ONLY when BOTH hold:
1. The character makes a real choice — they could plausibly have acted otherwise in this exact situation;
2. The choice changes some PERSISTENT state or the space of future options:
   - goals, plans, or intentions;
   - promises, duties, or obligations;
   - relationships, alliances, trust, or conflicts;
   - possession or concealment of important information;
   - location, route, resources, or access — ONLY when the movement itself creates or removes an important future opportunity (e.g., departing for another city, gaining entry somewhere decisive); ordinary movement within or between nearby scenes does NOT count;
   - identity, status, or role;
   - survival, or future opportunities to act.
The consequence does NOT need to appear in the current plot — effects often surface only in later plots. Judge the state change, not whether you can see the payoff.
Critical examples: refusing a marriage proposal (relationship and status change); deciding to pay or avoid a visit (future meetings become possible or impossible); revealing or deliberately withholding a secret (information possession changes).

Default to "is_critical": false. These are NOT critical:
- routine information exchange and passive answers;
- mere expression of emotion, wishes, or opinions (but an utterance that itself CREATES a commitment, threat, or disclosure is a choice — judge the act, not the feeling);
- openings, continuations, and closings of ordinary conversation;
- politeness, jokes, teasing, irony, or shifts of tone;
- explanations, repetitions, or defenses of a decision already made;
- routine social execution, even when done deliberately: introductions, greetings, following someone, sitting down, walking over, sending a routine note, keeping a routine appointment;
- behavior whose only effect is an immediate reaction within this scene, or only the scene's mood, wit, or color;
- anything where you are unsure a persistent state changed. When in doubt, set false.

Decision chains: when consecutive steps form ONE decision, mark ONLY the core choice as critical; the rest are false. This applies to BOTH speech chains and action chains:
- Speech chain: making the decision -> explaining it -> restating it -> answering questions about it.
- Action chain: declaring intent -> setting off -> traveling/arriving -> completing the declared action. Example: "declares she will walk to the next town" -> "sets off alone" -> "walks three miles" -> "arrives" is ONE decision; mark only the declaration.
If two steps would receive the same criticality_reason, they belong to one chain — keep only the step where the choice is actually made.

Provide "criticality_reason" for critical steps ONLY: (a) what the character chose; (b) what persistent state or future option it changed, specifically (whose state, which one). Do NOT predict or invent specific later events, and do NOT write generic claims like "this affects the story". If you cannot name the state change, set "is_critical" to false.

===Observation rules===

The feedback THIS character perceives after acting: others' visible reactions, spoken replies, or changes in the environment; in solo scenes, what they see, hear, or discover as a direct result. Write observations in the THIRD person. If none exists, set "observation" to null — do NOT fabricate. NEVER use the narrator's omniscient commentary or later events.

===Scene time anchor===

For each scene, record when the actions and dialogue IN THIS SCENE actually occur:
- "time_text": the scene's actual time or full time span, as stated or directly grounded in the text;
- "time_evidence": the verbatim quote(s) that support that time. If a span is supported by separate passages, join the excerpts with " ... ".

CRITICAL boundary: a time expression merely mentioned inside this scene is NOT necessarily the scene's time. Do NOT use dates/times that belong only to a future plan, a past memory, a different event, or a letter/document being read. If no text establishes when the current scene itself occurs, set both fields to null. Do NOT guess from outside knowledge.

Examples:
- "The next morning she was sitting by herself." -> time_text: "the next morning".
- Actions begin Saturday morning and continue through "On Sunday, after morning service" -> time_text: "from Saturday morning to Sunday after morning service".
- "The ball was fixed for the following Tuesday." in an earlier invitation scene -> null (Tuesday is the future ball's time, not the invitation scene's time).
- A character says, "They will join me early to-morrow." -> null unless other narration establishes the current scene's time (tomorrow is a future arrival).
- "The letter was dated Monday, August 2." -> null unless the text separately establishes that the current reading scene occurs on that date.

===Character Knowledge Rules===

Character contexts are provided below. When writing a character's thought:
1. The character only knows events they personally experienced or common knowledge — NOT events in their "did NOT witness" list, nor later events.
2. Do NOT use knowledge of this novel from outside the provided text.
3. Keep thoughts consistent with the personality shown in that character's prior experiences.

===Steps===

1. Identify scene(s) in the plot: continuous interactions in one setting. For each scene, describe the setting, list the participants, note their relationships, and mark the time anchor if present.
2. For each key character in the scene, extract their trajectory: an ordered sequence of T-A-O steps following the narrative order. Apply the individuality threshold above.

===Output Format===

Strict JSON (no markdown fences, no comments). The "plots" array must contain EXACTLY ONE entry — the single plot supplied in the input. Put EVERY scene you identify into that one entry's "scenes" array ("scene_id" 0, 1, 2, ... in narrative order). Never split the input into multiple plot entries, even when it interleaves parallel storylines.
{{
  "plots": [
    {{
      "plot_id": 0,
      "chapter_title": "{chapter_title}",
      "scenes": [
        {{
          "scene_id": 0,
          "description": "The setting and situation at the start of the scene.",
          "participants": ["CharacterA", "CharacterB"],
          "relationship_context": {{"CharacterA": "A's relationship to the others"}},
          "time_text": "..." or null,
          "time_evidence": "..." or null,
          "trajectories": [
            {{
              "character": "CharacterA",
              "steps": [
                {{
                  "step_id": 0,
                  "thought": "...",
                  "thought_evidence": "..." or null,
                  "action": "...",
                  "observation": "..." or null,
                  "counterfactual_action": ["...", "...", "..."],
                  "is_critical": true or false,
                  "criticality_reason": "..."
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}

===Requirements===

1. Output valid JSON only. Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED.
2. Use the canonical character names provided in the character contexts / key character list.
3. Stay faithful to the original text. Do not invent events, thoughts, quotes, or reactions with no basis in the text.
4. Keep all output in the same language as the source text.

===Input===

==Book title==
{title}

==Author==
{author}
{plot_context}{character_context}
==Plot text==
{text}
"""


def build_extraction_prompt_v2(title, author, text, chapter_title=None,
                               plot_summary=None, key_characters=None,
                               character_context=None):
    """Render EXTRACTION_PROMPT_V2 with optional plot metadata and character context."""
    plot_context = ''
    if plot_summary or key_characters:
        parts = []
        if plot_summary:
            parts.append(f'==Plot summary==\n{plot_summary}\n\n')
        if key_characters:
            lines = [f"- {c['name']}: {c.get('experience', c.get('description', ''))}"
                     for c in key_characters]
            parts.append('==Key characters in this plot==\n' + '\n'.join(lines) + '\n\n')
        plot_context = '\n' + ''.join(parts)

    ctx = ''
    if character_context:
        blocks = []
        for char, info in character_context.items():
            exp = '\n'.join(f'  - [{e["chapter"]}] {e["experience"]}'
                            for e in info.get('prior_experiences', []))
            nw = '\n'.join(f'  - [{e["chapter"]}] {e["summary"]}'
                           for e in info.get('not_witnessed', []))
            block = f'[{char}]\nPersonally experienced so far:\n{exp or "  (none)"}'
            if nw:
                block += f'\nDid NOT witness (does NOT know these):\n{nw}'
            blocks.append(block)
        ctx = ('\n==Character contexts (when writing a character\'s thought, '
               'use ONLY that character\'s knowledge)==\n' + '\n\n'.join(blocks) + '\n\n')

    return EXTRACTION_PROMPT_V2.format(
        title=title,
        author=author,
        chapter_title=chapter_title or 'Unknown',
        plot_context=plot_context,
        character_context=ctx,
        text=text,
    )
