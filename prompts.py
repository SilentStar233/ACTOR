# All prompts for the action-trajectory extraction pipeline.
#
# - PLOT_DETECTION_PROMPT: Stage A, plot boundary detection (with rolling character registry)
# - ALIAS_GROUP_PROMPT:     Stage A.6a, group character aliases
# - EXTRACTION_PROMPT:      Stage B, T-A-O trajectory extraction (v2, with grounding rules)
#   Use build_extraction_prompt() to render it with optional plot/character context.

# ---------------------------------------------------------------------------
# Stage A: plot boundary detection
# ---------------------------------------------------------------------------

PLOT_DETECTION_PROMPT = """Based on the provided book chunk, complete the following tasks:

1. Recognize chapter beginnings if they exist in the chunk. Identify the starting sentence of each chapter.

2. Identify the important plots in this chunk. Identify the beginning and ending of each plot by its first and last sentence. Determine the chapter title that the plot belongs to. Set "state" as "truncated" if the plot is truncated in this chunk, or "finished" otherwise. You will be provided with the truncated plots from the previous chunk, and you **must** extend them with the current chunk.

3. Summarize each plot (including the completed truncated ones). Just summarize, do not extend unrelated discussions. Score its prominence from 1 to 100 (how well-known this plot is to readers of the book).

4. List the key characters of each plot. For each: "name" (canonical name, see rule 5), "description" (~20 words, the character before this plot), "experience" (~30 words, the character's role, thoughts and behaviors in this plot, and any significant character development).

5. [IMPORTANT] Character naming: a registry of previously identified characters is provided below. When a registry character appears in this chunk, you MUST use their canonical name in "name". If a registry character appears under a new alias (e.g., a nickname), report it in "character_registry_updates". Only create a new registry entry for genuinely new characters.

6. Identify the optimal starting point for the subsequent chunk. If the last storyline has been extracted as a truncated plot, set next_chunk_start as None. Otherwise, set next_chunk_start as the first sentence of the last storyline.

===Output Format===
Strict JSON (no markdown fences, no comments):
{{
    "chapter_beginnings": [
        {{"beginning_sentence": "Exactly the first line of the chapter (namely the title)."}}
    ],
    "plots": [
        {{
            "chapter_title": "The chapter title that the plot belongs to. Null if not found.",
            "first_sentence": "Exactly the first sentence of the plot in this **chunk**.",
            "last_sentence": "Exactly the last sentence of the plot in this **chunk**. If the plot is truncated, the last sentence of this chunk.",
            "prominence": 95,
            "summary": "The summary of the plot.",
            "key_characters": [
                {{"name": "Canonical character name", "description": "~20 words", "experience": "~30 words"}}
            ],
            "state": "finished" or "truncated"
        }}
    ],
    "character_registry_updates": [
        {{"canonical": "Canonical name", "new_alias": "Alias seen in this chunk"}}
    ],
    "next_chunk_start": "The first sentence of the next chunk, or null."
}}

===Requirements===
1. Adhere strictly to the specified output JSON format. Ensure all DOUBLE QUOTES within all STRINGS are properly ESCAPED.
2. Maintain story fidelity: plots must accurately reflect the book's content. Do not invent events.
3. Keep all output in the same language as the source text.

===Input===

==Book title==
{title}

==Author==
{author}

==Character registry (use these canonical names)==
{registry}

==Truncated plots from previous chunk (to be finished)==
{truncated_plots}

==Chunk of Book Content==
{chunk}
"""

# ---------------------------------------------------------------------------
# Stage A.6a: alias grouping
# ---------------------------------------------------------------------------

ALIAS_GROUP_PROMPT = """You are given a list of character names that appear in the novel "{title}". Group the names that refer to the same person, and choose the most complete formal name as the canonical name.

Rules:
1. Honorifics (Mr./Mrs./Miss/Lady/Sir/Dr.) do not make a different person: "Mr. Darcy" and "Fitzwilliam Darcy" are the same person.
2. Nicknames and diminutives (e.g., "Lizzy", "Eliza") belong to their formal name.
3. Family members sharing a surname are DIFFERENT people: "Jane Bennet" and "Elizabeth Bennet" are different.
4. If a bare name is genuinely ambiguous (e.g., "Miss Bennet" could be any unmarried daughter), put it in "ambiguous" instead of guessing.
5. Do not invent characters that are not in the list.

===Input==
Names:
{names}

===Output Format===
Strict JSON (no markdown fences, no comments):
{{
    "groups": [
        {{"canonical": "Elizabeth Bennet", "aliases": ["Lizzy", "Eliza", "Miss Elizabeth Bennet"]}}
    ],
    "ambiguous": ["Miss Bennet"]
}}
"""

# ---------------------------------------------------------------------------
# Stage B.1: scene splitting (two-pass mode)
# ---------------------------------------------------------------------------

SCENE_SPLIT_PROMPT = """Based on the provided plot text from a novel, split it into scenes: continuous interactions in one setting.

For each scene, provide:
- "description": the setting and situation at the start of the scene.
- "participants": the characters present, using the canonical names provided below.
- "relationship_context": each participant's relationship to the others.
- "first_sentence": EXACTLY the first sentence of this scene in the plot text (verbatim, for locating the scene).

Rules:
1. Scenes must be consecutive and together cover the whole plot text.
2. A new scene starts when the setting changes or the interaction shifts to a different focus.
3. If the whole plot is one continuous interaction, return a single scene.
4. Keep all output in the same language as the source text.

===Output Format===
Strict JSON (no markdown fences, no comments):
{{
  "scenes": [
    {{
      "scene_id": 0,
      "description": "...",
      "participants": ["CharacterA"],
      "relationship_context": {{"CharacterA": "..."}},
      "first_sentence": "..."
    }}
  ]
}}

===Input===

==Book title==
{title}

==Plot summary==
{summary}

==Key characters==
{key_characters}

==Plot text==
{text}
"""

# ---------------------------------------------------------------------------
# Stage B: T-A-O trajectory extraction (v2)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Based on the provided plot text from a novel, extract the behavioral trajectories of the characters as Thought-Action-Observation (T-A-O) loops.

===Background===

We study how characters in novels make DECISIONS through their actions. Your job is to find every moment where a character makes a meaningful choice, and record it as a loop:
- Thought: the character's first-person inner reasoning that drives the choice
- Action: what the character decides to do
- Observation: the feedback the character perceives after acting

===What counts as an Action (CRITICAL)===

An Action is a decision-bearing behavior — a moment where the character has multiple options and chooses one. Include:
1. Decisions that change the situation: e.g., deciding to leave, refusing a request, disowning someone, drawing a weapon.
2. Decisional speech: when the act of speaking itself performs a choice — refusing, pressuring, promising, deceiving, mocking, persuading, threatening, deliberately withholding information.
3. Deliberate inaction: choosing to stay silent or not to intervene, when it is a meaningful choice.

Do NOT include:
1. Pure information exchange or passive answers (e.g., answering "What is his name?" with "Bingley").
2. Narrative descriptions of a character's personality (e.g., "She was a woman of mean understanding").
3. Accompanying gestures that merely decorate speech (e.g., "(smiles)", "(frowns)") — unless the gesture itself IS the decision (e.g., deliberately laying a hand on one's sword).

===Atomicity Rule===

Each step contains exactly ONE atomic decision. If the text describes several decisions in a row (e.g., "he resolved to repay the debt, then wrote a letter of approval"), split them into consecutive steps. Do not merge multiple decisions into one step.

===Field Requirements===

- "thought": Write in FIRST PERSON from the character's perspective, revealing the motivation and reasoning BEHIND the action. One or two sentences. [IMPORTANT] If the narration describes the character's mental state (e.g., "he wished", "she was determined", "he felt", "it gave him pleasure"), you MUST base the thought on it, faithfully rewriting it into first person. NEVER invent inner states that have no basis in the text.
- "thought_evidence": A short VERBATIM quote from the plot text that supports the thought. Quote the narration or speech it is grounded in. If no direct quote exists, quote the closest relevant passage.
- "action": Describe the FUNCTION of the behavior — what choice it accomplishes (e.g., "Refuses to commit to the visit and mocks his wife by suggesting Bingley might prefer her"), NOT a verbatim quote of the words spoken. One or two sentences.
- "observation": The feedback THIS character perceives after acting:
  * In scenes with other people: their visible reactions, spoken replies, or changes in the environment.
  * In solo scenes, prefer (in order): (1) the direct environmental result the character perceives — what they see, hear, or discover after acting; (2) new information the character learns as a direct result of the action (e.g., after reading the terms, he learns the cost is small).
  * If neither exists (e.g., a purely internal decision whose effect unfolds later), set "observation" to null. Do NOT fabricate one.
  * NEVER write as observation: the narrator's omniscient commentary (e.g., "he felt relieved", "little did he know..."), events that happen later, other characters' unexpressed inner thoughts, or the character's own subsequent thoughts.

===Character Knowledge Rules===

Character contexts are provided below. When writing a character's thought:
1. The character only knows events they personally experienced (listed in their context) or common knowledge. They do NOT know events in their "did NOT witness" list, nor any later events.
2. Do NOT use any knowledge of this novel from outside the provided text. The characters cannot anticipate the future.
3. Keep the thought consistent with the personality and attitudes shown in that character's prior experiences.

===Steps===

1. Identify scene(s) in the plot: continuous interactions in one setting. For each scene, describe the setting, list the participants, and note their relationships.
2. For each key character in the scene, extract their trajectory: an ordered sequence of T-A-O steps following the narrative order. Each step = exactly one decision. If a character makes no meaningful decision in a scene, omit that character's trajectory for that scene.{focus_instruction}

===Output Format===

Strict JSON (no markdown fences, no comments):
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
          "relationship_context": {{"CharacterA": "A's relationship to the other participants"}},
          "trajectories": [
            {{
              "character": "CharacterA",
              "steps": [
                {{
                  "step_id": 0,
                  "thought": "...",
                  "thought_evidence": "...",
                  "action": "...",
                  "observation": "..." or null
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
3. Stay faithful to the original text. Do not invent events, thoughts, or reactions that have no basis in the text.
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


def build_extraction_prompt(title, author, text, chapter_title=None,
                            plot_summary=None, key_characters=None,
                            character_context=None, focus_character=None):
    """Render EXTRACTION_PROMPT with optional plot metadata and character context.
    If focus_character is given, instruct the model to extract ONLY that character's
    trajectory (used by the per-character fallback for dense scenes)."""
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

    focus_instruction = ''
    if focus_character:
        focus_instruction = (
            f'\n3. [IMPORTANT] Extract ONLY the trajectory of "{focus_character}". '
            'Do NOT extract trajectories for any other character, even if they also '
            'make decisions in the text.')

    return EXTRACTION_PROMPT.format(
        title=title,
        author=author,
        chapter_title=chapter_title or 'Unknown',
        plot_context=plot_context,
        character_context=ctx,
        focus_instruction=focus_instruction,
        text=text,
    )
