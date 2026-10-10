# v2 plot prompts: plots are extraction segments, not arbitrary nested semantic
# storylines. Every source passage may belong to at most one output plot.

PLOT_DETECTION_PROMPT_V2 = """Based on the provided book chunk, partition the narratively relevant text into consecutive, NON-OVERLAPPING plot segments for downstream character-action extraction.

Tasks:
1. Recognize chapter beginnings if present.
2. Partition the chunk into important plot segments. For every segment, return its exact first and last sentences, chapter title, summary, prominence, and key character names. Do not write character descriptions or plot experiences; those are generated later after the boundaries are stable.
3. Maintain the supplied canonical character registry and report newly observed aliases.
4. Continue any truncated segment from the previous chunk before creating new segments.

===Non-overlap rules (CRITICAL)===

- These plots are EXTRACTION SEGMENTS, not arbitrary semantic plot labels.
- Every passage of source text may belong to AT MOST ONE output plot.
- Output plots in source-text order. Plot A's last sentence must occur before Plot B's first sentence.
- NEVER output both a broad parent storyline and a smaller storyline contained inside it.
- If a broad storyline contains independently meaningful episodes (different immediate goals, settings, or outcomes), output the smaller consecutive episodes only.
- If those episodes cannot stand alone, output only the broad segment and do not output its internal subplots.
- Adjacent plot segments may touch at a boundary but must not share a sentence.

Incorrect:
- Plot A: A character returns home through a companion's departure.
- Plot B: The character tells a friend about a recent argument (inside A).
- Plot C: The character's final talk with a rival (inside A).

Correct:
- Plot A: The character returns home.
- Plot B: The character tells a friend about a recent argument.
- Plot C: The household reacts to the companion's departure.
- Plot D: The character's final talk with a rival.
- Plot E: A sibling leaves for a distant town.

===Output Format===
Strict JSON only:
{{
  "chapter_beginnings": [{{"beginning_sentence": "Exact chapter opening"}}],
  "plots": [
    {{
      "chapter_title": "..." or null,
      "first_sentence": "Exact first sentence of this segment",
      "last_sentence": "Exact last sentence of this segment",
      "prominence": 1,
      "summary": "...",
      "key_character_names": ["Canonical name", "..."],
      "state": "finished" or "truncated"
    }}
  ],
  "character_registry_updates": [
    {{"canonical": "Canonical name", "new_alias": "Alias"}}
  ],
  "next_chunk_start": "Exact first sentence of the next chunk" or null
}}

===Requirements===
1. Preserve source fidelity; do not invent events.
2. All first/last sentences must be verbatim source text.
3. Use the same language as the source.
4. Check the final plot list for containment or overlap before answering.

==Book title==
{title}

==Author==
{author}

==Character registry==
{registry}

==Truncated plots from previous chunk==
{truncated_plots}

==Book chunk==
{chunk}
"""


OVERLAP_REPAIR_PROMPT = """The following source-text region was assigned to overlapping plot segments. Repartition the ENTIRE region into consecutive, non-overlapping extraction segments.

Rules:
1. Cover the supplied source region from beginning to end exactly once: no gaps and no duplicated passage.
2. Segments must be in source order and mutually exclusive.
3. Do not output both a broad parent segment and a contained child segment.
4. Start a new segment only when there is a meaningful change in immediate goal, setting, interaction, or outcome.
5. For every segment, provide an EXACT verbatim first sentence. The first segment must begin at the start of the supplied region. These first-sentence anchors define the deterministic boundaries; the program will end each segment where the next begins.
6. Keep the number of segments parsimonious; do not fragment every paragraph.

Output strict JSON only:
{{
  "segments": [
    {{
      "first_sentence": "Exact first sentence",
      "chapter_title": "..." or null,
      "prominence": 1,
      "summary": "...",
      "key_character_names": ["Canonical name", "..."]
    }}
  ]
}}

==Book==
{title} by {author}

==Overlapping old plots (reference only; do not preserve their boundaries)==
{old_plots}

==Source region to repartition==
{region}
"""
