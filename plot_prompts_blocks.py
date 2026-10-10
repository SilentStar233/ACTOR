# Block-ID plot-detection prompt: identical segmentation task to
# PLOT_DETECTION_PROMPT_V2, but boundaries are referenced by block IDs
# instead of verbatim sentences. The supplied chunk text has every block
# prefixed with [B00012]-style IDs; the model must cite those IDs exactly.

PLOT_DETECTION_PROMPT_BLOCKS = """Based on the provided book chunk, partition the narratively relevant text into consecutive, NON-OVERLAPPING plot segments for downstream character-action extraction.

The chunk text is divided into blocks, each prefixed with an ID like [B00012]. Refer to segment boundaries by these block IDs — never quote the text.

Tasks:
1. Recognize chapter beginnings if present.
2. Partition the chunk into important plot segments. For every segment, return its first and last block IDs, chapter title, summary, prominence, and key character names. Do not write character descriptions or plot experiences; those are generated later after the boundaries are stable.
3. Maintain the supplied canonical character registry and report newly observed aliases.
4. Continue any truncated segment from the previous chunk before creating new segments: keep its first_block unchanged and extend its last_block to a block inside THIS chunk.
5. SKIP matter that is ABOUT the book rather than PART of the story — never include it in any plot segment: copyright and publication pages, title pages, tables of contents, dedications, prefaces/forewords/introductions written by an editor, translator, or scholar about the work, the author's own notes about the text, critical essays, editorial commentary, annotations and footnotes, study questions and exercises, glossaries, appendices, and advertisements.
   However, text that is PART of the story MUST be kept, even when it looks like a preface: a narrator's or character's framing account, fictional letters or documents that open the story (e.g., an explorer's letters introducing the tale), a first-person introduction in a story character's voice, epigraphs quoted inside the narrative, and prologues in which characters speak or events occur. When unsure whether a passage belongs to the story, KEEP it.

===Non-overlap rules (CRITICAL)===

- These plots are EXTRACTION SEGMENTS, not arbitrary semantic plot labels.
- Every block of source text may belong to AT MOST ONE output plot.
- Output plots in source-text order. Plot A's last_block must precede Plot B's first_block.
- NEVER output both a broad parent storyline and a smaller storyline contained inside it.
- If a broad storyline contains independently meaningful episodes (different immediate goals, settings, or outcomes), output the smaller consecutive episodes only.
- If those episodes cannot stand alone, output only the broad segment and do not output its internal subplots.
- Adjacent plot segments may touch at a boundary but must not share a block.

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
  "chapter_beginnings": [{{"beginning_block": "B00012"}}],
  "plots": [
    {{
      "chapter_title": "..." or null,
      "first_block": "B00012",
      "last_block": "B00045",
      "prominence": 1,
      "summary": "...",
      "key_character_names": ["Canonical name", "..."],
      "state": "finished" or "truncated"
    }}
  ],
  "character_registry_updates": [
    {{"canonical": "Canonical name", "new_alias": "Alias"}}
  ]
}}

===Requirements===
1. Preserve source fidelity; do not invent events.
2. first_block/last_block must be IDs that appear in THIS chunk, copied exactly (a continued truncated plot keeps the first_block reported earlier).
3. Block IDs have leading zeros and fixed width: B00853 and B853 are DIFFERENT IDs, and B853 does not exist. Copy every ID character-for-character, including all leading zeros.
4. first_block must not come after last_block within a segment.
5. Use the same language as the source for summaries.
6. Check the final plot list for containment or overlap before answering.

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


OVERLAP_REPAIR_PROMPT_BLOCKS = """The following source-text region was assigned to overlapping plot segments. Repartition the ENTIRE region into consecutive, non-overlapping extraction segments.

The region is given as blocks prefixed with IDs like [B00012]. Cite boundaries by block ID only — never quote the text.

Rules:
1. Cover the supplied region from its first block to its last block exactly once: no gaps and no shared blocks.
2. Segments must be in source order and mutually exclusive.
3. Do not output both a broad parent segment and a contained child segment.
4. Start a new segment only when there is a meaningful change in immediate goal, setting, interaction, or outcome.
5. For every segment, provide the block ID at which it starts. The FIRST segment must start at the region's first block. Segments are consecutive: each segment ends at the block right before the next segment's first block, and the last segment ends at the region's last block.
6. Keep the number of segments parsimonious; do not fragment every block.
7. Block IDs have leading zeros and fixed width (B00853, not B853): copy them character-for-character, including all leading zeros.

Output strict JSON only:
{{
  "segments": [
    {{
      "first_block": "B00012",
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

==Region to repartition==
{region}
"""
