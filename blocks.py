# Deterministic block segmentation for the block-ID plot-detection protocol.
#
# A book is partitioned into consecutive "blocks" (stable, program-defined
# text units). The LLM references block IDs instead of quoting source
# sentences, so downstream alignment is a pure table lookup.
#
# Segmentation algorithm (identical for every book):
#   1. split on blank lines (\n\s*\n) into atomic spans;
#   2. any atom larger than MAX_BLOCK is recursively re-split by the next
#      delimiter level: single newline -> sentence end -> whitespace window;
#   3. consecutive atoms are greedily packed into blocks of ~TARGET_BLOCK
#      chars (never exceeding MAX_BLOCK unless an atom is unsplittable).
# This adapts to any source format: double-spaced line-wrapped text (atoms
# are 80-char lines -> packed), paragraph text (atoms are paragraphs),
# and newline-free text (giant atom -> sentence-level re-split -> packed).
# Offsets always index the ORIGINAL content string; blocks are contiguous
# up to whitespace gaps.
import json
import os
import re

TARGET_BLOCK = 1500
MAX_BLOCK = 3000

_BLANK_LINE_RE = re.compile(r'\n\s*\n')
_ABBREVIATION_RE = re.compile(
    r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sr|Jr|St|Mt|No|Nos|vs|etc|e\.g|i\.e)\.$',
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r'[.!?](?:["\'”’\)\]]+)?(?=\s|$)')


def _trim(content, start, end):
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_spans(content):
    spans = []
    start = 0
    for match in _BLANK_LINE_RE.finditer(content):
        left, right = _trim(content, start, match.start())
        if left < right:
            spans.append((left, right))
        start = match.end()
    left, right = _trim(content, start, len(content))
    if left < right:
        spans.append((left, right))
    return spans


def _newline_children(content, start, end):
    points = [m.start() for m in re.finditer(r'\n', content[start:end])]
    return _split_at_points(content, start, end, [start + p for p in points])


def _sentence_children(content, start, end):
    points = []
    for match in _SENTENCE_END_RE.finditer(content[start:end]):
        boundary = start + match.end()
        head = content[start:boundary]
        if _ABBREVIATION_RE.search(head):
            continue
        points.append(boundary)
    return _split_at_points(content, start, end, points)


def _whitespace_children(content, start, end):
    midpoint = (start + end) // 2
    candidates = [start + m.start()
                  for m in re.finditer(r'\s', content[start:end])]
    candidates = [c for c in candidates if c > start and c < end]
    if not candidates:
        return [(start, end)]
    cut = min(candidates, key=lambda pos: abs(pos - midpoint))
    return _split_at_points(content, start, end, [cut])


def _split_at_points(content, start, end, points):
    children = []
    cursor = start
    for point in points:
        left, right = _trim(content, cursor, point)
        if left < right:
            children.append((left, right))
        cursor = point
    left, right = _trim(content, cursor, end)
    if left < right:
        children.append((left, right))
    return children


def _force_split(content, start, end, out):
    if end - start <= MAX_BLOCK:
        out.append((start, end))
        return
    for splitter in (_newline_children, _sentence_children,
                     _whitespace_children):
        children = splitter(content, start, end)
        if len(children) > 1:
            for child_start, child_end in children:
                _force_split(content, child_start, child_end, out)
            return
    out.append((start, end))  # unsplittable single token; accept oversize


def _pack_blocks(spans):
    """Greedily pack consecutive atomic spans into ~TARGET_BLOCK blocks."""
    packed = []
    cur_start = cur_end = None
    for start, end in spans:
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        if (cur_end - cur_start >= TARGET_BLOCK
                or end - cur_start > MAX_BLOCK):
            packed.append((cur_start, cur_end))
            cur_start, cur_end = start, end
        else:
            cur_end = end
    if cur_start is not None:
        packed.append((cur_start, cur_end))
    return packed


def build_blocks(content):
    """Partition content into blocks; return [{'id', 'start', 'end'}, ...]."""
    coarse = _paragraph_spans(content)
    spans = []
    for start, end in coarse:
        _force_split(content, start, end, spans)
    spans = _pack_blocks(spans)
    return [
        {'id': f'B{i:05d}', 'start': start, 'end': end}
        for i, (start, end) in enumerate(spans)
    ]


def render_block(content, block):
    return f"[{block['id']}] {content[block['start']:block['end']]}"


def chunk_blocks(blocks, chunk_size):
    """Group consecutive blocks into chunks of roughly chunk_size chars."""
    chunks = []
    current = []
    current_len = 0
    for block in blocks:
        block_len = block['end'] - block['start']
        if current and current_len + block_len > chunk_size:
            chunks.append(current)
            current, current_len = [], 0
        current.append(block)
        current_len += block_len
    if current:
        chunks.append(current)
    return chunks


def render_chunk(content, chunk):
    return '\n\n'.join(render_block(content, block) for block in chunk)


def save_blocks(blocks, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'block_count': len(blocks), 'blocks': blocks},
                  f, ensure_ascii=False)


def load_blocks(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return {block['id']: block for block in data['blocks']}
