# Stage 0: split a book into chunks, respecting chapter boundaries when possible.
import re

# Chapter heading patterns (borrowed from CoSER's split.py, char-based).
_WORD_NUM = ('One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|'
             'Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|'
             'Thirty|Forty|Fifty|Sixty|Seventy|Eighty|Ninety|Hundred')
CHAPTER_PATTERNS = [
    rf'\n\s*(?:Chapter|CHAPTER|Prologue|Epilogue|Afterword|Preface|Introduction|'
    rf'Conclusion|Appendix|Interlude|Part|PART|Book|ACT|Act|Scene|SCENE)'
    rf'\s+(?:\d+|[IVXLCDM]+|(?:{_WORD_NUM}))\.?\s*\n',
    r'\n\s*(?:Chapter|CHAPTER)\s+.*\n',
    rf'\n\s*(?:{_WORD_NUM})\s*\n',
    r'\n\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\s*\n',
]
_CHAPTER_RE = re.compile('|'.join(f'(?:{p})' for p in CHAPTER_PATTERNS))


def split_chapters(content, min_chapter_chars=2000):
    """Split content into chapters by heading patterns. Returns list[str] or None."""
    matches = list(_CHAPTER_RE.finditer(content))
    if len(matches) < 3:
        return None

    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        if i == 0 and m.start() > 0:
            preface = content[:m.start()].strip()
            if preface:
                chapters.append(preface)
        chapters.append(content[start:end].strip())

    # Merge tiny chapters into the next one (title pages, dedications, etc.)
    merged = []
    buf = ''
    for ch in chapters:
        buf = (buf + '\n' + ch) if buf else ch
        if len(buf) >= min_chapter_chars:
            merged.append(buf)
            buf = ''
    if buf:
        if merged:
            merged[-1] += '\n' + buf
        else:
            merged.append(buf)
    return merged or None


def _hard_cut(text, chunk_size):
    """Cut text at paragraph boundaries near chunk_size."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # prefer to cut at a blank line within the last 20% of the window
            window = text[start:end]
            cut = window.rfind('\n\n', int(len(window) * 0.8))
            if cut != -1:
                end = start + cut + 2
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def create_chunks(content, chunk_size=60000):
    """Split book content into chunks of roughly chunk_size characters.

    Chapter-aligned when chapters are detectable; hard-cut at paragraph
    boundaries otherwise (or when a single chapter exceeds 2x chunk_size).
    """
    chapters = split_chapters(content)
    if chapters is None:
        return _hard_cut(content, chunk_size)

    chunks = []
    buf, buf_len = [], 0
    for ch in chapters:
        buf.append(ch)
        buf_len += len(ch)
        if buf_len >= chunk_size / 2:
            if buf_len <= 2 * chunk_size:
                chunks.append('\n'.join(buf))
                buf, buf_len = [], 0
            else:
                # oversized: flush at hard boundaries
                chunks.extend(_hard_cut('\n'.join(buf), chunk_size))
                buf, buf_len = [], 0
    if buf:
        chunks.append('\n'.join(buf))
    return chunks
