# Stage A.5: slice each plot's full text from the book using its first/last sentences.
import argparse
import json
import os

from extract import load_book

MAX_PLOT_CHARS = 40000

# 1-char -> 1-char normalization so positions are preserved.
_QUOTE_MAP = str.maketrans({
    '“': '"', '”': '"', '‘': "'", '’': "'",
    '—': '-', '–': '-', ' ': ' ',
})


def canon(s):
    return s.translate(_QUOTE_MAP)


def normalize(s):
    return ' '.join(s.split())


def find_forward(content, sentence, start_pos, end_pos):
    """Locate sentence in content[start_pos:end_pos]. Returns (start, end) or None.
    Exact match first, then whitespace-normalized prefix anchor."""
    idx = content.find(sentence, start_pos, end_pos)
    if idx != -1:
        return idx, idx + len(sentence)

    norm_sent = normalize(sentence)
    if not norm_sent:
        return None
    prefix = norm_sent[:40]
    first_word = prefix.split()[0]
    pos = start_pos
    while pos < end_pos:
        idx = content.find(first_word, pos, end_pos)
        if idx == -1:
            return None
        window = normalize(content[idx:idx + len(sentence) + 50])
        if window.startswith(prefix):
            return idx, idx + len(sentence)
        pos = idx + 1
    return None


def assemble_plot_texts(book_content, plots):
    """Attach 'text' to each plot. Plots are ordered; each plot's last sentence is
    searched within a bounded window, with the next plot's start as fallback end."""
    content = canon(book_content)
    pos = 0
    issues = []

    # pre-compute first-sentence spans (needed for the fallback end)
    first_spans = []
    for p in plots:
        first = canon(p.get('first_sentence', ''))
        span = find_forward(content, first, pos, len(content)) if first else None
        first_spans.append(span)
        if span:
            pos = span[0]

    for i, p in enumerate(plots):
        span_first = first_spans[i]
        if span_first is None:
            issues.append((p['plot_id'], 'first_sentence'))
            p['text'] = ''
            continue

        last = canon(p.get('last_sentence', '')) or canon(p.get('first_sentence', ''))
        window_end = min(span_first[0] + MAX_PLOT_CHARS, len(content))
        span_last = find_forward(content, last, span_first[0], window_end) if last else None

        if span_last is None:
            # fallback: end at the next plot's first sentence
            nxt = next((s for s in first_spans[i + 1:] if s), None)
            if nxt and nxt[0] > span_first[0]:
                span_last = (nxt[0], nxt[0])
                issues.append((p['plot_id'], 'last_sentence_fallback_next_plot'))
            else:
                span_last = (span_first[1], span_first[1])
                issues.append((p['plot_id'], 'last_sentence'))

        p['text'] = book_content[span_first[0]:span_last[1]].strip()
    return issues


def main():
    parser = argparse.ArgumentParser(description='Stage A.5: assemble plot texts')
    parser.add_argument('--src', default='data/src/all_books.jsonl')
    parser.add_argument('--title', default='Pride and Prejudice')
    parser.add_argument('--plots_dir', default='data/plots')
    args = parser.parse_args()

    plots_path = os.path.join(args.plots_dir, f'{args.title}.plots.json')
    with open(plots_path, encoding='utf-8') as f:
        data = json.load(f)

    book = load_book(args.src, args.title)
    issues = assemble_plot_texts(book['content'], data['plots'])

    with open(plots_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    lengths = sorted(len(p.get('text', '')) for p in data['plots'])
    empty = sum(1 for l in lengths if l == 0)
    huge = sum(1 for l in lengths if l > MAX_PLOT_CHARS)
    print(f'{len(lengths)} plots, text lengths: min={lengths[0]}, '
          f'median={lengths[len(lengths)//2]}, max={lengths[-1]}')
    print(f'empty: {empty}, over {MAX_PLOT_CHARS}: {huge}')
    if issues:
        print(f'issues ({len(issues)}):', issues)
    print(f'Updated: {plots_path}')


if __name__ == '__main__':
    main()
