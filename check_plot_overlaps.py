# Detect plot source-text interval overlaps after plot assembly and before action
# extraction. Future v2 pipelines should run this with --fail-on-overlap.
# Usage:
#   cd v2
#   python check_plot_overlaps.py
#   python check_plot_overlaps.py --fail-on-overlap
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from extract import load_book


def persisted_plot_spans(content, plots):
    """Read and validate spans created by align_plots_v2; never re-search anchors."""
    spans = []
    unresolved = []
    for plot in plots:
        plot_id = plot.get('plot_id')
        start = plot.get('source_start')
        end = plot.get('source_end')
        if not isinstance(start, int) or not isinstance(end, int):
            unresolved.append({
                'plot_id': plot_id,
                'reason': 'missing_source_span',
            })
            continue
        if not 0 <= start < end <= len(content):
            unresolved.append({
                'plot_id': plot_id,
                'reason': 'invalid_source_span',
                'source_start': start,
                'source_end': end,
            })
            continue
        source_text = content[start:end].strip()
        if source_text != (plot.get('text') or '').strip():
            unresolved.append({
                'plot_id': plot_id,
                'reason': 'source_text_mismatch',
                'source_start': start,
                'source_end': end,
            })
            continue
        spans.append({
            'plot_id': plot_id,
            'start': start,
            'end': end,
            'length': end - start,
        })
    return spans, unresolved


def locate_plot_spans(content, plots):
    """Compatibility wrapper for callers that require every persisted span."""
    spans, unresolved = persisted_plot_spans(content, plots)
    if unresolved:
        raise ValueError(f'Unresolved persisted source spans: {unresolved}')
    return spans


def find_overlaps(spans):
    overlaps = []
    for i, left in enumerate(spans):
        for right in spans[i + 1:]:
            size = min(left['end'], right['end']) - max(left['start'], right['start'])
            if size <= 0:
                continue
            if left['start'] <= right['start'] and right['end'] <= left['end']:
                relation = 'left_contains_right'
            elif right['start'] <= left['start'] and left['end'] <= right['end']:
                relation = 'right_contains_left'
            else:
                relation = 'partial_overlap'
            overlaps.append({
                'left_plot_id': left['plot_id'],
                'right_plot_id': right['plot_id'],
                'relation': relation,
                'overlap_chars': size,
                'left_coverage': round(size / left['length'], 4),
                'right_coverage': round(size / right['length'], 4),
                'left_span': [left['start'], left['end']],
                'right_span': [right['start'], right['end']],
            })
    return overlaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', default='Pride and Prejudice')
    ap.add_argument('--src', default='data/src/all_books.jsonl')
    ap.add_argument('--plots-dir', default='data/plots_final')
    ap.add_argument('--plots-file', default=None,
                    help='explicit plot JSON path; overrides --plots-dir')
    ap.add_argument('--output', default='data/logs/plot_overlap_report.json')
    ap.add_argument('--fail-on-overlap', action='store_true')
    args = ap.parse_args()

    plots_path = (args.plots_file or
                  os.path.join(args.plots_dir, f'{args.title}.plots.json'))
    with open(plots_path, encoding='utf-8') as f:
        data = json.load(f)
    book = load_book(args.src, args.title)
    spans, unresolved = persisted_plot_spans(book['content'], data['plots'])
    overlaps = find_overlaps(spans)
    safe_to_extract = not overlaps and not unresolved and len(spans) == len(data['plots'])

    report = {
        'book': args.title,
        'plot_count': len(data['plots']),
        'resolved_span_count': len(spans),
        'unresolved_span_count': len(unresolved),
        'unresolved_plots': unresolved,
        'overlap_count': len(overlaps),
        'overlaps': overlaps,
        'safe_to_extract': safe_to_extract,
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"plots={report['plot_count']}, resolved={len(spans)}, "
          f"unresolved={len(unresolved)}, overlaps={len(overlaps)}, "
          f"safe_to_extract={safe_to_extract}")
    print(f'report: {args.output}')
    if args.fail_on_overlap and not safe_to_extract:
        raise SystemExit('Plot spans are unresolved or overlap: stop before trajectory extraction.')


if __name__ == '__main__':
    main()
