# Block-ID alignment: resolve first_block/last_block to source spans by
# pure table lookup. No fuzzy matching — an unknown or misordered block ID
# is a hard validation error and stops the pipeline.
#
# Output schema matches align_plots_v2.py (source_start/source_end/text/
# first_sentence/last_sentence/source_alignment) so downstream stages
# (check_plot_overlaps, repair, extraction) work unchanged.
import argparse
import hashlib
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from assemble_plots import MAX_PLOT_CHARS
from extract import load_book

from blocks import load_blocks


_ID_RE = re.compile(r'^\[?B0*(\d+)\]?$')


def align_plots(content, blocks, plots):
    # The detect LLM sometimes drops zero-padding on block IDs ('B853'
    # instead of canonical 'B00853'), especially late in long books. Resolve
    # any B<digits> spelling to the table's canonical key; unknown or
    # misordered IDs remain hard errors.
    by_number = {}
    for key in blocks:
        m = _ID_RE.match(key)
        if m:
            by_number[int(m.group(1))] = key
    fixes = []

    def resolve(raw):
        if raw in blocks:
            return raw
        m = _ID_RE.match(str(raw or '').strip())
        if not m:
            return None
        canonical = by_number.get(int(m.group(1)))
        if canonical is not None:
            fixes.append((raw, canonical))
        return canonical

    aligned = []
    unresolved = []
    for plot in sorted(plots, key=lambda item: item['plot_id']):
        item = dict(plot)
        for field in ('source_start', 'source_end', 'text', 'source_alignment'):
            item.pop(field, None)
        raw_first = item.get('first_block')
        raw_last = item.get('last_block') or raw_first
        first_id = resolve(raw_first)
        last_id = resolve(raw_last) or first_id
        first = blocks.get(first_id) if first_id else None
        last = blocks.get(last_id) if last_id else None
        error = None
        if first is None:
            error = f'unknown first_block {raw_first!r}'
        elif last is None:
            error = f'unknown last_block {raw_last!r}'
        elif last['end'] <= first['start']:
            error = f'misordered blocks {first_id}..{last_id}'
        if error:
            unresolved.append({'plot_id': item['plot_id'], 'reason': error,
                               'first_block': raw_first, 'last_block': raw_last})
            item['source_alignment'] = {'status': 'unresolved'}
            aligned.append(item)
            continue
        item['first_block'] = first_id
        item['last_block'] = last_id
        source_start, source_end = first['start'], last['end']
        item['source_start'] = source_start
        item['source_end'] = source_end
        item['text'] = content[source_start:source_end].strip()
        item['first_sentence'] = content[first['start']:first['end']].strip()
        item['last_sentence'] = content[last['start']:last['end']].strip()
        item['source_alignment'] = {
            'status': 'resolved',
            'method': 'block_id',
            'oversize': source_end - source_start > MAX_PLOT_CHARS,
        }
        aligned.append(item)
    return aligned, unresolved, fixes


def main():
    ap = argparse.ArgumentParser(
        description='resolve block-ID plot boundaries to source spans')
    ap.add_argument('--src', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--input', required=True, help='detected plot JSON')
    ap.add_argument('--output', required=True, help='aligned plot JSON')
    ap.add_argument('--report', required=True)
    ap.add_argument('--coverage-warn', type=float, default=0.7,
                    help='warn when plot spans cover less than this fraction of the book')
    args = ap.parse_args()
    if not 0 < args.coverage_warn <= 1:
        raise ValueError('--coverage-warn must be in (0, 1]')

    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)
    blocks_path = os.path.join(
        os.path.dirname(args.input),
        data.get('blocks_file', f'{args.title}.blocks.json'))
    blocks = load_blocks(blocks_path)
    book = load_book(args.src, args.title)
    aligned, unresolved, id_fixes = align_plots(
        book['content'], blocks, data['plots'])
    if id_fixes:
        print(f'normalized {len(id_fixes)} non-canonical block IDs '
              f'(e.g. {id_fixes[0][0]!r} -> {id_fixes[0][1]!r})')

    oversize = [p['plot_id'] for p in aligned
                if p.get('source_alignment', {}).get('oversize')]
    resolved_spans = [(p['source_start'], p['source_end']) for p in aligned
                      if p.get('source_alignment', {}).get('status') in
                      ('resolved', 'repaired')]
    covered = sum(end - start for start, end in resolved_spans)
    coverage = covered / len(book['content']) if book['content'] else 0.0
    coverage_warning = bool(resolved_spans) and coverage < args.coverage_warn
    output = dict(data)
    output['plots'] = aligned
    output['source_alignment'] = {
        'content_sha256': hashlib.sha256(
            book['content'].encode('utf-8')).hexdigest(),
        'boundary_protocol': 'block_id',
        'plot_count': len(aligned),
        'resolved_count': len(aligned) - len(unresolved),
        'unresolved_count': len(unresolved),
        'oversize_plot_ids': oversize,
        'coverage': round(coverage, 4),
        'coverage_warning': coverage_warning,
    }
    report = {'book': args.title, **output['source_alignment'],
              'unresolved_plots': unresolved}
    for path, payload in ((args.output, output), (args.report, report)):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"aligned={report['resolved_count']}/{report['plot_count']} "
          f"unresolved={report['unresolved_count']} oversize={len(oversize)} "
          f"coverage={coverage:.1%}")
    if coverage_warning:
        print(f'WARNING: plot coverage {coverage:.1%} < {args.coverage_warn:.0%}; '
              'check whether narrative front/back matter was mistakenly skipped')
    print(f'aligned plots: {args.output}')
    print(f'report: {args.report}')
    if unresolved:
        raise SystemExit('Block-ID alignment unresolved: stop before overlap checks.')


if __name__ == '__main__':
    main()
