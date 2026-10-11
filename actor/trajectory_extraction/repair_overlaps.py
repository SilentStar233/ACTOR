# Block-ID overlap repair: repartition each overlapping cluster by block
# reference. Mirrors repair_overlapping_plots.py, but segment boundaries
# are validated block IDs resolved by table lookup — no text searching.
# Input plots JSON must carry block spans (output of actor.trajectory_extraction.align_blocks).
from actor.shared.paths import ROOT_DIR

import argparse
import json
import os
import sys

_HERE = ROOT_DIR

from actor.trajectory_extraction.check_plot_overlaps import find_overlaps
from actor.shared.utils import load_book
from actor.shared.llm import LLM

from actor.trajectory_extraction.blocks import load_blocks, render_block
from actor.trajectory_extraction.prompts import OVERLAP_REPAIR_PROMPT_BLOCKS


def build_clusters(spans):
    """Connected components of the interval-overlap graph."""
    by_id = {s['plot_id']: s for s in spans}
    adjacency = {s['plot_id']: set() for s in spans}
    for overlap in find_overlaps(spans):
        a, b = overlap['left_plot_id'], overlap['right_plot_id']
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = set()
    clusters = []
    for pid, neighbours in adjacency.items():
        if pid in seen or not neighbours:
            continue
        stack = [pid]
        component = []
        seen.add(pid)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        clusters.append(sorted((by_id[x] for x in component),
                               key=lambda s: s['start']))
    return clusters


def repartition_cluster(llm, model, book, content, ordered_blocks, by_id,
                        plots_by_id, cluster):
    block_index = {block['id']: i for i, block in enumerate(ordered_blocks)}
    lo = min(block_index[plots_by_id[s['plot_id']]['first_block']]
             for s in cluster)
    hi = max(block_index[plots_by_id[s['plot_id']]['last_block']]
             for s in cluster)
    region_blocks = ordered_blocks[lo:hi + 1]
    region = '\n\n'.join(render_block(content, b) for b in region_blocks)

    old = []
    old_ids = []
    for span in cluster:
        plot = plots_by_id[span['plot_id']]
        old_ids.append(span['plot_id'])
        old.append({'plot_id': span['plot_id'],
                    'chapter_title': plot.get('chapter_title'),
                    'summary': plot.get('summary'),
                    'first_block': plot.get('first_block'),
                    'last_block': plot.get('last_block')})
    prompt = OVERLAP_REPAIR_PROMPT_BLOCKS.format(
        title=book['title'], author=book['author'],
        old_plots=json.dumps(old, ensure_ascii=False, indent=2),
        region=region,
    )
    # One feedback retry: if the program validation rejects the segments,
    # tell the model why and let it fix them once.
    last_error = None
    for attempt in range(2):
        result = llm.call_json(prompt, model=model, max_tokens=16384,
                               use_cache=(attempt == 0))
        try:
            return _build_replacements(result, old_ids, lo, hi,
                                       ordered_blocks, block_index, content)
        except ValueError as error:
            last_error = error
            if attempt == 0:
                print(f'  cluster {old_ids}: validation failed ({error}); '
                      'retrying with feedback')
                prompt += (
                    f'\n\nYour previous answer was rejected by the program '
                    f'validator: {error}\nReturn corrected segments only.')
    raise ValueError(f'cluster {old_ids}: {last_error}')


def _build_replacements(result, old_ids, lo, hi, ordered_blocks,
                        block_index, content):
    segments = result.get('segments', [])
    if not segments:
        raise ValueError(f'cluster {old_ids}: no replacement segments')

    indices = []
    for i, segment in enumerate(segments):
        bid = segment.get('first_block')
        idx = block_index.get(bid)
        if idx is None or idx < lo or idx > hi:
            raise ValueError(
                f'cluster {old_ids}: segment {i} first_block {bid!r} '
                'unknown or outside region')
        indices.append(idx)
    if indices[0] != lo:
        raise ValueError(
            f'cluster {old_ids}: first segment must start at region block '
            f'{ordered_blocks[lo]["id"]}, got {segments[0].get("first_block")}')
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(f'cluster {old_ids}: unordered/duplicate segment blocks')

    replacements = []
    for i, segment in enumerate(segments):
        first = ordered_blocks[indices[i]]
        last = (ordered_blocks[indices[i + 1] - 1]
                if i + 1 < len(indices) else ordered_blocks[hi])
        source_start, source_end = first['start'], last['end']
        replacements.append({
            'chapter_title': segment.get('chapter_title'),
            'first_block': first['id'],
            'last_block': last['id'],
            'first_sentence': content[first['start']:first['end']].strip(),
            'last_sentence': content[last['start']:last['end']].strip(),
            'prominence': segment.get('prominence', 50),
            'summary': segment.get('summary', ''),
            'key_character_names': segment.get('key_character_names', []),
            'state': 'finished',
            'text': content[source_start:source_end].strip(),
            'source_start': source_start,
            'source_end': source_end,
            'source_alignment': {'status': 'repaired', 'method': 'block_id'},
            'repaired_from_plot_ids': old_ids,
        })
    # block-level contiguity check: replacements partition the region exactly
    cursor = lo
    for replacement in replacements:
        if block_index[replacement['first_block']] != cursor:
            raise ValueError(f'cluster {old_ids}: gap/overlap in replacements')
        cursor = block_index[replacement['last_block']] + 1
    if cursor != hi + 1:
        raise ValueError(f'cluster {old_ids}: replacements do not cover region end')
    return replacements, {'old_plot_ids': old_ids,
                          'region_blocks': [ordered_blocks[lo]['id'],
                                            ordered_blocks[hi]['id']],
                          'replacement_count': len(replacements)}


def main():
    ap = argparse.ArgumentParser(
        description='block-ID overlap repair for aligned plot JSON')
    ap.add_argument('--title', required=True)
    ap.add_argument('--plots', required=True, help='aligned plot JSON')
    ap.add_argument('--src', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--model', default='gpt-5.6-luna')
    args = ap.parse_args()

    with open(args.plots, encoding='utf-8') as f:
        data = json.load(f)
    blocks_path = os.path.join(
        os.path.dirname(args.plots),
        data.get('blocks_file', f'{args.title}.blocks.json'))
    if not os.path.exists(blocks_path):
        # aligned output often lives next to the detected file's sibling dirs
        alt = os.path.join(os.path.dirname(args.plots), '..', 'plots_detected',
                           os.path.basename(blocks_path))
        blocks_path = os.path.normpath(alt)
    blocks_map = load_blocks(blocks_path)
    ordered_blocks = sorted(blocks_map.values(), key=lambda b: b['start'])
    book = load_book(args.src, args.title)
    content = book['content']

    plots = data['plots']
    spans = [{'plot_id': p['plot_id'], 'start': p['source_start'],
              'end': p['source_end'],
              'length': p['source_end'] - p['source_start']} for p in plots]
    clusters = build_clusters(spans)
    if not clusters:
        print('No plot overlaps; copied input to output.')
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return

    llm = LLM(config_path=os.path.join(ROOT_DIR, 'config.json'), book=args.title)
    plots_by_id = {p['plot_id']: p for p in plots}
    replacements_by_old_id = {}
    cluster_reports = []
    for cluster in clusters:
        replacements, report = repartition_cluster(
            llm, args.model, book, content, ordered_blocks, blocks_map,
            plots_by_id, cluster)
        old_ids = report['old_plot_ids']
        replacements_by_old_id[min(old_ids)] = (set(old_ids), replacements)
        cluster_reports.append(report)
        print(f'cluster {old_ids} -> {len(replacements)} non-overlapping plots')

    repaired = []
    consumed = set()
    for plot in plots:
        pid = plot['plot_id']
        if pid in consumed:
            continue
        if pid in replacements_by_old_id:
            old_ids, replacements = replacements_by_old_id[pid]
            repaired.extend(replacements)
            consumed.update(old_ids)
        else:
            repaired.append(dict(plot))
            consumed.add(pid)

    repaired.sort(key=lambda p: p['source_start'])
    for pid, plot in enumerate(repaired):
        plot['plot_id'] = pid
    remaining = find_overlaps([
        {'plot_id': p['plot_id'], 'start': p['source_start'],
         'end': p['source_end'], 'length': p['source_end'] - p['source_start']}
        for p in repaired])
    if remaining:
        raise ValueError(f'overlaps remain after repair: {remaining}')

    output = dict(data)
    output['plots'] = repaired
    output['plot_overlap_repair'] = {
        'model': args.model,
        'boundary_protocol': 'block_id',
        'original_plot_count': len(plots),
        'repaired_plot_count': len(repaired),
        'clusters': cluster_reports,
        'remaining_overlap_count': 0,
    }
    for path, payload in ((args.output, output),
                          (args.report, output['plot_overlap_repair'])):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'plots {len(plots)} -> {len(repaired)}, overlaps=0')
    print(f'output: {args.output}')


if __name__ == '__main__':
    main()
