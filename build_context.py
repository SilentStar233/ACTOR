# Stage A.6: alias grouping (A.6a) + per-plot character context generation (A.6b).
import argparse
import json
import os

from llm import LLM
from prompts import ALIAS_GROUP_PROMPT
from utils import safe_title


def build_alias_map(registry, plots, title, llm=None, model=None, use_llm_cleanup=True):
    """Build name -> canonical map from the rolling registry.
    Optionally run one LLM call to merge leftovers that escaped the registry."""
    alias_map = {}
    for entry in registry:
        canon = entry['canonical']
        alias_map[canon] = canon
        for a in entry.get('aliases', []):
            alias_map[a] = canon

    # collect names in plots that are not covered by the registry
    leftover = set()
    for p in plots:
        for c in plot_key_characters(p):
            n = c.get('name')
            if n and n not in alias_map:
                leftover.add(n)

    result = {'alias_map': alias_map, 'ambiguous': []}
    if leftover and use_llm_cleanup and llm is not None:
        names = sorted(leftover)
        known = sorted(set(alias_map.values()))
        prompt = ALIAS_GROUP_PROMPT.format(
            title=title,
            names=json.dumps({'known_canonical_characters': known,
                              'unresolved_names': names},
                             ensure_ascii=False, indent=2),
        )
        try:
            resp = llm.call_json(prompt, model=model or llm.default_model)
            for g in resp.get('groups', []):
                canon = g.get('canonical')
                aliases = g.get('aliases', [])
                # target: existing canonical if any member of the group is already mapped,
                # otherwise the LLM-chosen canonical (new character)
                target = None
                for n in [canon] + aliases:
                    if n in alias_map:
                        target = alias_map[n]
                        break
                if target is None:
                    target = canon
                alias_map.setdefault(target, target)
                for n in [canon] + aliases:
                    alias_map[n] = target
            result['ambiguous'] = resp.get('ambiguous', [])
            result['leftover_resolved'] = names
        except Exception as e:
            print(f'  alias LLM cleanup failed ({e}); keeping registry-only map')
            for n in leftover:
                alias_map[n] = n
    else:
        for n in leftover:
            alias_map[n] = n
    return result


def canonical(name, alias_map):
    return alias_map.get(name, name)


def plot_key_characters(plot):
    characters = []
    for character in plot.get('key_characters') or []:
        if isinstance(character, dict) and character.get('name'):
            characters.append(character)
        elif isinstance(character, str) and character:
            characters.append({'name': character})
    for name in plot.get('key_character_names') or []:
        if name and not any(character['name'] == name for character in characters):
            characters.append({'name': name})
    return characters


def build_character_context(plots, alias_map, max_recent=10, max_anchor=5):
    """For each plot N and each of its key characters, aggregate:
    - prior_experiences: from plots 0..N-1 where the character appears
    - not_witnessed: one-line summaries of plots 0..N-1 where the character is absent
    """
    # character -> list of (plot_id, chapter, experience, prominence)
    appearances = {}
    for p in plots:
        for c in plot_key_characters(p):
            name = canonical(c.get('name'), alias_map)
            appearances.setdefault(name, []).append({
                'plot_id': p['plot_id'],
                'chapter': p.get('chapter_title') or 'Unknown',
                'experience': c.get('experience', ''),
                'prominence': p.get('prominence', 0),
            })

    contexts = {}
    for p in plots:
        n = p['plot_id']
        ctx = {}
        present = {canonical(c.get('name'), alias_map) for c in plot_key_characters(p)}
        for char in present:
            hist = [a for a in appearances.get(char, []) if a['plot_id'] < n]
            # truncation: most recent N + highest-prominence anchors
            recent = hist[-max_recent:]
            anchors = sorted(hist[:-max_recent], key=lambda a: -a['prominence'])[:max_anchor]
            prior = sorted(anchors + recent, key=lambda a: a['plot_id'])

            not_wit = [{
                'plot_id': q['plot_id'],
                'chapter': q.get('chapter_title') or 'Unknown',
                'summary': q.get('summary', '')[:150],
            } for q in plots if q['plot_id'] < n
                and char not in {canonical(c.get('name'), alias_map)
                                 for c in plot_key_characters(q)}]

            ctx[char] = {'prior_experiences': prior, 'not_witnessed': not_wit}
        contexts[str(n)] = ctx
    return contexts


def main():
    parser = argparse.ArgumentParser(description='Stage A.6: alias + character context')
    parser.add_argument('--title', default='Pride and Prejudice')
    parser.add_argument('--plots_dir', default='data/plots')
    parser.add_argument('--output_dir', default='data/contexts')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--model', default=None)
    parser.add_argument('--no_llm_cleanup', action='store_true')
    args = parser.parse_args()

    plots_path = os.path.join(args.plots_dir, f'{safe_title(args.title)}.plots.json')
    with open(plots_path, encoding='utf-8') as f:
        data = json.load(f)
    plots = data['plots']
    registry = data.get('registry', [])

    llm = None if args.no_llm_cleanup else LLM(config_path=args.config, book=args.title)

    print('Building alias map...')
    alias_result = build_alias_map(registry, plots, title=args.title, llm=llm,
                                   model=args.model,
                                   use_llm_cleanup=not args.no_llm_cleanup)
    alias_map = alias_result['alias_map']
    print(f'  {len(alias_map)} name mappings, '
          f'{len(set(alias_map.values()))} canonical characters')
    if alias_result.get('ambiguous'):
        print(f'  ambiguous: {alias_result["ambiguous"]}')

    print('Building character contexts...')
    contexts = build_character_context(plots, alias_map)

    out_dir = os.path.join(args.output_dir, safe_title(args.title))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'alias_map.json'), 'w', encoding='utf-8') as f:
        json.dump(alias_result, f, ensure_ascii=False, indent=2)
    ctx_path = os.path.join(out_dir, 'character_context.json')
    with open(ctx_path, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)

    print(f'  contexts for {len(contexts)} plots')
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    main()
