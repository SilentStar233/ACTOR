# One-call, book-level entity normalization for final trajectory character names.
# Keeps source names unchanged; emits source_character -> entity_id mappings for
# later character-timeline/profile stages.
# Usage:
#   cd repository-root
#   python -m actor.trajectory_extraction.build_entities --title "Book" \
#     --traj-dir "data/trajectories_auto/Book" \
#     --timeline "data/timelines/Book.auto.final.json" \
#     --alias-map "data/contexts/Book/alias_map.json" \
#     --output "data/entities/Book.entities.json"
from actor.shared.paths import ROOT_DIR

import argparse
import json
import os
import re
import sys
from collections import defaultdict

_HERE = ROOT_DIR

from actor.shared.llm import LLM


ENTITY_PROMPT = """You are resolving persistent character entities in the novel "{title}".

Different source names may refer to one persistent person because of nicknames, titles, marriage/name changes, or different forms of address. Group ONLY names that refer to the same person over the story timeline.

Important rules:
1. Every source_name in the input must appear EXACTLY ONCE, either in one entity's source_names or in unresolved.
2. Never merge people merely because they share a surname, title, gender, household, or nearby scenes.
3. Marriage/name changes SHOULD be merged when the supplied time-ordered appearances establish continuity (e.g., a woman before and after marriage).
4. Existing alias-map values are hints, not facts: you may merge multiple existing canonical names if the novel evidence shows they are one person.
5. If identity cannot be safely established, place that exact source_name in unresolved. Do not guess.
6. canonical_name should be one of the source_names in its entity group, preferably the earliest stable full name.
7. Keep groups conservative: a wrong merge is worse than an unresolved name.
8. Before answering, re-check your output against the input list: no source_name may appear more than once across all entities and unresolved, and none may be omitted.

Output strict JSON only:
{{
  "entities": [
    {{
      "canonical_name": "one source name from this group",
      "source_names": ["exact source names from the input"],
      "reason": "brief evidence-based identity explanation"
    }}
  ],
  "unresolved": [
    {{
      "source_name": "exact source name from the input",
      "reason": "why identity is ambiguous"
    }}
  ]
}}

==Existing alias-map hints==
{alias_hints}

==Source names with time-ordered appearances==
{appearances}
"""


def slugify(name):
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return slug or 'unnamed_entity'


def read_timeline(timeline_path):
    with open(timeline_path, encoding='utf-8') as f:
        timeline = json.load(f)
    return {
        (scene['plot_id'], scene['scene_id']): scene
        for scene in timeline['scenes']
    }


def collect_appearances(traj_dir, timeline_by_scene):
    """Gather final source names and concise ordered evidence from trajectories."""
    appearances = defaultdict(list)
    source_action_count = defaultdict(int)
    total_actions = 0
    for filename in sorted(os.listdir(traj_dir)):
        if not filename.startswith('plot_') or not filename.endswith('.json'):
            continue
        with open(os.path.join(traj_dir, filename), encoding='utf-8') as f:
            data = json.load(f)
        plot_id = data['plot_id']
        for plot in data.get('plots', []):
            for scene in plot.get('scenes', []):
                scene_id = scene.get('scene_id')
                timeline = timeline_by_scene.get((plot_id, scene_id))
                if timeline is None:
                    raise ValueError(f'missing timeline entry p{plot_id}_s{scene_id}')
                for trajectory in scene.get('trajectories', []):
                    name = trajectory.get('character')
                    if not name:
                        continue
                    steps = trajectory.get('steps', [])
                    source_action_count[name] += len(steps)
                    total_actions += len(steps)
                    # One appearance record per trajectory, not per action.
                    if len(appearances[name]) < 4:
                        sample_actions = [step.get('action', '')[:160]
                                          for step in steps[:2]]
                        appearances[name].append({
                            'timeline_order': timeline['timeline_order'],
                            'time_text': timeline.get('time_text'),
                            'scene': (scene.get('description') or '')[:260],
                            'action_samples': sample_actions,
                        })
    for name in appearances:
        appearances[name].sort(key=lambda item: item['timeline_order'])
    return appearances, source_action_count, total_actions


def load_alias_hints(path, source_names):
    if not path or not os.path.exists(path):
        return {name: name for name in source_names}
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    alias_map = raw.get('alias_map', raw)
    return {name: alias_map.get(name, name) for name in source_names}


def repair_resolution(result, source_names):
    """Deterministic conservative repair for a resolution that still violates
    the partition rules after the feedback retry. Never raises:
    - a name assigned more than once keeps only its first assignment
      (entities emptied this way are dropped);
    - unknown names are dropped;
    - names the LLM omitted are appended as singleton entities (the
      conservative no-merge default, keeping their actions attributable).
    """
    source_set = set(source_names)
    seen = set()
    entities = []
    for entity in result.get('entities', []):
        names = [n for n in (entity.get('source_names') or [])
                 if n in source_set and n not in seen]
        if not names:
            continue
        seen.update(names)
        entity = dict(entity)
        entity['source_names'] = names
        if entity.get('canonical_name') not in names:
            entity['canonical_name'] = names[0]
        entities.append(entity)
    unresolved = []
    for item in result.get('unresolved', []):
        name = item.get('source_name')
        if name in source_set and name not in seen:
            seen.add(name)
            unresolved.append(item)
    for name in sorted(source_set - seen):
        entities.append({'canonical_name': name, 'source_names': [name],
                         'reason': 'auto-added singleton: LLM omitted this name'})
    return {'entities': entities, 'unresolved': unresolved}


def validate_resolution(result, source_names):
    source_set = set(source_names)
    seen = {}
    entities = result.get('entities', [])
    unresolved = result.get('unresolved', [])
    for index, entity in enumerate(entities):
        names = entity.get('source_names') or []
        canonical = entity.get('canonical_name')
        if not names:
            raise ValueError(f'entity {index}: empty source_names')
        if canonical not in names:
            raise ValueError(f'entity {index}: canonical_name must be in source_names')
        for name in names:
            if name not in source_set:
                raise ValueError(f'entity {index}: unknown source_name {name!r}')
            if name in seen:
                raise ValueError(f'source_name {name!r} appears twice')
            seen[name] = ('entity', index)
    for index, item in enumerate(unresolved):
        name = item.get('source_name')
        if name not in source_set:
            raise ValueError(f'unresolved {index}: unknown source_name {name!r}')
        if name in seen:
            raise ValueError(f'source_name {name!r} appears twice')
        seen[name] = ('unresolved', index)
    missing = sorted(source_set - set(seen))
    if missing:
        raise ValueError(f'LLM omitted source names: {missing}')


def build_output(title, result, source_names, alias_hints,
                 source_action_count, total_actions):
    entities = []
    mappings = []
    used_ids = set()
    for entity in result['entities']:
        base = slugify(entity['canonical_name'])
        entity_id = base
        suffix = 2
        while entity_id in used_ids:
            entity_id = f'{base}_{suffix}'
            suffix += 1
        used_ids.add(entity_id)
        names = sorted(entity['source_names'])
        entities.append({
            'entity_id': entity_id,
            'canonical_name': entity['canonical_name'],
            'source_names': names,
            'reason': entity.get('reason', ''),
            'source_action_count': sum(source_action_count[name] for name in names),
        })
        for name in names:
            mappings.append({
                'source_character': name,
                'entity_id': entity_id,
                'canonical_name': entity['canonical_name'],
                'alias_map_hint': alias_hints[name],
                'status': 'resolved',
            })
    unresolved = []
    for item in result['unresolved']:
        name = item['source_name']
        unresolved.append({
            'source_character': name,
            'entity_id': None,
            'canonical_name': None,
            'alias_map_hint': alias_hints[name],
            'status': 'unresolved',
            'reason': item.get('reason', ''),
            'source_action_count': source_action_count[name],
        })
    mappings.sort(key=lambda item: item['source_character'])
    unresolved.sort(key=lambda item: item['source_character'])
    resolved_actions = sum(item['source_action_count'] for item in entities)
    unresolved_actions = sum(item['source_action_count'] for item in unresolved)
    if resolved_actions + unresolved_actions != total_actions:
        raise ValueError('action coverage mismatch')
    return {
        'book': title,
        'resolution_method': 'single_book_level_llm_call',
        'entity_count': len(entities),
        'source_name_count': len(source_names),
        'source_action_count': total_actions,
        'resolved_action_count': resolved_actions,
        'unresolved_action_count': unresolved_actions,
        'entities': entities,
        'source_character_mappings': mappings,
        'unresolved': unresolved,
    }


def main():
    ap = argparse.ArgumentParser(description='one-call book-level entity normalization')
    ap.add_argument('--title', required=True)
    ap.add_argument('--traj-dir', required=True)
    ap.add_argument('--timeline', required=True)
    ap.add_argument('--alias-map', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='gpt-5.6-luna')
    args = ap.parse_args()

    timeline_by_scene = read_timeline(args.timeline)
    appearances, action_counts, total_actions = collect_appearances(
        args.traj_dir, timeline_by_scene)
    source_names = sorted(appearances)
    alias_hints = load_alias_hints(args.alias_map, source_names)
    appearance_payload = [
        {
            'source_name': name,
            'existing_alias_map_hint': alias_hints[name],
            'action_count': action_counts[name],
            'appearances': appearances[name],
        }
        for name in source_names
    ]

    llm = LLM(config_path=os.path.join(ROOT_DIR, 'config.json'), book=args.title)
    prompt = ENTITY_PROMPT.format(
        title=args.title,
        alias_hints=json.dumps(alias_hints, ensure_ascii=False, indent=2),
        appearances=json.dumps(appearance_payload, ensure_ascii=False, indent=2),
    )
    # The one-call resolution occasionally breaks the partition rules (a
    # source_name assigned twice, or names omitted). Retry once with the
    # validation error fed back before giving up.
    for attempt in range(2):
        result = llm.call_json(prompt, model=args.model, max_tokens=65536,
                               use_cache=(attempt == 0))
        try:
            validate_resolution(result, source_names)
            break
        except ValueError as e:
            if attempt == 1:
                # The feedback retry fixed one violation but may introduce
                # another (dup -> omission). Repair deterministically instead
                # of killing the book; the result is always a valid partition.
                print(f'  entity validation failed again ({e}); applying '
                      'deterministic repair')
                result = repair_resolution(result, source_names)
                validate_resolution(result, source_names)
                break
            print(f'  entity validation failed ({e}); retrying with feedback')
            prompt = (
                f'The following entity normalization for "{args.title}" is '
                f'invalid: {e}. Rules: every source_name must appear in '
                "exactly one entity or unresolved entry; canonical_name must "
                "be one of the entity's source_names; no source_name may be "
                'omitted. Return ONLY the corrected full JSON.\n\n'
                '===Invalid response===\n'
                + json.dumps(result, ensure_ascii=False, indent=2))
    output = build_output(args.title, result, source_names, alias_hints,
                          action_counts, total_actions)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"source_names={output['source_name_count']}, entities={output['entity_count']}, "
          f"unresolved={len(output['unresolved'])}")
    print(f"actions: resolved={output['resolved_action_count']}, "
          f"unresolved={output['unresolved_action_count']}")
    print(f'output: {args.output}')


if __name__ == '__main__':
    main()
