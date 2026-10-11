# Build one chronologically ordered action timeline per resolved character entity.
# Pure program step: no LLM calls and no source trajectory mutation.
# Usage:
#   cd repository-root
#   python -m actor.trajectory_extraction.build_character_timelines --title "Book" \
#     --traj-dir "data/trajectories_auto/Book" \
#     --timeline "data/timelines/Book.auto.final.json" \
#     --entities "data/entities/Book.entities.json" \
#     --output-dir "data/character_timelines/Book"
import argparse
import json
import os
import shutil
from collections import defaultdict



def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def make_action_uid(plot_id, scene_id, trajectory_index, step_id):
    return (f'p{plot_id:03d}_s{scene_id:03d}_'
            f't{trajectory_index:03d}_a{step_id:03d}')


def main():
    ap = argparse.ArgumentParser(description='build chronologically ordered entity action timelines')
    ap.add_argument('--title', required=True)
    ap.add_argument('--traj-dir', required=True)
    ap.add_argument('--timeline', required=True)
    ap.add_argument('--entities', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    timeline = load_json(args.timeline)
    scene_index = {
        (scene['plot_id'], scene['scene_id']): scene
        for scene in timeline['scenes']
    }
    entities_data = load_json(args.entities)
    source_to_entity = {
        item['source_character']: item
        for item in entities_data['source_character_mappings']
    }
    entity_meta = {entity['entity_id']: entity for entity in entities_data['entities']}

    grouped = defaultdict(list)
    unresolved_actions = []
    source_action_count = 0
    effective_critical_count = 0
    seen_uids = set()

    for filename in sorted(os.listdir(args.traj_dir)):
        if not filename.startswith('plot_') or not filename.endswith('.json'):
            continue
        data = load_json(os.path.join(args.traj_dir, filename))
        plot_id = data['plot_id']
        for plot in data.get('plots', []):
            for scene in plot.get('scenes', []):
                scene_id = scene.get('scene_id')
                scene_meta = scene_index.get((plot_id, scene_id))
                if scene_meta is None:
                    raise ValueError(f'missing timeline scene p{plot_id}_s{scene_id}')
                for trajectory_index, trajectory in enumerate(scene.get('trajectories', [])):
                    source_character = trajectory.get('character')
                    mapping = source_to_entity.get(source_character)
                    for step in trajectory.get('steps', []):
                        source_action_count += 1
                        step_id = step.get('step_id')
                        action_uid = make_action_uid(plot_id, scene_id,
                                                     trajectory_index, step_id)
                        if action_uid in seen_uids:
                            raise ValueError(f'duplicate action_uid: {action_uid}')
                        seen_uids.add(action_uid)
                        effective_critical = (
                            step.get('is_critical') is True and
                            step.get('audit_verdict') == 'keep'
                        )
                        item = {
                            'action_uid': action_uid,
                            'source_character': source_character,
                            'entity_id': mapping['entity_id'] if mapping else None,
                            'canonical_name': mapping['canonical_name'] if mapping else None,
                            'scene_uid': scene_meta['scene_uid'],
                            'plot_id': plot_id,
                            'scene_id': scene_id,
                            'text_order': scene_meta['text_order'],
                            'timeline_order': scene_meta['timeline_order'],
                            'time_text': scene_meta.get('time_text'),
                            'time_evidence': scene_meta.get('time_evidence'),
                            'scene_description': scene.get('description'),
                            'trajectory_index': trajectory_index,
                            'step_id': step_id,
                            'thought': step.get('thought'),
                            'thought_evidence': step.get('thought_evidence'),
                            'action': step.get('action'),
                            'observation': step.get('observation'),
                            'counterfactual_action': step.get('counterfactual_action', []),
                            'is_critical': step.get('is_critical'),
                            'criticality_reason': step.get('criticality_reason'),
                            'audit_verdict': step.get('audit_verdict'),
                            'audit_why': step.get('audit_why'),
                            'effective_critical': effective_critical,
                        }
                        if effective_critical:
                            effective_critical_count += 1
                        if mapping is None or mapping['status'] != 'resolved':
                            unresolved_actions.append(item)
                        else:
                            grouped[mapping['entity_id']].append(item)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    output_action_count = 0
    output_effective_critical = 0
    index_entities = []
    for entity_id, meta in sorted(entity_meta.items(),
                                  key=lambda item: item[1]['canonical_name']):
        actions = grouped.get(entity_id, [])
        actions.sort(key=lambda item: (
            item['timeline_order'], item['text_order'],
            item['trajectory_index'], item['step_id'], item['action_uid'],
        ))
        previous = -1
        for action_index, action in enumerate(actions):
            if action['timeline_order'] < previous:
                raise ValueError(f'non-monotonic timeline for {entity_id}')
            previous = action['timeline_order']
            action['character_action_order'] = action_index
        entity_critical = sum(a['effective_critical'] for a in actions)
        output_action_count += len(actions)
        output_effective_critical += entity_critical
        payload = {
            'book': args.title,
            'entity_id': entity_id,
            'canonical_name': meta['canonical_name'],
            'source_names': meta['source_names'],
            'entity_reason': meta.get('reason', ''),
            'action_count': len(actions),
            'effective_critical_count': entity_critical,
            'actions': actions,
        }
        with open(os.path.join(args.output_dir, f'{entity_id}.json'),
                  'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        index_entities.append({
            'entity_id': entity_id,
            'canonical_name': meta['canonical_name'],
            'source_names': meta['source_names'],
            'action_count': len(actions),
            'effective_critical_count': entity_critical,
            'file': f'{entity_id}.json',
        })

    if source_action_count != output_action_count + len(unresolved_actions):
        raise ValueError('action coverage mismatch after entity grouping')
    if effective_critical_count != output_effective_critical + sum(
            a['effective_critical'] for a in unresolved_actions):
        raise ValueError('effective critical coverage mismatch')

    report = {
        'book': args.title,
        'source_action_count': source_action_count,
        'timeline_action_count': output_action_count,
        'unresolved_action_count': len(unresolved_actions),
        'source_effective_critical_count': effective_critical_count,
        'timeline_effective_critical_count': output_effective_critical,
        'unresolved_effective_critical_count': sum(
            a['effective_critical'] for a in unresolved_actions),
        'entity_count': len(index_entities),
        'entities': index_entities,
    }
    with open(os.path.join(args.output_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if unresolved_actions:
        with open(os.path.join(args.output_dir, 'unresolved_actions.json'),
                  'w', encoding='utf-8') as f:
            json.dump({'book': args.title, 'actions': unresolved_actions}, f,
                      ensure_ascii=False, indent=2)

    print(f"entities={len(index_entities)}, actions={output_action_count}, "
          f"effective_critical={output_effective_critical}, "
          f"unresolved_actions={len(unresolved_actions)}")
    print(f'output: {args.output_dir}')


if __name__ == '__main__':
    main()
