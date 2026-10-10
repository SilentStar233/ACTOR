# Generate dynamic profile trials from entity action timelines.
# Each effective critical action triggers one API request. The request receives
# the last saved profile plus the complete, full-field action window since it.
# A changed=false result saves only a tag and retains the window for later.
# Usage:
#   cd v2
#   python generate_profiles.py --top 5
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from llm import LLM
from profile_prompts import PROFILE_INITIALIZATION_PROMPT, PROFILE_UPDATE_PROMPT
from utils import atomic_json_dump

TITLE = 'Pride and Prejudice'
DEFAULT_CHARACTER_DIR = f'data/character_timelines/{TITLE}'
DEFAULT_ENTITIES = f'data/entities/{TITLE}.entities.json'
DEFAULT_PLOTS = f'data/plots_final/{TITLE}.plots.json'
DEFAULT_OUTPUT_DIR = f'data/profile_trials/{TITLE}'


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_key_characters(plot):
    characters = []
    for character in plot.get('key_characters') or []:
        if isinstance(character, dict) and character.get('name'):
            characters.append(character)
        elif isinstance(character, str) and character:
            characters.append({'name': character})
    for name in plot.get('key_character_names') or []:
        if name and not any(character.get('name') == name for character in characters):
            characters.append({'name': name})
    return characters


def find_pre_action_description(entity, first_action, plots_by_id):
    """Use only the detector's pre-plot description, never plot experience."""
    plot = plots_by_id.get(first_action['plot_id'], {})
    names = set(entity['source_names'])
    names.add(first_action['source_character'])
    names.add(entity['canonical_name'])
    for character in plot_key_characters(plot):
        if character.get('name') in names and character.get('description'):
            return character['description'], character.get('name')
    return ('No pre-action character description was extracted for this character. '
            'Remain cautious and state only what the current designation supports.'), None


def action_payload(action, is_current_trigger):
    return {
        'action_uid': action['action_uid'],
        'timeline_order': action['timeline_order'],
        'time_text': action.get('time_text'),
        'source_character': action['source_character'],
        'thought': action.get('thought'),
        'thought_evidence': action.get('thought_evidence'),
        'action': action.get('action'),
        'observation': action.get('observation'),
        'is_current_trigger': is_current_trigger,
    }


def call_profile_json(llm, prompt, model, max_tokens=1024, validator=None):
    """Use shared GPT-compatible JSON calling, rate limiting, and retries.

    Profile outputs are additionally validated by the caller. Do not reuse a
    cached semantic failure forever; completed entity files are the durable
    resume mechanism for this stage.
    """
    try:
        text = llm.call(prompt, model=model, max_tokens=max_tokens,
                        json_mode=True, use_cache=True)
        result = json.loads(text)
        if validator:
            validator(result)
        return result
    except Exception:
        # A transport-cached response can still be structurally unsuitable for
        # this profile call. Retry once fresh; successful earlier calls remain
        # cached, so an entity that fails later does not repay its whole prefix.
        text = llm.call(prompt, model=model, max_tokens=max_tokens,
                        json_mode=True, use_cache=False)
        result = json.loads(text)
        if validator:
            validator(result)
        return result


def validate_profile_text(profile):
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError('profile text missing')
    if len(profile) > 1300:
        raise ValueError(f'profile exceeds compact-size limit: {len(profile)} chars')


def validate_update(result):
    changed = result.get('changed')
    profile = result.get('profile')
    summary = result.get('change_summary')
    if not isinstance(changed, bool):
        raise ValueError('update response missing boolean changed')
    if changed:
        validate_profile_text(profile)
    if not changed and profile is not None:
        raise ValueError('changed=false response must have profile=null')
    if not isinstance(summary, str) or len(summary) > 400:
        raise ValueError('update response missing or oversized change_summary')


def validate_initial(result):
    validate_profile_text(result.get('profile'))


def generate_entity_profile(llm, model, title, entity, character_path, plots_by_id):
    data = load_json(character_path)
    actions = data['actions']
    if not actions:
        raise ValueError(f"{entity['entity_id']}: no actions")
    first = actions[0]
    pre_description, description_source_name = find_pre_action_description(
        entity, first, plots_by_id)
    init_prompt = PROFILE_INITIALIZATION_PROMPT.format(
        title=title,
        entity_id=entity['entity_id'],
        source_name=first['source_character'],
        timeline_order=first['timeline_order'],
        time_text=first.get('time_text') or 'unknown',
        pre_action_description=pre_description,
    )
    init = call_profile_json(llm, init_prompt, model=model, max_tokens=4096,
                             validator=validate_initial)
    profile = init.get('profile')
    try:
        validate_profile_text(profile)
    except ValueError as error:
        raise ValueError(f"{entity['entity_id']}: invalid initial profile: {error}") from error

    version = 0
    profile_versions = [{
        'profile_version': version,
        'before_action_uid': first['action_uid'],
        'timeline_order': first['timeline_order'],
        'source_character': first['source_character'],
        'pre_action_description': pre_description,
        'pre_action_description_source_name': description_source_name,
        'profile': profile,
        'evidence_assessment': init.get('evidence_assessment', ''),
    }]
    update_checks = []
    pending = []
    max_window_actions = max_window_chars = 0

    for action in actions:
        current_is_trigger = action['effective_critical'] is True
        pending.append(action_payload(action, current_is_trigger))
        if not current_is_trigger:
            continue
        max_window_actions = max(max_window_actions, len(pending))
        window_json = json.dumps(pending, ensure_ascii=False, indent=2)
        max_window_chars = max(max_window_chars, len(window_json))
        prompt = PROFILE_UPDATE_PROMPT.format(
            title=title,
            entity_id=entity['entity_id'],
            source_name=action['source_character'],
            timeline_order=action['timeline_order'],
            time_text=action.get('time_text') or 'unknown',
            profile_version=version,
            previous_profile=profile,
            action_window=window_json,
        )
        result = call_profile_json(llm, prompt, model=model, max_tokens=8192,
                                   validator=validate_update)
        validate_update(result)
        check = {
            'after_action_uid': action['action_uid'],
            'timeline_order': action['timeline_order'],
            'source_character': action['source_character'],
            'window_start_action_uid': pending[0]['action_uid'],
            'window_end_action_uid': action['action_uid'],
            'window_action_count': len(pending),
            'window_char_count': len(window_json),
            'changed': result['changed'],
            'profile_version_before': version,
            'change_summary': result['change_summary'],
        }
        if result['changed']:
            version += 1
            profile = result['profile']
            check['profile_version_after'] = version
            profile_versions.append({
                'profile_version': version,
                'after_action_uid': action['action_uid'],
                'timeline_order': action['timeline_order'],
                'source_character': action['source_character'],
                'window_start_action_uid': pending[0]['action_uid'],
                'window_action_count': len(pending),
                'profile': profile,
                'change_summary': result['change_summary'],
            })
            pending = []
        else:
            check['profile_version_after'] = version
        update_checks.append(check)

    critical_count = sum(action['effective_critical'] for action in actions)
    if len(update_checks) != critical_count:
        raise ValueError(f"{entity['entity_id']}: update checks do not cover critical actions")
    return {
        'book': title,
        'entity_id': entity['entity_id'],
        'canonical_name': entity['canonical_name'],
        'source_names': entity['source_names'],
        'action_count': len(actions),
        'effective_critical_count': critical_count,
        'profile_version_count': len(profile_versions),
        'changed_update_count': sum(check['changed'] for check in update_checks),
        'unchanged_update_count': sum(not check['changed'] for check in update_checks),
        'unabsorbed_action_count': len(pending),
        'max_window_action_count': max_window_actions,
        'max_window_char_count': max_window_chars,
        'profile_versions': profile_versions,
        'update_checks': update_checks,
    }


def main():
    ap = argparse.ArgumentParser(description='dynamic profile trial generator')
    ap.add_argument('--title', default=TITLE)
    ap.add_argument('--character-dir', default=DEFAULT_CHARACTER_DIR)
    ap.add_argument('--entities', default=DEFAULT_ENTITIES)
    ap.add_argument('--plots-file', default=DEFAULT_PLOTS)
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    ap.add_argument('--resume', action='store_true',
                    help='reuse completed entity JSON files in output-dir')
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--entity-ids', default=None,
                    help='comma-separated entities; overrides --top')
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()

    entities_data = load_json(args.entities)
    entities = entities_data['entities']
    zero_action_entities = [
        entity for entity in entities
        if int(entity.get('source_action_count') or 0) <= 0
    ]
    eligible_entities = [
        entity for entity in entities
        if int(entity.get('source_action_count') or 0) > 0
    ]
    if args.entity_ids:
        ids = set(args.entity_ids.split(','))
        known_ids = {entity['entity_id'] for entity in entities}
        if not ids <= known_ids:
            raise ValueError('unknown entity_id requested')
        selected = [entity for entity in eligible_entities
                    if entity['entity_id'] in ids]
    else:
        selected = sorted(
            eligible_entities,
            key=lambda entity: -int(entity.get('source_action_count') or 0),
        )[:args.top]
    plots_by_id = {plot['plot_id']: plot for plot in load_json(args.plots_file)['plots']}

    if os.path.exists(args.output_dir) and not args.resume:
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    failed_entities = []
    pending_entities = []
    for entity in selected:
        existing_path = os.path.join(args.output_dir, f"{entity['entity_id']}.json")
        if args.resume and os.path.exists(existing_path):
            try:
                existing = load_json(existing_path)
                if not existing.get('profile_versions'):
                    raise ValueError('profile_versions missing or empty')
                expected_actions = int(entity.get('source_action_count') or 0)
                if int(existing.get('action_count') or -1) != expected_actions:
                    raise ValueError(
                        f'action_count changed: profile={existing.get("action_count")} '
                        f'upstream={expected_actions}')
                results.append(existing)
                print(f"{entity['canonical_name']}: resumed")
                continue
            except Exception as error:
                print(f"{entity['canonical_name']}: invalid resume file; "
                      f"regenerating ({error})")
        pending_entities.append(entity)

    llm = LLM(config_path='config.json', book=args.title)
    with ThreadPoolExecutor(max_workers=max(
            1, min(args.workers, len(pending_entities) or 1))) as executor:
        futures = {
            executor.submit(
                generate_entity_profile, llm, args.model, args.title, entity,
                os.path.join(args.character_dir, f"{entity['entity_id']}.json"),
                plots_by_id,
            ): entity for entity in pending_entities
        }
        for future in as_completed(futures):
            entity = futures[future]
            try:
                result = future.result()
                results.append(result)
                path = os.path.join(args.output_dir, f"{entity['entity_id']}.json")
                atomic_json_dump(path, result)
                print(f"{entity['canonical_name']}: "
                      f"versions={result['profile_version_count']}, "
                      f"changed={result['changed_update_count']}, "
                      f"unchanged={result['unchanged_update_count']}, "
                      f"max_window={result['max_window_action_count']} actions")
            except Exception as error:
                failed_entities.append({
                    'entity_id': entity['entity_id'],
                    'canonical_name': entity.get('canonical_name'),
                    'error': str(error)[:1000],
                })
                print(f"{entity['canonical_name']}: FAILED: {error}", flush=True)

    results.sort(key=lambda item: item['canonical_name'])
    index = {
        'book': args.title,
        'model': args.model,
        'eligible_entity_count': len(selected),
        'entity_count': len(results),
        'failed_entity_count': len(failed_entities),
        'failed_entities': failed_entities,
        'skipped_entities': [{
            'entity_id': entity['entity_id'],
            'canonical_name': entity.get('canonical_name'),
            'reason': 'zero_actions',
        } for entity in zero_action_entities],
        'entities': [{
            'entity_id': item['entity_id'],
            'canonical_name': item['canonical_name'],
            'action_count': item['action_count'],
            'effective_critical_count': item['effective_critical_count'],
            'profile_version_count': item['profile_version_count'],
            'changed_update_count': item['changed_update_count'],
            'unchanged_update_count': item['unchanged_update_count'],
            'unabsorbed_action_count': item['unabsorbed_action_count'],
            'max_window_action_count': item['max_window_action_count'],
            'max_window_char_count': item['max_window_char_count'],
            'file': f"{item['entity_id']}.json",
        } for item in results],
    }
    atomic_json_dump(os.path.join(args.output_dir, 'index.json'), index)
    if failed_entities:
        print(f"partial profile result: {len(failed_entities)} entities failed; "
              "rerun with --resume to retry only those entities", flush=True)
    print(f'output: {args.output_dir}')


if __name__ == '__main__':
    main()
