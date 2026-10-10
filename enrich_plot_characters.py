# Generate compact character description/experience for finalized plots.
# Runs after plot boundaries and source spans are stable; one API request per plot.
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from llm import LLM
from utils import atomic_json_dump

ENRICHMENT_PROMPT = """Generate compact character details for one plot from the novel "{title}".

You are positioned immediately AFTER this plot's boundaries have been finalized. Use only the supplied plot metadata and plot text. Do not use outside knowledge of the novel.

For every supplied character name, return:
- "name": exactly the supplied canonical name;
- "description": at most 25 words, covering identity, social position, stable status, or relevant relationships that are supported by this plot and its summary;
- "experience": at most 40 words, describing what this character does, decides, intends, learns, or contributes inside this plot.

Do not invent characters, relationships, traits, or events. Do not describe events outside this plot. Preserve names exactly.

Output strict JSON only:
{{
  "plot_id": {plot_id},
  "key_characters": [
    {{"name": "...", "description": "...", "experience": "..."}}
  ]
}}

==Plot metadata==
plot_id: {plot_id}
chapter_title: {chapter_title}
summary: {summary}

==Character names==
{character_names}

==Plot text==
{plot_text}
"""


def character_names_for_plot(plot):
    names = [name for name in plot.get('key_character_names') or [] if name]
    for character in plot.get('key_characters') or []:
        if isinstance(character, dict) and character.get('name'):
            names.append(character['name'])
        elif isinstance(character, str) and character:
            names.append(character)
    return list(dict.fromkeys(names))


def is_enriched(plot, names):
    characters = plot.get('key_characters') or []
    if len(characters) != len(names):
        return False
    by_name = {}
    for character in characters:
        if not isinstance(character, dict):
            return False
        name = character.get('name')
        if not name or not character.get('description') or not character.get('experience'):
            return False
        by_name[name] = True
    return set(by_name) == set(names)


def word_count(text):
    return len((text or '').split())


def validate_result(result, plot, names):
    if result.get('plot_id') != plot['plot_id']:
        raise ValueError(f"plot {plot['plot_id']}: model returned plot_id "
                         f"{result.get('plot_id')}")
    characters = result.get('key_characters')
    if not isinstance(characters, list):
        raise ValueError(f"plot {plot['plot_id']}: key_characters is not a list")
    if len(characters) != len(names):
        raise ValueError(f"plot {plot['plot_id']}: expected {len(names)} characters, "
                         f"got {len(characters)}")
    actual_names = [character.get('name') for character in characters]
    if actual_names != names:
        raise ValueError(f"plot {plot['plot_id']}: character order/names mismatch: "
                         f"expected {names}, got {actual_names}")
    for character in characters:
        description = character.get('description')
        experience = character.get('experience')
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"plot {plot['plot_id']} character {character.get('name')}: "
                             'missing description')
        if not isinstance(experience, str) or not experience.strip():
            raise ValueError(f"plot {plot['plot_id']} character {character.get('name')}: "
                             'missing experience')
        if word_count(description) > 35:
            raise ValueError(f"plot {plot['plot_id']} character {character.get('name')}: "
                             'description exceeds 35 words')
        if word_count(experience) > 55:
            raise ValueError(f"plot {plot['plot_id']} character {character.get('name')}: "
                             'experience exceeds 55 words')
    return [
        {
            'name': character['name'],
            'description': character['description'].strip(),
            'experience': character['experience'].strip(),
        }
        for character in characters
    ]


def call_enrichment_json(llm, prompt, model, max_tokens):
    completion = llm.client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        response_format={'type': 'json_object'},
        max_completion_tokens=max_tokens,
    )
    choice = completion.choices[0]
    finish_reason = getattr(choice, 'finish_reason', None)
    usage = getattr(completion, 'usage', None)
    llm._log({
        'model': model,
        'cached': False,
        'input_chars': len(prompt),
        'input_tokens': getattr(usage, 'prompt_tokens', None),
        'output_tokens': getattr(usage, 'completion_tokens', None),
        'finish_reason': finish_reason,
        'profile_json_mode': True,
    })
    if finish_reason == 'length':
        raise ValueError('Character enrichment response truncated')
    return json.loads(choice.message.content)


def enrich_one_plot(llm, model, title, plot, max_tokens):
    names = character_names_for_plot(plot)
    prompt = ENRICHMENT_PROMPT.format(
        title=title,
        plot_id=plot['plot_id'],
        chapter_title=plot.get('chapter_title') or 'unknown',
        summary=plot.get('summary') or '(no summary)',
        character_names='\n'.join(f'- {name}' for name in names) or '(none)',
        plot_text=plot.get('text', ''),
    )
    result = call_enrichment_json(llm, prompt, model=model, max_tokens=max_tokens)
    return validate_result(result, plot, names), len(prompt)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description='generate final-plot character details')
    ap.add_argument('--title', required=True)
    ap.add_argument('--plots-file', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--config', default='config.json')
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--only', type=int, nargs='*', default=None)
    ap.add_argument('--force', action='store_true',
                    help='regenerate plots that already have complete key_characters')
    ap.add_argument('--dry-run', action='store_true',
                    help='call the model and write the report without modifying plots')
    ap.add_argument('--allow-failures', action='store_true',
                    help='report failed plots but exit 0 (batch best-effort mode)')
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be at least 1')
    if args.max_tokens < 1:
        raise ValueError('--max-tokens must be at least 1')

    data = load_json(args.plots_file)
    plots = data.get('plots', [])
    selected = []
    skipped = []
    for plot in plots:
        names = character_names_for_plot(plot)
        if args.only is not None and plot['plot_id'] not in args.only:
            continue
        if not names:
            skipped.append({'plot_id': plot['plot_id'], 'reason': 'no character names'})
            continue
        if not args.force and is_enriched(plot, names):
            skipped.append({'plot_id': plot['plot_id'], 'reason': 'already enriched'})
            continue
        selected.append((plot, names))
    if args.limit is not None:
        selected = selected[:args.limit]

    llm = LLM(config_path=args.config, book=args.title)
    llm.default_model = args.model
    enriched = {}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(enrich_one_plot, llm, args.model, args.title, plot,
                            args.max_tokens): plot
            for plot, _ in selected
        }
        for future in as_completed(futures):
            plot = futures[future]
            try:
                characters, input_chars = future.result()
            except Exception as error:
                failures.append({
                    'plot_id': plot['plot_id'],
                    'error': str(error),
                })
                print(f"plot {plot['plot_id']}: FAILED - {str(error)[:200]}")
                continue
            enriched[plot['plot_id']] = {
                'key_characters': characters,
                'input_chars': input_chars,
            }
            print(f"plot {plot['plot_id']}: {len(characters)} characters enriched")

    selected_ids = {plot['plot_id'] for plot, _ in selected}
    missing_ids = sorted(selected_ids - set(enriched))
    if missing_ids:
        failures.extend({'plot_id': plot_id, 'error': 'no result returned'}
                        for plot_id in missing_ids)

    # Second pass: one automatic retry for failed plots (transient truncation
    # / format flukes). Deterministic failures will simply fail again.
    if failures:
        retry_ids = {f['plot_id'] for f in failures}
        retry_plots = [plot for plot, _ in selected
                       if plot['plot_id'] in retry_ids]
        print(f'second pass: retrying {len(retry_plots)} failed plots once')
        retried_ok = set()
        with ThreadPoolExecutor(max_workers=min(2, args.workers)) as executor:
            futures = {
                executor.submit(enrich_one_plot, llm, args.model, args.title,
                                plot, args.max_tokens): plot
                for plot in retry_plots
            }
            for future in as_completed(futures):
                plot = futures[future]
                try:
                    characters, input_chars = future.result()
                except Exception as error:
                    for failure in failures:
                        if failure['plot_id'] == plot['plot_id']:
                            failure['error'] = (
                                failure['error'][:120]
                                + f' | retry: {str(error)[:120]}')
                    continue
                enriched[plot['plot_id']] = {
                    'key_characters': characters,
                    'input_chars': input_chars,
                }
                retried_ok.add(plot['plot_id'])
                print(f"plot {plot['plot_id']}: retry OK, "
                      f'{len(characters)} characters')
        failures = [f for f in failures if f['plot_id'] not in retried_ok]

    if not args.dry_run and enriched:
        # Back up the single source of truth before the first in-place
        # rewrite; a later failed run can then be rolled back manually.
        backup = args.plots_file + '.bak'
        if not os.path.exists(backup):
            import shutil
            shutil.copyfile(args.plots_file, backup)
        for plot in plots:
            if plot['plot_id'] in enriched:
                plot['key_characters'] = enriched[plot['plot_id']]['key_characters']
        data['character_detail_enrichment'] = {
            'model': args.model,
            'enriched_plot_count': len(enriched),
            'failed_plot_count': len(failures),
            'input_policy': 'full_plot_text',
            'report': args.report,
        }
        atomic_json_dump(args.plots_file, data)

    report = {
        'book': args.title,
        'model': args.model,
        'plots_file': args.plots_file,
        'total_plots': len(plots),
        'selected_plots': len(selected),
        'enriched_plots': len(enriched),
        'failed_plots': failures,
        'skipped_plots': skipped,
        'dry_run': args.dry_run,
        'input_policy': 'full_plot_text',
        'results': [
            {'plot_id': plot_id, **enriched[plot_id]}
            for plot_id in sorted(enriched)
        ],
    }
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"enriched={len(enriched)}/{len(selected)}, failed={len(failures)}, "
          f"skipped={len(skipped)}")
    print(f'report: {args.report}')
    print(llm.cost_summary())
    if failures and not args.allow_failures:
        raise SystemExit('Character detail enrichment failed; refusing to continue.')
    if failures:
        print(f'WARNING: {len(failures)} plots unenriched '
              '(--allow-failures); continuing')


if __name__ == '__main__':
    main()
