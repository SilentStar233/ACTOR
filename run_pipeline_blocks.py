# Full block-ID pipeline with stage-level resume, best-effort gates, and a
# per-book status file. Plot stage (detect -> align -> check -> repair ->
# enrich) followed by the unchanged v2 downstream stages.
#
# - Each stage is skipped when its artifacts already exist and validate.
# - --best-effort: LLM-judgment gates (timeline uncertainty, review
#   unresolved, scene-time failures, enrichment failures) are recorded to
#   needs_review.json and the pipeline continues; data-integrity gates
#   (alignment, overlap, extraction) always stop the book.
# - logs/<book>/status.json tracks the stage machine for the batch scheduler.
# Usage: cd pipeline_flat_github && python3 run_pipeline_blocks.py --title "Book"
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = _HERE  # Flat distribution: every pipeline script lives beside this one.
sys.path.insert(0, _HERE)
from utils import atomic_json_dump, safe_title

PY = sys.executable


def run(args, label, cwd):
    print(f'\n=== {label} ===')
    print('+', ' '.join(args))
    result = subprocess.run(args, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f'{label} failed ({result.returncode})')


def run_soft(args, label, cwd):
    """Run a degradable stage; return the exit code instead of raising."""
    print(f'\n=== {label} ===')
    print('+', ' '.join(args))
    return subprocess.run(args, cwd=cwd).returncode


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def valid_json(path, required_keys=()):
    if not os.path.exists(path):
        return None
    try:
        data = load_json(path)
    except Exception:
        return None
    if any(key not in data for key in required_keys):
        return None
    return data


def main():
    ap = argparse.ArgumentParser(
        description='block-ID full pipeline with stage resume and best-effort gates')
    ap.add_argument('--src', default='data/src/all_books.jsonl')
    ap.add_argument('--title', required=True)
    ap.add_argument('--chunk-size', type=int, default=30000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--extract-model', default='gpt-5.6-luna')
    ap.add_argument('--judge-model', default='gpt-5.6-luna')
    ap.add_argument('--skip-plot-stage', action='store_true',
                    help='require plots_final to exist and skip detect/align/repair')
    ap.add_argument('--plot-stage-only', action='store_true')
    ap.add_argument('--extract-plots', default=None,
                    help='comma-separated plot ids to extract (default: all)')
    ap.add_argument('--best-effort', action='store_true',
                    help='degrade LLM-judgment gates to needs_review.json and continue')
    ap.add_argument('--profile-top', type=int, default=0)
    ap.add_argument('--profile-workers', type=int, default=5)
    ap.add_argument('--profile-resume', action='store_true')
    args = ap.parse_args()
    if args.chunk_size < 1 or args.workers < 1 or args.profile_workers < 1:
        raise ValueError('chunk-size/workers must be positive')
    args.src = os.path.abspath(args.src)

    title = args.title
    safe = safe_title(title)
    log_dir = os.path.join(_HERE, 'data', 'logs', safe)
    os.makedirs(log_dir, exist_ok=True)

    # Per-book mutual exclusion: a second pipeline for the same book (double
    # batch, or manual run alongside a batch) exits immediately with code 17.
    lock_file = open(os.path.join(log_dir, '.lock'), 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f'another pipeline is already running for {title}; exit 17')
        sys.exit(17)
    # All LLM caches/cost logs of child processes land under this book's dir.
    os.environ['AT_RUN_DIR'] = os.path.join(_HERE, 'data', 'logs')
    os.environ['AT_BOOK_CACHE_DIR'] = os.path.join(
        _HERE, 'data', 'book_cache')

    sys.path.insert(0, _V2)
    from check_plot_overlaps import find_overlaps
    from extract import load_book

    book = load_book(args.src, title)
    content_sha = hashlib.sha256(book['content'].encode('utf-8')).hexdigest()

    status_path = os.path.join(log_dir, 'status.json')
    needs_review_path = os.path.join(log_dir, 'needs_review.json')

    scope = 'plot_stage' if args.plot_stage_only else 'full'

    def write_status(stage, state, error=None):
        payload = {'book': title, 'stage': stage, 'status': state,
                   'scope': scope,
                   'content_sha256': content_sha,
                   'best_effort': args.best_effort,
                   'pid': os.getpid(), 'hostname': os.uname().nodename,
                   'updated_at': time.time()}
        if error:
            payload['error'] = str(error)[:500]
        atomic_json_dump(status_path, payload)

    def add_needs_review(gate, detail):
        entries = []
        if os.path.exists(needs_review_path):
            try:
                entries = load_json(needs_review_path).get('items', [])
            except Exception:
                entries = []
        entries.append({'gate': gate, 'detail': str(detail)[:2000],
                        'time': time.time()})
        atomic_json_dump(needs_review_path, {'book': title, 'items': entries})
        print(f'  [needs_review] {gate}: {str(detail)[:200]}')

    # ---- paths relative to the flat repository root ----
    detected = f'data/plots_detected/{safe}.plots.json'
    blocks_file = f'data/plots_detected/{safe}.blocks.json'
    assembled = f'data/plots_assembled/{safe}.plots.json'
    final = f'data/plots_final/{safe}.plots.json'
    alignment_report = f'data/logs/{safe}/{safe}.block_alignment.json'
    initial_report = f'data/logs/{safe}/{safe}.block_overlaps.initial.json'
    final_report = f'data/logs/{safe}/{safe}.block_overlaps.final.json'
    repair_report = f'data/logs/{safe}/{safe}.block_repair.json'
    enrich_report = f'data/logs/{safe}/{safe}.character_detail_enrichment.json'
    plots_file_v2 = final
    contexts_v2 = 'data/contexts'
    traj_root_v2 = 'data/trajectories_auto'
    traj_dir_v2 = f'{traj_root_v2}/{safe}'
    audit_report_v2 = f'data/logs/{safe}/{safe}.audit_critical.json'
    time_report_v2 = f'data/logs/{safe}/{safe}.scene_time_relabel.json'
    timeline_default_v2 = f'data/timelines/{safe}.auto.default.json'
    timeline_proposals_v2 = f'data/timelines/{safe}.auto.anomalies.json'
    timeline_reviews_v2 = f'data/timelines/{safe}.auto.reviewed_moves.json'
    timeline_final_v2 = f'data/timelines/{safe}.auto.final.json'
    entities_v2 = f'data/entities/{safe}.entities.json'
    character_timelines_v2 = f'data/character_timelines/{safe}'
    alias_map_v2 = f'{contexts_v2}/{safe}/alias_map.json'

    for sub in ('plots_detected', 'plots_assembled', 'plots_final'):
        os.makedirs(os.path.join(_HERE, 'data', sub), exist_ok=True)

    here = lambda p: os.path.join(_HERE, p)
    at_v2 = lambda p: os.path.join(_V2, p)

    # ---- stage-done checks -------------------------------------------------
    def detect_done():
        data = valid_json(here(detected), ('plots',))
        return bool(data and data['plots']) and os.path.exists(here(blocks_file))

    def align_done():
        data = valid_json(here(assembled), ('plots',))
        if not data:
            return False
        return all(p.get('source_alignment', {}).get('status') == 'resolved'
                   for p in data['plots'])

    def final_done():
        data = valid_json(here(final), ('plots',))
        if not data or not data['plots']:
            return False
        spans = [{'plot_id': p['plot_id'], 'start': p['source_start'],
                  'end': p['source_end'],
                  'length': p['source_end'] - p['source_start']}
                 for p in data['plots']]
        return not find_overlaps(spans)

    def enrich_done():
        report = valid_json(here(enrich_report))
        if report is None:
            return False
        return not report.get('failed_plots') or args.best_effort

    def context_done():
        return (valid_json(at_v2(f'{contexts_v2}/{safe}/character_context.json'))
                is not None
                and valid_json(at_v2(alias_map_v2)) is not None)

    def extract_done():
        data = valid_json(here(final), ('plots',))
        if not data:
            return False
        ids = [p['plot_id'] for p in data['plots']]
        if args.extract_plots:
            wanted = {int(x) for x in args.extract_plots.split(',')}
            ids = [i for i in ids if i in wanted]
        for pid in ids:
            traj = valid_json(os.path.join(
                at_v2(traj_dir_v2), f'plot_{pid}.json'), ('plots',))
            if not traj:
                return False
            steps = sum(len(t.get('steps', []))
                        for pl in traj['plots'] for s in pl.get('scenes', [])
                        for t in s.get('trajectories', []))
            if steps == 0:
                return False
        return True

    def audit_done():
        report = valid_json(at_v2(audit_report_v2))
        return bool(report and report.get('total_audited') is not None)

    def time_relabel_done():
        return valid_json(at_v2(time_report_v2)) is not None

    def timeline_final_done():
        data = valid_json(at_v2(timeline_final_v2), ('scenes',))
        return bool(data and data['scenes'])

    def entities_done():
        data = valid_json(at_v2(entities_v2), ('entities',))
        return bool(data and data['entities'])

    def character_timelines_done():
        index = valid_json(at_v2(
            f'{character_timelines_v2}/index.json'))
        return bool(index)

    try:
        # ================= stage 1: plots =================
        if not detect_done():
            write_status('detect', 'running')
            run([PY, 'detect_plots_blocks.py', '--src', args.src, '--title', title,
                 '--chunk-size', str(args.chunk_size), '--model', args.extract_model,
                 '--output-dir', 'data/plots_detected'],
                'Block-ID plot detection', cwd=_HERE)
        else:
            print('skip detect (done)')
        write_status('detect', 'stage_ok')

        if args.skip_plot_stage and not os.path.exists(here(final)):
            raise SystemExit(f'--skip-plot-stage but {final} does not exist')

        if not args.skip_plot_stage:
            if not align_done():
                write_status('align', 'running')
                run([PY, 'align_blocks.py', '--src', args.src, '--title', title,
                     '--input', detected, '--output', assembled,
                     '--report', alignment_report],
                    'Block table alignment', cwd=_HERE)
            else:
                print('skip align (done)')
            alignment = load_json(here(alignment_report))
            if alignment.get('coverage_warning'):
                print(f'\n*** COVERAGE WARNING: {title} plot coverage '
                      f"{alignment['coverage']:.1%} — 人工确认前言/正文是否被误删 ***")
                add_needs_review('coverage', alignment)
            write_status('align', 'stage_ok')

            if not final_done():
                run([PY, 'check_plot_overlaps.py', '--src', args.src, '--title', title,
                     '--plots-file', assembled, '--output', initial_report],
                    'Initial overlap check', cwd=_V2)
                initial = load_json(here(initial_report))
                if initial['unresolved_span_count']:
                    raise RuntimeError('alignment incomplete; see alignment report')
                write_status('repair', 'running')
                if initial['overlap_count']:
                    run([PY, 'repair_overlaps_blocks.py', '--src', args.src,
                         '--title', title, '--plots', assembled,
                         '--output', final, '--report', repair_report,
                         '--model', args.judge_model],
                        'Overlap repair', cwd=_HERE)
                else:
                    shutil.copyfile(here(assembled), here(final))
                run([PY, 'check_plot_overlaps.py', '--src', args.src, '--title', title,
                     '--plots-file', plots_file_v2, '--output',
                     final_report, '--fail-on-overlap'],
                    'Final overlap gate', cwd=_V2)
            else:
                print('skip repair/final (done)')
            write_status('repair', 'stage_ok')

        if not enrich_done():
            write_status('enrich', 'running')
            enrich_args = [PY, 'enrich_plot_characters.py', '--title', title,
                           '--plots-file', plots_file_v2, '--report',
                           enrich_report, '--model', args.extract_model,
                           '--workers', str(args.workers)]
            if args.best_effort:
                enrich_args.append('--allow-failures')
            run(enrich_args, 'Character detail enrichment', cwd=_V2)
            report = load_json(here(enrich_report))
            if report.get('failed_plots'):
                add_needs_review('enrich_failures', report['failed_plots'])
        else:
            print('skip enrich (done)')
        write_status('enrich', 'stage_ok')

        if args.plot_stage_only:
            # plot-stage-only is a complete unit of work: mark the book done
            # so the batch summary counts it correctly.
            write_status('done', 'ok_with_warnings'
                         if os.path.exists(needs_review_path) else 'ok')
            print(f'\nPlot stage complete: {final}')
            return

        # ================= stage 2: context =================
        if not context_done():
            write_status('context', 'running')
            run([PY, 'build_context.py', '--title', title,
                 '--plots_dir', 'data/plots_final',
                 '--output_dir', contexts_v2,
                 '--config', 'config.json', '--model', args.extract_model],
                'Alias map and character contexts', cwd=_V2)
        else:
            print('skip context (done)')
        write_status('context', 'stage_ok')

        # ================= stage 3: extraction =================
        if not extract_done():
            write_status('extract', 'running')
            extract_args = [PY, 'run_extract_v2.py', '--title', title,
                            '--plots-file', plots_file_v2,
                            '--contexts-dir', contexts_v2,
                            '--output_dir', traj_root_v2, '--src', args.src,
                            '--model', args.extract_model,
                            '--workers', str(args.workers)]
            if args.extract_plots:
                extract_args += ['--plots', args.extract_plots]
            run(extract_args, 'Scene/action/counterfactual extraction', cwd=_V2)
        else:
            print('skip extract (done)')
        write_status('extract', 'stage_ok')

        # ================= stage 4: audit =================
        if not audit_done():
            write_status('audit', 'running')
            run([PY, 'audit_critical.py', '--title', title,
                 '--traj-dir', traj_dir_v2, '--report', audit_report_v2,
                 '--plots-file', plots_file_v2, '--workers', str(args.workers),
                 '--model', args.judge_model],
                'Positive-only critical audit', cwd=_V2)
        else:
            print('skip audit (done)')
        write_status('audit', 'stage_ok')

        # ================= stage 5: scene-time relabel (degradable) ==========
        if not time_relabel_done():
            write_status('scene_time', 'running')
            code = run_soft([PY, 'repair_scene_times.py', '--all', '--apply',
                             '--title', title, '--plots-file', plots_file_v2,
                             '--traj-dir', traj_dir_v2, '--output', time_report_v2,
                             '--model', args.extract_model,
                             '--workers', str(args.workers)],
                            'Strict scene-time relabel', cwd=_V2)
            if code != 0:
                if not args.best_effort:
                    raise RuntimeError(f'scene-time relabel failed ({code})')
                add_needs_review('scene_time_relabel',
                                 f'stage exited {code}; original time labels kept')
                atomic_json_dump(at_v2(time_report_v2), {
                    'book': title, 'degraded': True,
                    'reason': 'stage failed; original labels kept'})
        else:
            print('skip scene-time relabel (done)')
        write_status('scene_time', 'stage_ok')

        # ================= stage 6: timeline =================
        if not timeline_final_done():
            write_status('timeline', 'running')
            run([PY, 'build_timeline.py', '--title', title,
                 '--traj-dir', traj_dir_v2, '--output', timeline_default_v2],
                'Default text-order timeline', cwd=_V2)

            proposals = valid_json(at_v2(timeline_proposals_v2))
            if proposals is None:
                run([PY, 'detect_timeline_anomalies.py',
                     '--input', timeline_default_v2,
                     '--output', timeline_proposals_v2, '--model', args.judge_model],
                    'Temporal anomaly proposals', cwd=_V2)
                proposals = load_json(at_v2(timeline_proposals_v2))

            review = valid_json(at_v2(timeline_reviews_v2))
            proposal_problem = (proposals.get('validation_errors')
                                or proposals.get('uncertain'))
            if proposal_problem:
                detail = {'validation_errors': proposals.get('validation_errors'),
                          'uncertain': proposals.get('uncertain')}
                if not args.best_effort:
                    raise RuntimeError(
                        f'Timeline proposals need review: {detail}')
                add_needs_review('timeline_proposals', detail)
                review = {'book': title, 'time_corrections': [],
                          'accepted_moves': [], 'rejected_moves': [],
                          'unresolved': [], 'validation_errors': [],
                          'degraded': 'proposals gate bypassed (best-effort)'}
                atomic_json_dump(at_v2(timeline_reviews_v2), review)
            elif review is None:
                run([PY, 'review_timeline_anomalies.py',
                     '--timeline', timeline_default_v2,
                     '--proposals', timeline_proposals_v2,
                     '--output', timeline_reviews_v2,
                     '--plots-file', plots_file_v2, '--model', args.judge_model],
                    'Source-grounded move review', cwd=_V2)
                review = load_json(at_v2(timeline_reviews_v2))

            review_problem = (review.get('unresolved')
                              or review.get('validation_errors'))
            if review_problem:
                detail = {'unresolved': review.get('unresolved'),
                          'validation_errors': review.get('validation_errors')}
                if not args.best_effort:
                    raise RuntimeError(
                        f'Timeline review has unresolved cases: {detail}')
                add_needs_review('timeline_review', detail)
                review = dict(review)
                review['accepted_moves'] = []
                review['time_corrections'] = []
                review['degraded'] = 'review gate bypassed (best-effort)'
                atomic_json_dump(at_v2(timeline_reviews_v2), review)

            run([PY, 'apply_timeline_moves.py', '--input', timeline_default_v2,
                 '--moves', timeline_reviews_v2, '--output', timeline_final_v2],
                'Apply validated timeline moves', cwd=_V2)
        else:
            print('skip timeline (done)')
        write_status('timeline', 'stage_ok')

        # ================= stage 7: entities & character timelines ===========
        if not entities_done():
            write_status('entities', 'running')
            run([PY, 'build_entities.py', '--title', title,
                 '--traj-dir', traj_dir_v2, '--timeline', timeline_final_v2,
                 '--alias-map', alias_map_v2, '--output', entities_v2,
                 '--model', args.judge_model],
                'Book-level entity normalization', cwd=_V2)
        else:
            print('skip entities (done)')
        write_status('entities', 'stage_ok')

        if not character_timelines_done():
            write_status('character_timelines', 'running')
            run([PY, 'build_character_timelines.py', '--title', title,
                 '--traj-dir', traj_dir_v2, '--timeline', timeline_final_v2,
                 '--entities', entities_v2,
                 '--output-dir', character_timelines_v2],
                'Entity action timelines', cwd=_V2)
        else:
            print('skip character timelines (done)')
        write_status('character_timelines', 'stage_ok')

        print(f'\nPipeline complete through character action timelines:\n'
              f'  timeline: {timeline_final_v2}\n'
              f'  entities: {entities_v2}\n'
              f'  character timelines: {character_timelines_v2}')

        if args.profile_top > 0:
            write_status('profiles', 'running')
            profile_dir = f'data/profile_trials/{safe}'
            profile_args = [PY, 'generate_profiles.py', '--title', title,
                            '--character-dir', character_timelines_v2,
                            '--entities', entities_v2,
                            '--plots-file', plots_file_v2,
                            '--output-dir', profile_dir,
                            '--top', str(args.profile_top),
                            '--model', args.judge_model,
                            '--workers', str(args.profile_workers)]
            if args.profile_resume:
                profile_args.append('--resume')
            run(profile_args, 'Dynamic character profiles', cwd=_V2)
            print(f'  profiles: {profile_dir}')
            write_status('profiles', 'stage_ok')

        final_state = ('ok_with_warnings'
                       if os.path.exists(needs_review_path) else 'ok')
        write_status('done', final_state)

    except BaseException as error:
        state = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        detail = (f'{type(error).__name__}: {error}\n'
                  + traceback.format_exc()[-1500:])
        try:
            write_status('failed', state, detail)
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
