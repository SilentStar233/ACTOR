# Rewash counterfactual actions: rewrite the 3 counterfactuals of each action so
# that, WITHIN one record, they mirror the real action on four dimensions —
# length (position-randomized), sentence count, format, and quote texture —
# while staying OUT OF CHARACTER. The real action is never modified.
#
# Usage:
#   cd repository-root
#   python3 -m actor.action_refinement.actions --book "Anna Karenina" \
#       --entity konstantin_dmitrievitch_levin --limit 20
#   python3 -m actor.action_refinement.actions --book "Anna Karenina"   # whole book
#   python3 -m actor.action_refinement.actions --all                    # all available books
from actor.shared.paths import ROOT_DIR

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = ROOT_DIR

from actor.shared.llm import LLM
from actor.shared.utils import (atomic_json_dump, list_book_titles, resolve_book_dirs,
                   tree_signature)

PROMPT_VERSION = 'rewash_v3'
# Books root for the flat distribution.
BOOKS_ROOT = os.path.join(_HERE, 'data')

# Modern colloquialisms that betray an LLM-written quote.
# Regex form with word boundaries (avoids matching "ok" inside other words).
MODERN_BLACKLIST_RE = re.compile(
    r'\b(ok|okay|yeah|gonna|wanna|gotta|guys|buddy)\b', re.IGNORECASE)
# Spoken-register markers signalling a loose, colloquial quote texture.
SPOKEN_MARKERS = ["'tis", "'t is", "there, there", 'why,', "as 'tis",
                  'indeed', 'mark my word', 'i dare say', 'i daresay']


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

def word_count(text):
    return len(text.split())


def extract_quote(text):
    """Return the quoted span of a "<manner> <verb>: \"<quote>\"" action, else None."""
    m = re.search(r'"([^"]+)"', text)
    return m.group(1) if m else None


def count_sentences(text):
    return len([s for s in re.split(r'[.!?]+', text) if s.strip()])


def quote_texture(text):
    """Loose colloquial texture: asides, filler, dashes, self-interruption."""
    low = text.lower()
    return (any(k in low for k in SPOKEN_MARKERS)
            or '—' in text or '–' in text or '--' in text)




# ---------------------------------------------------------------------------
# Target-length sampling (chosen position randomized)
# ---------------------------------------------------------------------------

def sample_targets(L, rng):
    """Sample 3 counterfactual target lengths around L.

    The chosen's length rank among the 4 options is balanced across {1,2,3,4}
    (deterministically, via the per-action rng): we pick how many
    counterfactuals fall below L (n_below ∈ {0,1,2,3}), then sample that many
    targets below L and the rest above L. This keeps any single length
    position from being systematically favoured, so DPO cannot learn a length
    prior. Targets need not be distinct — the randomization already removes
    any systematic signal.
    """
    lo = max(4, round(0.7 * L))
    hi = max(L + 1, round(1.3 * L))
    n_below = rng.randint(0, 3)

    below = list(range(lo, L))            # strictly < L
    above = list(range(L + 1, hi + 1))    # strictly > L

    def pick(pool):
        return rng.choice(pool) if pool else None

    targets = []
    for _ in range(n_below):
        v = pick(below)
        if v is not None:
            targets.append(v)
    while len(targets) < 3:
        v = pick(above) or pick(below)
        if v is None:                      # both pools empty (tiny L)
            v = rng.randint(lo, hi)
        targets.append(v)
    return sorted(targets)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(book, character, profile, action, orig_cfs, targets):
    action_text = action['action']
    L = word_count(action_text)
    quote = extract_quote(action_text)
    hq = quote is not None
    lines = [
        f'You are given a character\'s real action from the novel "{book}", and three',
        'original counterfactual alternatives. REWRITE the three counterfactuals.',
        '',
        '==Length==',
        f'The real action is {L} words. Write the three counterfactuals at these',
        'target lengths (each within ±10%):',
        f'  - counterfactual 1: ~{targets[0]} words',
        f'  - counterfactual 2: ~{targets[1]} words',
        f'  - counterfactual 3: ~{targets[2]} words',
        '',
        '==Format & texture (mirror THIS real action)==',
        'The real action chains verbs/participles with NO "he/she" subject anywhere',
        'outside the quote (e.g. "Convinces himself..., begins to..."). Each',
        'counterfactual MUST follow the same style: open with a verb/participle/',
        'manner word, and NEVER use "he", "she", or the character\'s name as a',
        'subject outside the quoted line.',
    ]
    if hq:
        n_sent = count_sentences(quote)
        lines += [
            'The real action has the form "<manner/tone> <speech verb>: \\"<quote>\\"".',
            f'Its quoted line has {n_sent} sentence(s) and about {word_count(quote)} words.',
            'Each counterfactual MUST:',
            '- use the same shell: a manner/tone modifier + a speech verb, third person;',
            f'- contain a quoted line of about {n_sent} sentence(s), comparable to the real one;',
        ]
        if quote_texture(quote):
            lines += [
                '- match the register: read the real quote and mirror how it actually',
                '  sounds — it is loose and spoken, with asides, filler or',
                '  self-interruption; yours should have a similar loose texture,',
                '  not one polished well-formed sentence;',
            ]
        else:
            lines += [
                '- match the register: read the real quote and mirror how it actually',
                '  sounds — its degree of polish, rhythm and formality; keep yours',
                '  comparable, neither noticeably more casual nor more polished;',
            ]
    else:
        lines += [
            'The real action is a non-speech behavior with NO quoted line (a physical',
            'act, a meaningful silence, or an internal decision as outwardly framed).',
            'Each counterfactual MUST likewise be a functional description of a',
            'non-speech behavior of the SAME KIND, with NO quote, of comparable',
            'length and third-person voice.',
        ]
    lines += [
        '',
        '==Content (out of character)==',
        f'Each counterfactual must be something {character} would NOT do or say —',
        'violating their personality, values, or style — yet remain situationally',
        'plausible for someone in this scene.',
        '',
        'The three counterfactuals must deviate along meaningfully different',
        'dimensions whenever the scene permits — for example emotional tone,',
        'behavioral goal, or behavioral method. Do not produce three variations',
        'that all occupy the same emotional pole or merely paraphrase one another.',
        '',
        'The original counterfactuals are reference material only; you may freely',
        'adjust their deviation direction to ensure diversity. Expand with plausible',
        'detail to reach the target lengths.',
        '',
        '==Grounding constraints==',
        'Despite being "out of character", each counterfactual MUST stay grounded in',
        'the character\'s current psychological state and the scene\'s established',
        'facts:',
        '- Do NOT jump to a completely unrelated context (e.g. if the character is',
        '  obsessively tearing wallpaper, do NOT have them suddenly prepare to receive',
        '  visitors or tidy the room for guests — that breaks psychological continuity).',
        '- The counterfactual must be an alternative response to the SAME situation',
        '  the character faces at this exact moment, not a response to a different one.',
        '- Respect temporal/causal facts already established in the scene description',
        '  (e.g. if a door has already been opened, do NOT have the character say "you',
        '  may enter" as if it is still closed).',
        '- The deviation is in personality/goal/method, NOT in the reality of what',
        '  has already happened in this scene.',
        '',
        '==Character profile==',
        profile,
        '',
        '==Scene==',
        action.get('scene_description') or '(none)',
        '',
        "==Character's thought==",
        action.get('thought') or '(none)',
        '',
        f'==Real action ({L} words)==',
        action_text,
        '',
        '==Original counterfactuals (deviation direction)==',
        f'1. {orig_cfs[0]}',
        f'2. {orig_cfs[1]}',
        f'3. {orig_cfs[2]}',
        '',
        '===Output Format===',
        'Strict JSON (no markdown fences):',
        '{"counterfactuals": ["...", "...", "..."]}',
        'The "counterfactuals" array MUST contain EXACTLY THREE separate strings.',
        'Each string is ONE complete counterfactual, fully self-contained. Do NOT',
        'merge the three into a single long string, and do NOT split one into many.',
        '',
        'Requirements:',
        f'- Lengths ~{targets[0]}/{targets[1]}/{targets[2]} words respectively (±10% each).',
        '- Mirror the real action\'s format (quote iff the real action quotes).',
        '- Each clearly out of character, each plausible, all three distinct.',
        '- The three should deviate along meaningfully different dimensions',
        '  (e.g. emotional tone / behavioral goal / behavioral method);',
        '  do NOT let all three land on the same emotional pole.',
        '- Do NOT copy the real action\'s quote.',
        f'- Match the diction and period register of "{book}" and of the real',
        '  action itself; absolutely NO modern colloquialisms',
        '  ("ok", "okay", "yeah", "gonna", "guys", "cool", "buddy", etc.).',
        '- Same language as the source (English).',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(cfs, action, targets):
    """Return (ok, reasons). Checks length, format, texture, distinctness."""
    reasons = []
    if not isinstance(cfs, list) or len(cfs) != 3:
        return False, ['need exactly 3 counterfactuals']
    action_text = action['action']
    L = word_count(action_text)
    real_q = extract_quote(action_text)
    hq = real_q is not None
    lens = []
    for i, cf in enumerate(cfs):
        if not isinstance(cf, str) or not cf.strip():
            reasons.append(f'cf{i + 1} empty')
            continue
        w = word_count(cf)
        lens.append(w)
        t = targets[i]
        # Length tolerance:
        # - very short targets: ±15% is sub-word, use an absolute ±2 words
        #   (a 4-word target accepts 2-6);
        # - very long real actions (>200 words): the model cannot reliably
        #   sustain 300+ word counterfactuals, so relax to ±25%.
        frac = 0.25 if L > 200 else 0.15
        tol = max(frac * t, 2.0)
        if abs(w - t) > tol:
            reasons.append(f'cf{i + 1} is {w} words, target ~{t}')
        # The real action never uses a "he/she" subject outside the quote;
        # counterfactuals must match that verb-chained style.
        nonquote = re.sub(r'"[^"]*"', '', cf)
        if re.search(r'\b(he|she)\b', nonquote, re.IGNORECASE):
            reasons.append(f'cf{i + 1} uses a "he/she" subject outside the quote')
        canon = (action.get('canonical_name') or '').split()
        stripped = cf.lstrip('",\'(').strip()
        if canon and any(stripped.startswith(n) for n in canon):
            reasons.append(f'cf{i + 1} opens with the character name')
        # also forbid "he/she" right after a leading participle phrase
        if re.match(r'^[A-Za-z-]+ing\b[^,]*,\s+(he|she)\b', cf, re.IGNORECASE):
            reasons.append(f'cf{i + 1} inserts a pronoun subject after a participle')
        cfq = extract_quote(cf)
        if hq and cfq is None:
            reasons.append(f'cf{i + 1} lacks a quote')
        if not hq and cfq is not None:
            reasons.append(f'cf{i + 1} should have no quote (pure action)')
        # texture: real quote colloquial -> cf quote must not be a bare polished
        # clause. Skip this check for very short quotes (<8 words): too little
        # room to show texture, so we don't penalize the model unfairly.
        if hq and quote_texture(real_q) and cfq is not None:
            if word_count(real_q) >= 8:
                if count_sentences(cfq) == 1 and ',' not in cfq and not quote_texture(cfq):
                    reasons.append(f'cf{i + 1} quote too polished for the colloquial real quote')
        low = cf.lower()
        m = MODERN_BLACKLIST_RE.search(cf)
        if m:
            reasons.append(f'cf{i + 1} contains modern term "{m.group(0)}"')
        if 'counterfactual' in low or 'rewritten' in low or '```' in cf:
            reasons.append(f'cf{i + 1} contains meta text')
    norm = lambda s: re.sub(r'\s+', ' ', s.lower().strip())
    if len({norm(c) for c in cfs}) != 3:
        reasons.append('counterfactuals not distinct')
    if any(norm(c) == norm(action['action']) for c in cfs):
        reasons.append('a counterfactual duplicates the real action')
    return (not reasons), reasons


def length_rank(action_len, cf_lens):
    """chosen's rank among the 4 option lengths (1=shortest, 4=longest)."""
    return 1 + sum(1 for c in cf_lens if c < action_len)


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

def load_latest_profile(profile_dir, entity_id, entity_reason):
    path = os.path.join(profile_dir, f'{entity_id}.json')
    if os.path.exists(path):
        try:
            data = load_json(path)
            versions = data.get('profile_versions') or []
            if versions:
                last = versions[-1]
                for key in ('text', 'profile', 'pre_action_description'):
                    if last.get(key):
                        return last[key]
        except Exception:
            pass
    return entity_reason or f'A character named in the novel.'


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-action processing
# ---------------------------------------------------------------------------

SINGLE_CF_THRESHOLD = 150  # real actions longer than this are done per-counterfactual


def build_single_cf_prompt(book, character, profile, action, orig_cf, target, index):
    """One counterfactual per call: avoids the model merging 3 long ones."""
    base = build_prompt(book, character, profile, action, [orig_cf, orig_cf, orig_cf],
                        [target, target, target])
    base += (
        f'\n\n==IMPORTANT: single output==\nThis time write ONLY counterfactual #{index} '
        f'(~{target} words), guided by original counterfactual #{index} above. '
        'Output JSON with a single-element array: {"counterfactuals": ["..."]}.')
    return base


def process_action(llm, book, character, profile, action, model, max_retries=3):
    orig_cfs = action.get('counterfactual_action') or []
    if len(orig_cfs) != 3:
        return None, 'need 3 original counterfactuals'
    rng = random.Random(action['action_uid'])  # deterministic per action
    L = word_count(action['action'])
    targets = sample_targets(L, rng)
    base_prompt = build_prompt(book, character, profile, action, orig_cfs, targets)
    # Long actions need big completions: 3 counterfactuals each up to 1.3L words.
    max_tokens = min(32768, max(8192, int(3 * 1.3 * L * 1.5) + 1024))

    # Very long real actions: the model merges 3 long counterfactuals into one
    # blob, so generate each counterfactual in its own call.
    if L > SINGLE_CF_THRESHOLD:
        reasons = []
        for attempt in range(max_retries + 1):
            cfs = []
            failed_call = None
            for i in range(3):
                p = build_single_cf_prompt(book, character, profile, action,
                                           orig_cfs[i], targets[i], i + 1)
                if attempt > 0:
                    # Bypass cache on retries: the cached response may be the
                    # very one that failed validation.
                    p += f'\n\n(Retry {attempt}: previous output failed checks; write a fresh, compliant counterfactual.)'
                try:
                    data = llm.call_json(p, model=model, max_tokens=max_tokens,
                                         use_cache=(attempt == 0))
                    one = data.get('counterfactuals') or []
                    cfs.append(one[0] if one else '')
                except Exception as e:
                    # A bad long-form call must not kill the whole book run.
                    failed_call = f'cf{i + 1} llm error: {str(e)[:80]}'
                    break
            if failed_call:
                reasons = [failed_call]
                continue
            ok, reasons = validate(cfs, action, targets)
            if ok:
                cf_lens = [word_count(c) for c in cfs]
                meta = {'model': model, 'prompt_version': PROMPT_VERSION,
                        'targets': targets, 'actual_lens': cf_lens,
                        'length_rank': length_rank(L, cf_lens),
                        'attempts': attempt + 1, 'mode': 'per_counterfactual'}
                return {'counterfactual_action_rewashed': cfs, 'rewash_meta': meta}, None
        return None, '; '.join(reasons)

    prompt = base_prompt
    last_reasons = []
    for attempt in range(max_retries + 1):
        try:
            data = llm.call_json(prompt, model=model, max_tokens=max_tokens)
        except Exception as e:
            last_reasons = [f'llm error: {str(e)[:80]}']
            break
        cfs = data.get('counterfactuals')
        ok, reasons = validate(cfs, action, targets)
        if ok:
            cf_lens = [word_count(c) for c in cfs]
            meta = {
                'model': model,
                'prompt_version': PROMPT_VERSION,
                'targets': targets,
                'actual_lens': cf_lens,
                'length_rank': length_rank(word_count(action['action']), cf_lens),
                'attempts': attempt + 1,
            }
            return {'counterfactual_action_rewashed': cfs, 'rewash_meta': meta}, None
        last_reasons = reasons
        # Append concrete failure feedback and retry.
        prompt = (base_prompt
                  + '\n\n==Previous attempt failed==\n- ' + '\n- '.join(reasons)
                  + '\nFix these issues and output ONLY the corrected JSON.')
    return None, '; '.join(last_reasons)


# ---------------------------------------------------------------------------
# Per-entity file processing
# ---------------------------------------------------------------------------

def rewash_entity(llm, book, ct_dir, out_dir, profile_dir, entity_id,
                  model, limit, workers):
    src = os.path.join(ct_dir, f'{entity_id}.json')
    if not os.path.exists(src):
        return 0, 0, f'missing {src}'
    # Resume from the rewashed output if it exists (it carries rewash_meta);
    # otherwise start fresh from the source timelines.
    out_path = os.path.join(out_dir, f'{entity_id}.json')
    try:
        if os.path.exists(out_path):
            data = load_json(out_path)
        else:
            data = load_json(src)
    except Exception as e:
        return 0, 0, f'corrupt json for {entity_id}: {str(e)[:80]}'
    profile = load_latest_profile(profile_dir, entity_id, data.get('entity_reason'))
    character = data.get('canonical_name') or entity_id

    actions = data.get('actions', [])
    pending = [a for a in actions
               if a.get('rewash_meta', {}).get('prompt_version') != PROMPT_VERSION]

    # Source-data flaws (original counterfactuals != 3) can never be rewashed:
    # mark them skipped once, log them, and never retry on later runs.
    skipped = [a for a in pending
               if len(a.get('counterfactual_action') or []) != 3]
    if skipped:
        for a in skipped:
            a['rewash_meta'] = {'prompt_version': PROMPT_VERSION,
                                'skipped': 'source_cf_count'}
        print(f'    {entity_id}: skipped {len(skipped)} actions '
              f'(source counterfactual count != 3): '
              f'{[a["action_uid"] for a in skipped]}', flush=True)
    pending = [a for a in pending
               if len(a.get('counterfactual_action') or []) == 3]
    if limit:
        pending = pending[:limit]

    done = fail = 0
    failures = []

    def work(action):
        return process_action(llm, book, character, profile, action, model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, a): a for a in pending}
        for fut in as_completed(futs):
            a = futs[fut]
            try:
                result, err = fut.result()
            except Exception as e:
                # Last line of defense: an unexpected worker bug must fail this
                # one action, never the whole book.
                result, err = None, f'unexpected: {str(e)[:100]}'
            if result:
                a['counterfactual_action_rewashed'] = result['counterfactual_action_rewashed']
                a['rewash_meta'] = result['rewash_meta']
                done += 1
            else:
                a.setdefault('rewash_meta', {})['error'] = err
                failures.append((a['action_uid'], err))
                fail += 1
            if (done + fail) % 25 == 0:
                print(f'    {entity_id}: {done} done, {fail} failed', flush=True)
                # Periodic flush: crash-safe progress for long batches.
                atomic_json_dump(out_path, data)

    atomic_json_dump(out_path, data)
    if failures:
        for uid, err in failures[:5]:
            print(f'    FAIL {uid}: {err[:100]}', flush=True)
    return done, fail, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_books_file(path):
    """One book title per line ('#' comments and blank lines ignored)."""
    books = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                books.append(line)
    return books


def main():
    ap = argparse.ArgumentParser(description='rewash counterfactual actions')
    ap.add_argument('--book', default=None, help='single book title')
    ap.add_argument('--all', action='store_true',
                    help='all books under --books-root')
    ap.add_argument('--books-file', default=None,
                    help='text file with one book title per line (for batch/shards)')
    ap.add_argument('--books-root', default=BOOKS_ROOT,
                    help='root dir containing <book>/character_timelines/')
    ap.add_argument('--entity', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--config', default=os.path.join(_HERE, 'config.json'))
    args = ap.parse_args()

    if args.books_file:
        books = load_books_file(args.books_file)
    elif args.all:
        books = list_book_titles(args.books_root)
    elif args.book:
        books = [args.book]
    else:
        raise SystemExit('pass --book TITLE, --books-file FILE, or --all')

    total_done = total_fail = 0
    for book in books:
        # One LLM per book: cost.jsonl and the disk cache live under
        # <run_dir>/<book>/, so multi-book runs never mix per-book logs.
        llm = LLM(config_path=args.config, book=book)
        llm.default_model = args.model
        ct_dir, out_dir, profile_dir = resolve_book_dirs(args.books_root, book)
        if ct_dir is None:
            print(f'== {book}: SKIP (no character_timelines) ==', flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)

        if args.entity:
            entity_ids = [args.entity]
        else:
            entity_ids = [f[:-5] for f in sorted(os.listdir(ct_dir))
                          if f.endswith('.json') and f != 'index.json'
                          and 'unresolved' not in f]
        print(f'== {book}: {len(entity_ids)} entities ==', flush=True)
        book_done = book_fail = 0
        for eid in entity_ids:
            done, fail, err = rewash_entity(
                llm, book, ct_dir, out_dir, profile_dir, eid, args.model,
                args.limit, args.workers)
            book_done += done
            book_fail += fail
            print(f'  {eid}: rewashed={done} failed={fail}', flush=True)
        total_done += book_done
        total_fail += book_fail
        print(f'== {book} done: rewashed={book_done} failed={book_fail}, '
              f'cost={llm.cost_summary()}', flush=True)
        # Per-book summary for the batch scheduler: complete means EVERY action
        # of EVERY entity file carries a counterfactual_action_rewashed field.
        total_actions = complete_actions = 0
        for eid in entity_ids:
            src_path = os.path.join(ct_dir, f'{eid}.json')
            out_path = os.path.join(out_dir, f'{eid}.json')
            try:
                source_data = load_json(src_path)
                source_actions = source_data.get('actions', [])
            except Exception:
                # A corrupt/missing source entity must keep the book incomplete.
                total_actions += 1
                continue
            try:
                output_data = load_json(out_path)
                output_by_uid = {
                    action.get('action_uid'): action
                    for action in output_data.get('actions', [])
                }
            except Exception:
                total_actions += len(source_actions)
                continue
            for source_action in source_actions:
                total_actions += 1
                a = output_by_uid.get(source_action.get('action_uid'), {})
                if ('counterfactual_action_rewashed' in a
                        or a.get('rewash_meta', {}).get('skipped')):
                    complete_actions += 1
        atomic_json_dump(os.path.join(out_dir, '_summary.json'), {
            'book': book, 'model': args.model, 'prompt_version': PROMPT_VERSION,
            'source_signature': tree_signature(ct_dir),
            'entities': len(entity_ids),
            'total_actions': total_actions,
            'complete_actions': complete_actions,
            'incomplete': total_actions - complete_actions,
            'complete': total_actions > 0 and complete_actions == total_actions,
            'rewashed_this_run': book_done, 'failed_this_run': book_fail,
        })

    print(f'\nAll done. rewashed={total_done} failed={total_fail}')


if __name__ == '__main__':
    main()
