#!/usr/bin/env python3
"""Build eval questions and/or preference training samples from action trajectory data.

Reads character_timelines_rewashed, profiles, and trajectories_auto to produce:
  - eval JSONL: 4-choice questions (1 real action + 3 rewashed counterfactuals)
  - train JSONL: DPO pairs (chosen=real action, rejected=best counterfactual)

Three modes:
  all         — all actions → eval only
  split       — tail (20%) → eval, head (80%) → train
  train-only  — all actions → train only

Layout B (server):
  books_root/
    character_timelines/{title}/          (index.json + {entity_id}.json)
    character_timelines_rewashed/{title}/ (rewashed {entity_id}.json)
    profile_trials/{title}/               ({entity_id}.json)
    trajectories_auto/{title}/            (plot_{id}.json)

Usage:
  python -m actor.dataset_building.build --books-root data \
      --title "A Christmas Carol" --mode all \
      --eval-output-dir data/dataset/eval

  python -m actor.dataset_building.build --books-root data \
      --title "The Brothers Karamazov" --mode split --split-ratio 0.2 \
      --eval-output-dir data/dataset/eval --train-output-dir data/dataset/train
"""
from actor.shared.paths import ROOT_DIR

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

_HERE = ROOT_DIR

from actor.shared.utils import (atomic_json_dump, list_book_titles, resolve_book_dirs,
                   safe_title, tree_signature)

NO_DESC_MARKER = "No pre-action character description"
BUILD_META_VERSION = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_rewashed(book_dir_rewashed, entity_id):
    path = os.path.join(book_dir_rewashed, f"{entity_id}.json")
    if not os.path.exists(path):
        return None
    return load_json(path)


def load_profile(profile_dir, entity_id):
    path = os.path.join(profile_dir, f"{entity_id}.json")
    if not os.path.exists(path):
        return None
    return load_json(path)


def load_trajectory(traj_dir, plot_id):
    path = os.path.join(traj_dir, f"plot_{plot_id}.json")
    if not os.path.exists(path):
        return None
    return load_json(path)


# ---------------------------------------------------------------------------
# Trajectory / scene helpers
# ---------------------------------------------------------------------------

_traj_cache = {}


def get_scene_info(traj_dir, target_action):
    """Get scene metadata (participants, time_text) from trajectory."""
    plot_id = target_action.get("plot_id")
    if plot_id is None:
        return {}
    if plot_id not in _traj_cache:
        _traj_cache[plot_id] = load_trajectory(traj_dir, plot_id)
    traj = _traj_cache[plot_id]
    if not traj:
        return {}
    target_scene_id = target_action.get("scene_id")
    for plot in traj.get("plots", []):
        for scene in plot.get("scenes", []):
            if scene.get("scene_id") == target_scene_id:
                return {
                    "participants": scene.get("participants", []),
                    "time_text": scene.get("time_text"),
                }
    return {}


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

_profile_cache = {}
_name_to_entity = {}


def load_all_profiles(profile_dir):
    """Load all profile files and build name→entity_id map."""
    if not os.path.isdir(profile_dir):
        return
    for f in sorted(os.listdir(profile_dir)):
        if not f.endswith(".json") or f == "index.json":
            continue
        entity_id = f[:-5]
        path = os.path.join(profile_dir, f)
        try:
            data = load_json(path)
            _profile_cache[entity_id] = data
            canonical = data.get("canonical_name", "")
            if canonical:
                _name_to_entity[canonical] = entity_id
        except Exception:
            pass


def get_participant_descriptions(participants):
    """Get pre_action_description for each participant from profile v0."""
    descriptions = {}
    for name in participants:
        entity_id = _name_to_entity.get(name)
        if entity_id and entity_id in _profile_cache:
            data = _profile_cache[entity_id]
            versions = data.get("profile_versions", [])
            if versions:
                desc = versions[0].get("pre_action_description", "")
                if desc and NO_DESC_MARKER not in desc:
                    descriptions[name] = desc
    return descriptions


def find_profile_for_action(profile_data, target_action):
    """Find the most recent profile version applicable to the target action.

    Returns (version_index, profile_text, distance).
    """
    pv = profile_data.get("profile_versions", []) if profile_data else []
    if not pv:
        return None, None, None
    target_tl_order = target_action.get("timeline_order", 0)
    best_idx = 0
    best_order = pv[0].get("timeline_order", 0)
    for i, v in enumerate(pv):
        v_order = v.get("timeline_order", 0)
        if i == 0:
            if v_order <= target_tl_order and v_order >= best_order:
                best_idx = i
                best_order = v_order
        else:
            if v_order < target_tl_order and v_order > best_order:
                best_idx = i
                best_order = v_order
    profile_text = pv[best_idx].get("profile", "")
    distance = target_tl_order - best_order
    return best_idx, profile_text, distance


def get_profile_text(profile_data, action, entity_reason):
    """Get profile text for an action, with fallback to entity_reason."""
    idx, text, dist = find_profile_for_action(profile_data, action)
    if text:
        return idx, text, dist
    return None, entity_reason or "A character named in the novel.", 0


# ---------------------------------------------------------------------------
# Memory construction
# ---------------------------------------------------------------------------

def build_recent_memory(all_actions, target_action, max_scenes=3, max_total_chars=2000):
    """Short-term memory: most recent scenes (excluding target scene)."""
    target_order = target_action.get("character_action_order", 0)
    target_scene = target_action.get("scene_uid", "")
    prior = [a for a in all_actions
             if a.get("character_action_order", 0) < target_order]
    if not prior:
        return []
    scene_groups = defaultdict(list)
    for a in prior:
        sid = a.get("scene_uid", "")
        if sid == target_scene:
            continue
        scene_groups[sid].append(a)
    scene_order = []
    for sid, actions in scene_groups.items():
        max_order = max(a.get("character_action_order", 0) for a in actions)
        scene_order.append((max_order, sid))
    scene_order.sort(reverse=True)
    recent_scene_ids = set(sid for _, sid in scene_order[:max_scenes])
    memory_entries = []
    seen_uids = set()
    for sid in sorted(recent_scene_ids,
                      key=lambda s: min(a.get("character_action_order", 0)
                                        for a in scene_groups[s])):
        for a in sorted(scene_groups[sid],
                        key=lambda x: x.get("character_action_order", 0)):
            if a["action_uid"] not in seen_uids:
                memory_entries.append(a)
                seen_uids.add(a["action_uid"])
    memory_entries.sort(key=lambda x: x.get("character_action_order", 0))
    result = []
    total_chars = 0
    for a in memory_entries:
        entry = {
            "thought": (a.get("thought") or "")[:100],
            "action": (a.get("action") or "")[:100],
            "observation": (a.get("observation") or "")[:80] if a.get("observation") else None,
        }
        entry_chars = sum(len(v) for v in entry.values() if v)
        if total_chars + entry_chars > max_total_chars:
            break
        result.append(entry)
        total_chars += entry_chars
    return result


def build_long_term_memory(all_actions, target_action, max_total_chars=4000):
    """Long-term memory: all prior effective_critical actions."""
    target_order = target_action.get("character_action_order", 0)
    prior_critical = [
        a for a in all_actions
        if a.get("character_action_order", 0) < target_order
        and a.get("effective_critical", False)
    ]
    if not prior_critical:
        return []
    prior_critical.sort(key=lambda x: x.get("character_action_order", 0))
    result = []
    total_chars = 0
    for a in prior_critical:
        entry = {
            "action_uid": a.get("action_uid", ""),
            "thought": a.get("thought") or "",
            "action": a.get("action") or "",
            "observation": a.get("observation") or "",
            "character_action_order": a.get("character_action_order", 0),
            "scene_uid": a.get("scene_uid", ""),
        }
        entry_chars = sum(len(v) for v in entry.values() if isinstance(v, str))
        if total_chars + entry_chars > max_total_chars:
            break
        result.append(entry)
        total_chars += entry_chars
    return result


# ---------------------------------------------------------------------------
# Scene description (spoiler-free)
# ---------------------------------------------------------------------------

def build_scene_description(all_actions, target_action, scene_info=None):
    """Scene description that does NOT leak the target action."""
    target_order = target_action.get("character_action_order", 0)
    target_scene = target_action.get("scene_uid", "")
    parts = []
    if scene_info:
        if scene_info.get("time_text"):
            parts.append(f"Time: {scene_info['time_text']}")
        if scene_info.get("participants"):
            parts.append(f"Present: {', '.join(scene_info['participants'])}")
    prior_in_scene = []
    for a in all_actions:
        if a.get("scene_uid") != target_scene:
            continue
        if a.get("character_action_order", 0) >= target_order:
            continue
        if a.get("entity_id") == target_action.get("entity_id"):
            prior_in_scene.append(a)
    if prior_in_scene:
        parts.append("What has happened so far in this scene:")
        for a in sorted(prior_in_scene, key=lambda x: x.get("character_action_order", 0)):
            entry_parts = []
            if a.get("action"):
                entry_parts.append(f"Action: {a['action'][:150]}")
            if a.get("observation"):
                entry_parts.append(f"Result: {a['observation'][:120]}")
            parts.append("  - " + " | ".join(entry_parts))
    else:
        parts.append("(The scene has just begun.)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Profile usability / freshness
# ---------------------------------------------------------------------------

def is_profile_usable(profile_data, timeline_data):
    if not has_valid_profile(profile_data):
        return False
    pv = profile_data.get("profile_versions", []) if profile_data else []
    if len(pv) > 1:
        return True
    if timeline_data.get("effective_critical_count", 0) > 0:
        return True
    pre_desc = pv[0].get("pre_action_description", "") if pv else ""
    if NO_DESC_MARKER in pre_desc:
        return False
    return True


def has_valid_profile(profile_data):
    """A dataset record must carry a real generated character profile."""
    versions = profile_data.get("profile_versions", []) if profile_data else []
    return any(isinstance(v.get("profile"), str) and v["profile"].strip()
               for v in versions if isinstance(v, dict))


def profile_freshness_label(distance):
    if distance is None:
        return None
    if distance <= 15:
        return "fresh"
    elif distance <= 40:
        return "moderate"
    else:
        return "stale"


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def has_valid_rewashed_cfs(action, use_rewashed=True):
    """Check that action has 3 valid counterfactuals (rewashed or original)."""
    key = "counterfactual_action_rewashed" if use_rewashed else "counterfactual_action"
    cfs = action.get(key)
    if not cfs or len(cfs) != 3:
        return False
    return all(isinstance(cf, str) and cf.strip() for cf in cfs)


def is_rewash_skipped(action):
    meta = action.get("rewash_meta", {})
    return bool(meta.get("skipped") or meta.get("error"))


def word_count(text):
    return len(text.split())


def pick_best_rejected(cfs, action_text):
    """Pick the counterfactual whose length is closest to the real action."""
    L = word_count(action_text)
    return min(cfs, key=lambda cf: abs(word_count(cf) - L))


# ---------------------------------------------------------------------------
# Shared context builder
# ---------------------------------------------------------------------------

def build_context_fields(timeline_data, profile_data, action, scene_info,
                         book_dir_layout, use_rewashed=True):
    """Build the shared context fields used by both eval and train records."""
    actions = timeline_data.get("actions", [])
    entity_reason = timeline_data.get("entity_reason")

    recent_memory = build_recent_memory(actions, action)
    long_term_memory = build_long_term_memory(actions, action)
    scene_description = build_scene_description(actions, action, scene_info)
    participants = scene_info.get("participants", [])
    participant_descs = get_participant_descriptions(participants) if participants else {}

    version_idx, profile_text, distance = get_profile_text(
        profile_data, action, entity_reason)

    return {
        "profile": {
            "version": version_idx,
            "text": profile_text,
            "freshness": profile_freshness_label(distance),
            "distance": distance,
        },
        "recent_memory": recent_memory,
        "long_term_memory": long_term_memory,
        "scene": {
            "description": scene_description,
            "participant_descriptions": participant_descs,
            "time_text": scene_info.get("time_text") or action.get("time_text"),
        },
    }


# ---------------------------------------------------------------------------
# Eval question builder
# ---------------------------------------------------------------------------

def build_eval_question(book_title, timeline_data, profile_data, action,
                         scene_info, split_label, book_dir_layout, rng):
    """Construct one eval question dict."""
    entity_id = timeline_data["entity_id"]
    canonical_name = timeline_data["canonical_name"]
    action_uid = action["action_uid"]

    ctx = build_context_fields(
        timeline_data, profile_data, action, scene_info,
        book_dir_layout)

    # Options: 1 real + 3 counterfactuals (rewashed preferred)
    cfs_key = "counterfactual_action_rewashed"
    cfs = action.get(cfs_key) or action.get("counterfactual_action", [])

    options = [{"text": action["action"], "is_correct": True}]
    for cf in cfs:
        options.append({"text": cf, "is_correct": False})
    rng.shuffle(options)
    for i, opt in enumerate(options):
        opt["id"] = "ABCD"[i]
    correct_answer = next(opt["id"] for opt in options if opt["is_correct"])

    return {
        "question_id": f"{safe_title(book_title)}::{entity_id}::{action_uid}",
        "book": book_title,
        "character": canonical_name,
        "entity_id": entity_id,
        "action_uid": action_uid,
        "is_critical": action.get("effective_critical", False),
        "scene_uid": action.get("scene_uid", ""),
        "character_action_order": action.get("character_action_order", 0),
        "split": split_label,
        **ctx,
        "options": [{"id": o["id"], "text": o["text"]} for o in options],
        "correct_answer": correct_answer,
    }


# ---------------------------------------------------------------------------
# Train sample builder
# ---------------------------------------------------------------------------

def build_train_sample(book_title, timeline_data, profile_data, action,
                        scene_info, split_label, book_dir_layout):
    """Construct one preference training sample dict."""
    entity_id = timeline_data["entity_id"]
    canonical_name = timeline_data["canonical_name"]
    action_uid = action["action_uid"]

    ctx = build_context_fields(
        timeline_data, profile_data, action, scene_info,
        book_dir_layout)

    action_text = action["action"]
    cfs = action.get("counterfactual_action_rewashed") or action.get("counterfactual_action", [])
    rejected = pick_best_rejected(cfs, action_text)

    return {
        "sample_id": f"{safe_title(book_title)}::{entity_id}::{action_uid}",
        "book": book_title,
        "character": canonical_name,
        "entity_id": entity_id,
        "action_uid": action_uid,
        "is_critical": action.get("effective_critical", False),
        "scene_uid": action.get("scene_uid", ""),
        "character_action_order": action.get("character_action_order", 0),
        "split": split_label,
        **ctx,
        "chosen": action_text,
        "rejected": rejected,
    }


# ---------------------------------------------------------------------------
# Per-scene dedup (eval only)
# ---------------------------------------------------------------------------

def dedup_by_scene(actions, rng):
    scene_critical = defaultdict(list)
    scene_noncritical = defaultdict(list)
    for a in actions:
        sid = a.get("scene_uid", "")
        if a.get("effective_critical", False):
            scene_critical[sid].append(a)
        else:
            scene_noncritical[sid].append(a)
    critical_candidates = [rng.choice(v) for v in scene_critical.values()]
    noncritical_candidates = [rng.choice(v) for v in scene_noncritical.values()]
    return critical_candidates, noncritical_candidates


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def split_actions(actions, mode, split_ratio):
    """Return (eval_actions, train_actions) based on mode."""
    if mode == "all":
        return actions, []
    if mode == "train-only":
        return [], actions
    if mode == "split":
        if not actions:
            return [], []
        ordered = sorted(actions,
                         key=lambda a: a.get("character_action_order", 0))
        split_point = math.ceil(len(ordered) * (1 - split_ratio))
        # tail = last split_ratio → eval; head = rest → train
        train = ordered[:split_point]
        eval_ = ordered[split_point:]
        return eval_, train
    raise ValueError(f"unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Main per-book processor
# ---------------------------------------------------------------------------

def process_book(books_root, title, args, rng):
    """Process one book, return (eval_questions, train_samples)."""
    # Clear caches per book
    _traj_cache.clear()
    _profile_cache.clear()
    _name_to_entity.clear()

    ct_dir, rewashed_dir, profile_dir = resolve_book_dirs(books_root, title)
    if ct_dir is None:
        print(f"  [skip] no character_timelines for {title}")
        return [], []

    safe = safe_title(title)
    traj_dir = os.path.join(books_root, "trajectories_auto", safe)

    # Load entity index
    index_path = os.path.join(ct_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"  [skip] no index.json")
        return [], []
    index = load_json(index_path)
    entities = index.get("entities", [])
    print(f"  {title}: {len(entities)} characters")

    # Load all profiles for participant descriptions
    if profile_dir and os.path.isdir(profile_dir):
        load_all_profiles(profile_dir)

    eval_questions = []
    train_samples = []

    for entity in entities:
        entity_id = entity["entity_id"]
        canonical_name = entity.get("canonical_name", entity_id)

        # Load rewashed timeline (fallback to original if not found)
        timeline = load_rewashed(rewashed_dir, entity_id)
        if timeline is None:
            timeline = load_rewashed(ct_dir, entity_id)
        if timeline is None:
            continue

        # Load profile
        profile = load_profile(profile_dir, entity_id) if profile_dir else None

        actions = timeline.get("actions", [])
        if not actions:
            continue

        # Shared train/eval quality gate: require a generated profile. Missing
        # profiles become eligible automatically after a later profile retry.
        if not has_valid_profile(profile):
            continue

        # Split
        eval_actions, train_actions = split_actions(actions, args.mode, args.split_ratio)

        # --- Build eval questions ---
        if (eval_actions
                # Apply the character-level threshold to the complete timeline,
                # not merely the tail selected for eval in split mode.  Use the
                # loaded list as the source of truth in case action_count is
                # missing or stale.
                and len(actions) >= args.min_actions
                and is_profile_usable(profile, timeline)):
            # Gate 3: hard filter
            candidates = [a for a in eval_actions
                          if has_valid_rewashed_cfs(a, args.use_rewashed)
                          and not is_rewash_skipped(a)
                          and (args.include_first_action
                               or a.get("character_action_order", 0) != 0)]
            # Gate 4: per-scene dedup
            critical_pool, noncritical_pool = dedup_by_scene(candidates, rng)
            # Gate 5: stratified sampling
            n_critical = min(len(critical_pool), args.max_critical_per_char)
            n_noncritical = min(len(noncritical_pool), args.max_noncritical_per_char)
            selected_critical = rng.sample(critical_pool, n_critical) if n_critical > 0 else []
            selected_noncritical = rng.sample(noncritical_pool, n_noncritical) if n_noncritical > 0 else []
            selected = selected_critical + selected_noncritical
            for action in selected:
                scene_info = get_scene_info(traj_dir, action)
                q = build_eval_question(
                    title, timeline, profile, action, scene_info,
                    "tail" if args.mode == "split" else "all",
                    None, rng)
                eval_questions.append(q)
            if selected:
                print(f"    {canonical_name}: {len(selected_critical)}c + "
                      f"{len(selected_noncritical)}nc = {len(selected)} eval")

        # --- Build train samples ---
        if (train_actions
                and timeline.get("action_count", 0) >= args.min_train_actions):
            for action in train_actions:
                # Gate: hard filter (no dedup, no sampling)
                if not has_valid_rewashed_cfs(action, args.use_rewashed):
                    continue
                if is_rewash_skipped(action):
                    continue
                if action.get("character_action_order", 0) == 0:
                    continue  # no memory
                scene_info = get_scene_info(traj_dir, action)
                s = build_train_sample(
                    title, timeline, profile, action, scene_info,
                    "head" if args.mode == "split" else "all", None)
                train_samples.append(s)

    return eval_questions, train_samples


def build_input_signature(books_root, title):
    """Detect whether one book's upstream dataset inputs changed."""
    safe = safe_title(title)
    paths = (
        os.path.join(books_root, 'character_timelines', safe),
        os.path.join(books_root, 'character_timelines_rewashed', safe),
        os.path.join(books_root, 'profile_trials', safe),
        os.path.join(books_root, 'trajectories_auto', safe),
    )
    return ':'.join(tree_signature(path) for path in paths)


def output_meta_path(output_dir, title):
    return os.path.join(output_dir, safe_title(title) + '.meta.json')


def write_jsonl_atomic(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build eval questions and/or preference training samples")
    ap.add_argument("--books-root", required=True,
                    help="Root dir with character_timelines/, profile_trials/, etc.")
    ap.add_argument("--title", default=None, help="Single book title")
    ap.add_argument("--titles-file", default=None, help="File with one title per line")
    ap.add_argument("--mode", default="all", choices=["all", "split", "train-only"],
                    help="all=eval only; split=tail eval + head train; train-only=train only")
    ap.add_argument("--split-ratio", type=float, default=0.2,
                    help="Tail ratio for split mode (default 0.2)")
    ap.add_argument("--eval-output-dir", default=os.path.join(ROOT_DIR, "data", "dataset", "eval"))
    ap.add_argument("--train-output-dir", default=os.path.join(ROOT_DIR, "data", "dataset", "train"))
    ap.add_argument("--min-actions", type=int, default=20)
    ap.add_argument("--min-train-actions", type=int, default=5)
    ap.add_argument("--max-critical-per-char", type=int, default=15)
    ap.add_argument("--max-noncritical-per-char", type=int, default=5)
    ap.add_argument("--include-first-action", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rewashed", action="store_true",
                    help="Use original counterfactual_action instead of rewashed")
    args = ap.parse_args()

    args.use_rewashed = not args.no_rewashed
    args.books_root = os.path.abspath(args.books_root)
    if not 0 <= args.split_ratio <= 1:
        raise ValueError('--split-ratio must be between 0 and 1')
    if args.min_actions < 1 or args.min_train_actions < 1:
        raise ValueError('minimum action thresholds must be >= 1')
    rng = random.Random(args.seed)

    # Determine book list
    if args.title:
        books = [args.title]
    elif args.titles_file:
        with open(args.titles_file, encoding="utf-8") as f:
            books = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        books = list_book_titles(args.books_root)

    seen_safe_titles = {}
    for title in books:
        safe = safe_title(title)
        if safe in seen_safe_titles and seen_safe_titles[safe] != title:
            raise SystemExit(
                f"safe_title collision: {seen_safe_titles[safe]!r} vs {title!r}")
        seen_safe_titles[safe] = title

    print(f"Mode: {args.mode}, books: {len(books)}")
    print(f"  use_rewashed={args.use_rewashed}, min_actions={args.min_actions}, "
          f"max_critical={args.max_critical_per_char}, "
          f"max_noncritical={args.max_noncritical_per_char}, "
          f"min_train_actions={args.min_train_actions}")

    all_eval = []
    all_train = []

    for title in books:
        print(f"\n== {title} ==")
        eval_qs, train_ss = process_book(args.books_root, title, args, rng)
        all_eval.extend(eval_qs)
        all_train.extend(train_ss)

        # Write per-book output, including empty-but-valid JSONL files.
        source_signature = build_input_signature(args.books_root, title)
        meta = {
            'meta_version': BUILD_META_VERSION,
            'book': title,
            'mode': args.mode,
            'source_signature': source_signature,
            'config': {
                'split_ratio': args.split_ratio,
                'min_actions': args.min_actions,
                'min_train_actions': args.min_train_actions,
                'max_critical_per_char': args.max_critical_per_char,
                'max_noncritical_per_char': args.max_noncritical_per_char,
                'include_first_action': args.include_first_action,
                'use_rewashed': args.use_rewashed,
                'seed': args.seed,
            },
        }
        if args.mode in ('all', 'split'):
            out_path = os.path.join(args.eval_output_dir,
                                    f"{safe_title(title)}.jsonl")
            write_jsonl_atomic(out_path, eval_qs)
            atomic_json_dump(
                output_meta_path(args.eval_output_dir, title),
                dict(meta, output_kind='eval', record_count=len(eval_qs)))
            print(f"  eval: {len(eval_qs)} questions -> {out_path}")

        if args.mode in ('split', 'train-only'):
            out_path = os.path.join(args.train_output_dir,
                                    f"{safe_title(title)}.jsonl")
            write_jsonl_atomic(out_path, train_ss)
            atomic_json_dump(
                output_meta_path(args.train_output_dir, title),
                dict(meta, output_kind='train', record_count=len(train_ss)))
            print(f"  train: {len(train_ss)} samples -> {out_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Total eval questions: {len(all_eval)}")
    print(f"  Total train samples:  {len(all_train)}")
    if all_eval:
        chars = len(set(q["entity_id"] for q in all_eval))
        print(f"  Eval characters: {chars}")
    if all_train:
        chars = len(set(s["entity_id"] for s in all_train))
        print(f"  Train characters: {chars}")


if __name__ == "__main__":
    main()
