# Shared helpers for v2 action extraction.
# Used by run_extract_v2.py for input loading, long-plot scene splitting, and
# quote fidelity checks.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import SCENE_SPLIT_PROMPT
from prompts_v2 import build_extraction_prompt_v2
from extract_trajectory import slice_scene_texts
from utils import safe_title

SINGLE_CALL_MAX_CHARS = 8000  # plots longer than this are extracted per-scene


def call_extraction_json(llm, prompt, model, max_tokens):
    """Call an extraction model via llm.call_json with JSON mode.

    All cache/retry/rate-limit behavior lives in llm.py; this wrapper only
    handles providers that reject response_format (e.g. DeepSeek-style
    APIs) by falling back to plain text + parse.
    """
    try:
        return llm.call_json(prompt, model=model, max_tokens=max_tokens,
                             json_mode=True)
    except Exception as error:
        # Fall back only when the provider itself rejects JSON mode (a 400
        # about response_format). Content-level failures such as
        # 'JSON parsing failed' must NOT trigger a wasteful re-call.
        import openai
        if isinstance(error, openai.BadRequestError):
            message = str(error).lower()
            if ('response_format' in message or 'json_object' in message
                    or 'json' in message):
                return llm.call_json(prompt, model=model,
                                     max_tokens=max_tokens)
        raise


def load_inputs(title, plots_file, contexts_dir):
    if not plots_file:
        raise ValueError('plots_file is required')
    plots_path = plots_file
    with open(plots_path, encoding='utf-8') as f:
        plots_data = json.load(f)
    ctx_path = os.path.join(contexts_dir, safe_title(title),
                            'character_context.json')
    with open(ctx_path, encoding='utf-8') as f:
        contexts = json.load(f)
    return plots_data, contexts


def plot_key_characters(plot):
    characters = []
    for character in plot.get('key_characters') or []:
        if isinstance(character, dict) and character.get('name'):
            characters.append(character)
        elif isinstance(character, str) and character:
            characters.append({'name': character, 'experience': ''})
    for name in plot.get('key_character_names') or []:
        if name and not any(character.get('name') == name for character in characters):
            characters.append({'name': name, 'experience': ''})
    return characters


def build_key_characters(plot, plot_ctx):
    plot_characters = plot_key_characters(plot)
    if plot_ctx:
        return [{'name': char,
                 'experience': next((c.get('experience', '') for c in plot_characters
                                     if c.get('name') == char), '')}
                for char in plot_ctx.keys()]
    return plot_characters


def check_quotes(data, plot_text):
    """Hard check: dialogue quotes must appear (near-)verbatim in the plot text.
    Tolerates quotes interrupted by dialogue tags via 4-word chunk matching."""
    norm = lambda s: ' '.join(s.split()).lower()
    content = norm(plot_text)
    n_quote = n_hit = 0
    for pl in data.get('plots', []):
        for s in pl.get('scenes', []):
            for t in s.get('trajectories', []):
                for st in t.get('steps', []):
                    a = st.get('action', '')
                    if '"' not in a:
                        continue
                    q = a.split('"')[1] if len(a.split('"')) > 1 else ''
                    if len(q) <= 15:
                        continue
                    n_quote += 1
                    nq = norm(q)
                    if nq in content:
                        n_hit += 1
                        continue
                    words = nq.split()
                    chunks = [' '.join(words[i:i + 4]) for i in range(0, len(words) - 3, 4)]
                    if chunks and sum(c in content for c in chunks) / len(chunks) >= 0.9:
                        n_hit += 1
    return n_quote, n_hit


def extract_one_scene(llm, title, author, plot, scene, scene_text, plot_ctx):
    participants = scene.get('participants', [])
    scene_ctx = {c: plot_ctx[c] for c in participants if c in plot_ctx}
    prompt = build_extraction_prompt_v2(
        title=title, author=author, text=scene_text,
        chapter_title=plot.get('chapter_title'),
        plot_summary=plot.get('summary'),
        key_characters=[{'name': c, 'experience': ''} for c in participants],
        character_context=scene_ctx,
    )
    data = call_extraction_json(llm, prompt, llm.default_model, max_tokens=65536)
    inner = (data.get('plots') or [{}])[0]
    return (inner.get('scenes') or [{}])[0]


def extract_character_scene(llm, title, author, plot, scene, scene_text,
                            plot_ctx, character):
    scene_ctx = {character: plot_ctx[character]} if character in plot_ctx else {}
    prompt = build_extraction_prompt_v2(
        title=title, author=author, text=scene_text,
        chapter_title=plot.get('chapter_title'),
        plot_summary=plot.get('summary'),
        key_characters=[{'name': character, 'experience': ''}],
        character_context=scene_ctx,
    )
    data = call_extraction_json(llm, prompt, llm.default_model, max_tokens=65536)
    inner = (data.get('plots') or [{}])[0]
    out = (inner.get('scenes') or [{}])[0]
    trajectories = out.get('trajectories') or []
    matching = [trajectory for trajectory in trajectories
                if trajectory.get('character') == character]
    if len(matching) != 1:
        raise ValueError(
            f'character fallback expected exactly one trajectory for {character}, '
            f'got {len(matching)} matching and {len(trajectories)} total')
    trajectory = matching[0]
    return out, trajectory


def merge_character_fallback(merged, character_result, trajectory):
    """Merge a single-character trajectory into the scene-level extraction result."""
    trajectories = merged.setdefault('trajectories', [])
    character = trajectory.get('character')
    if not character:
        raise ValueError('character fallback returned trajectory without character')
    if any(item.get('character') == character for item in trajectories):
        raise ValueError(f'duplicate fallback trajectory for {character}')
    trajectories.append(trajectory)
    for field in ('time_text', 'time_evidence'):
        if merged.get(field) is None and character_result.get(field) is not None:
            merged[field] = character_result[field]


def extract_scene_with_character_fallback(llm, title, author, plot, scene,
                                          scene_text, plot_ctx, fallback_enabled=True):
    if not fallback_enabled:
        return extract_one_scene(llm, title, author, plot, scene, scene_text, plot_ctx)
    try:
        return extract_one_scene(llm, title, author, plot, scene, scene_text, plot_ctx)
    except Exception as error:
        participants = scene.get('participants') or []
        if not participants:
            raise
        print(f'    scene fallback to per-character: {str(error)[:120]}')

    merged = {
        'description': scene.get('description'),
        'participants': participants,
        'relationship_context': scene.get('relationship_context', {}),
        'time_text': scene.get('time_text'),
        'time_evidence': scene.get('time_evidence'),
        'trajectories': [],
    }
    for character in participants:
        character_result, trajectory = extract_character_scene(
            llm, title, author, plot, scene, scene_text, plot_ctx, character)
        merge_character_fallback(merged, character_result, trajectory)
    return merged


def extract_per_scene(llm, title, author, plot, plot_ctx, fallback_enabled=True):
    """Split into scenes first, then run v2 extraction per scene and merge."""
    kc = build_key_characters(plot, plot_ctx)
    kc_str = '\n'.join(f"- {c['name']}" for c in kc) if kc else '(unknown)'
    split = call_extraction_json(
        llm,
        SCENE_SPLIT_PROMPT.format(
            title=title, summary=plot.get('summary', ''),
            key_characters=kc_str, text=plot['text']),
        llm.default_model,
        max_tokens=65536,
    )
    scenes = split.get('scenes', [])
    if not scenes:
        raise ValueError('scene split returned no scenes')
    scene_texts = slice_scene_texts(plot['text'], scenes)

    merged_scenes = []
    for i, (scene, scene_text) in enumerate(zip(scenes, scene_texts)):
        out = extract_scene_with_character_fallback(
            llm, title, author, plot, scene, scene_text, plot_ctx,
            fallback_enabled=fallback_enabled)
        out['scene_id'] = i
        out.setdefault('description', scene.get('description'))
        out.setdefault('participants', scene.get('participants', []))
        out.setdefault('relationship_context', scene.get('relationship_context', {}))
        merged_scenes.append(out)
        print(f'    scene {i}: {len(scene_text)} chars, '
              f'{sum(len(t.get("steps", [])) for t in out.get("trajectories", []))} steps')
    return {'plots': [{'plot_id': 0, 'chapter_title': plot.get('chapter_title'),
                       'scenes': merged_scenes}]}


