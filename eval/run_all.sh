#!/usr/bin/env bash
# Convenience wrapper for the unified evaluator; the server is started separately.
# Usage: bash eval/run_all.sh MODEL_NAME
set -Eeuo pipefail
cd "$(dirname "$0")/.."

model="${1:-${MODEL:-}}"
python="${PYTHON:-python3}"
if [[ -z "$model" ]]; then
  echo "Usage: bash eval/run_all.sh MODEL_NAME" >&2
  exit 2
fi
if ! "$python" -c 'import openai' >/dev/null 2>&1; then
  echo "Missing dependency: run $python -m pip install -r requirements.txt" >&2
  exit 1
fi

# Environment takes precedence; root config.json is an optional fallback.
if [[ -f config.json ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    OPENAI_API_KEY="$("$python" -c 'import json; print(json.load(open("config.json", encoding="utf-8")).get("api_key", ""))')"
    export OPENAI_API_KEY
  fi
  if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
    OPENAI_BASE_URL="$("$python" -c 'import json; print(json.load(open("config.json", encoding="utf-8")).get("base_url", ""))')"
    export OPENAI_BASE_URL
  fi
fi
if [[ -z "${OPENAI_API_KEY:-}" || "$OPENAI_API_KEY" == REPLACE_* ]]; then
  echo "Set OPENAI_API_KEY (EMPTY for local vLLM), or fill in config.json" >&2
  exit 1
fi

questions="${EVAL_QUESTIONS:-data/eval_questions_all.jsonl}"
if [[ -z "${EVAL_QUESTIONS:-}" ]]; then
  shopt -s nullglob
  files=(data/dataset_final/eval/*.jsonl)
  if (( ${#files[@]} == 0 )); then
    echo "No final eval JSONL in data/dataset_final/eval/. Build data or set EVAL_QUESTIONS." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$questions")"
  # mktemp prevents concurrent wrappers from sharing a merge temporary.
  tmp="$(mktemp "${questions}.XXXXXX")"
  trap 'rm -f -- "$tmp"' EXIT
  for file in "${files[@]}"; do
    cat -- "$file" >> "$tmp"
    # Protect against source files without a final newline.
    printf '\n' >> "$tmp"
  done
  mv -- "$tmp" "$questions"
  trap - EXIT
fi
if [[ ! -s "$questions" ]]; then
  echo "Question file is missing or empty: $questions" >&2
  exit 1
fi

guided="${EVAL_GUIDED_CHOICE:-off}"
thinking="${EVAL_DISABLE_THINKING:-off}"
for switch in "$guided" "$thinking"; do
  if [[ "$switch" != on && "$switch" != off ]]; then
    echo "EVAL_GUIDED_CHOICE and EVAL_DISABLE_THINKING must be on/off" >&2
    exit 2
  fi
done
command=("$python" -u eval/run_eval.py
  --questions "$questions"
  --output "${EVAL_OUTPUT:-data/eval_results/results.jsonl}"
  --model "$model"
  --workers "${EVAL_WORKERS:-4}"
  --max-tokens "${EVAL_MAX_TOKENS:-4}"
  --ablation "${EVAL_ABLATION:-no_book}"
  --context-variant "${EVAL_CONTEXT_VARIANT:-baseline}"
  --partition-mode "${EVAL_PARTITION_MODE:-auto}"
  --resume)
if [[ "${EVAL_TEMPERATURE:-0}" == omit ]]; then
  command+=(--omit-temperature)
else
  command+=(--temperature "${EVAL_TEMPERATURE:-0}")
fi
if [[ "${EVAL_SEED:-0}" == omit ]]; then
  command+=(--omit-seed)
else
  command+=(--seed "${EVAL_SEED:-0}")
fi
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then command+=(--base-url "$OPENAI_BASE_URL"); fi
if [[ -n "${EVAL_SCORES:-}" ]]; then command+=(--scores "$EVAL_SCORES"); fi
if [[ -n "${EVAL_PARTITION_SCORES:-}" ]]; then command+=(--partition-scores "$EVAL_PARTITION_SCORES"); fi
if [[ "$guided" == on ]]; then command+=(--guided-choice A B C D); fi
if [[ "$thinking" == on ]]; then command+=(--disable-thinking); fi
"${command[@]}"
