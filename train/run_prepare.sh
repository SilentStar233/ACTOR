#!/usr/bin/env bash
# Prepare preference pairs from the final training partition.
# Usage: bash train/run_prepare.sh MODEL_PATH OUTPUT_DIR [INPUT_DIR]
set -Eeuo pipefail

cd "$(dirname "$0")/.."

if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash train/run_prepare.sh MODEL_PATH OUTPUT_DIR [INPUT_DIR]" >&2
  exit 2
fi

model_path="$1"
output_dir="$2"
input_dir="${3:-data/dataset_final/train}"

if [[ ! -d "$model_path" ]]; then
  echo "Model directory not found: $model_path" >&2
  exit 1
fi
if [[ ! -d "$input_dir" ]]; then
  echo "Training data directory not found: $input_dir" >&2
  exit 1
fi

python3 -u train/prepare_preference_dataset.py \
  --input-dir "$input_dir" \
  --output-dir "$output_dir" \
  --model-path "$model_path" \
  --prompt-module eval/dataset_prompts.py \
  --ablation no_book \
  --validation-ratio "${VALIDATION_RATIO:-0.01}" \
  --max-length "${MAX_LENGTH:-8192}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-7168}" \
  --overlength-policy "${OVERLENGTH_POLICY:-error}"
