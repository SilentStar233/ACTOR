#!/usr/bin/env bash
# RPO training on a prepared preference dataset.
# Usage: bash train/run_train.sh MODEL_PATH PREPARED_DIR OUTPUT_DIR
set -Eeuo pipefail

cd "$(dirname "$0")/.."

if (( $# != 3 )); then
  echo "Usage: bash train/run_train.sh MODEL_PATH PREPARED_DIR OUTPUT_DIR" >&2
  exit 2
fi

model_path="$1"
prepared_dir="$2"
output_dir="$3"
train_gpus="${TRAIN_GPUS:-1}"
resume="${RESUME:-none}"

if [[ ! -f "$model_path/config.json" ]]; then
  echo "Model config not found: $model_path/config.json" >&2
  exit 1
fi
if [[ ! -f "$prepared_dir/manifest.json" || ! -f "$prepared_dir/train.jsonl" ]]; then
  echo "Prepared data missing; run train/run_prepare.sh first: $prepared_dir" >&2
  exit 1
fi
if [[ ! "$train_gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_GPUS must be a positive integer" >&2
  exit 2
fi
if [[ "$resume" == none && -d "$output_dir" ]]; then
  shopt -s nullglob dotglob
  contents=("$output_dir"/*)
  if (( ${#contents[@]} > 0 )); then
    echo "Output directory is not empty: $output_dir. Set RESUME=auto or choose a new output path." >&2
    exit 1
  fi
fi

args=(train/train_rpo_qlora.py
  --model-path "$model_path"
  --dataset-dir "$prepared_dir"
  --output-dir "$output_dir"
  --model-precision "${MODEL_PRECISION:-4bit}"
  --max-length "${MAX_LENGTH:-8192}"
  --max-prompt-length "${MAX_PROMPT_LENGTH:-7168}"
  --beta "${BETA:-0.1}"
  --rpo-alpha "${RPO_ALPHA:-1.0}"
  --learning-rate "${LEARNING_RATE:-5e-6}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}"
  --max-steps "${MAX_STEPS:--1}"
  --batch-size "${BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --resume "$resume"
  --run-name "${RUN_NAME:-character-action-rpo-no-book}")

if [[ "${BNB_RESUME_FIX:-0}" == 1 ]]; then
  if [[ "$resume" == none ]]; then
    echo "BNB_RESUME_FIX=1 is only meaningful with RESUME=auto or a checkpoint path" >&2
    exit 2
  fi
  args=(train/resume_bnb_state.py "${args[@]}")
fi

if (( train_gpus == 1 )); then
  python3 -u "${args[@]}"
else
  torchrun --standalone --nnodes=1 --nproc-per-node="$train_gpus" "${args[@]}"
fi
