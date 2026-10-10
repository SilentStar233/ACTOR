#!/usr/bin/env bash
set -u

cd "$(dirname "$0")" || exit 1

if [[ ! -f config.json ]]; then
  echo "Missing config.json. Run: cp config.example.json config.json"
  exit 1
fi

MODEL="${MODEL:-gpt-5.6-luna}"

python3 check_pipeline.py \
  --src data/src/all_books.jsonl \
  --books-root data \
  --eval-only-books eval_only_books.txt \
  --train-eval-books train_eval_books.txt \
  --train-only-books train_only_books.txt || exit 1

python3 -u run_full_pipeline.py \
  --src data/src/all_books.jsonl \
  --books-root data \
  --eval-only-books eval_only_books.txt \
  --train-eval-books train_eval_books.txt \
  --train-only-books train_only_books.txt \
  --model "$MODEL" \
  --parallel-books 1 \
  --profile-workers 2 \
  --rewash-workers 2 \
  --anonymize-workers 2 \
  --parallel-anonymize-files 1 \
  --max-rpm 60 \
  --max-tpm 200000
