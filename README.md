# 1. Prepare the Data and Environment

Run from the repository root:

```bash
python3 -m pip install -r requirements.txt
cp config.example.json config.json
```

Edit `config.json` to provide the API key and OpenAI-compatible endpoint used for dataset construction:

```json
{
  "api_key": "your-api-key",
  "base_url": "https://your-service-address/v1"
}
```

Then prepare these four input files:

| Path                       | Contents                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `data/src/all_books.jsonl` | Original novel texts; one JSON object per book per line                               |
| `eval_only_books.txt`      | Titles of books used only to generate evaluation questions                            |
| `train_eval_books.txt`     | Titles of books split into training and evaluation sets in chronological action order |
| `train_only_books.txt`     | Titles of books used only to generate training samples                                |

Source text format:

```json
{ "title": "Book title", "author": "Author", "content": "Full novel text..." }
```

Book list requirements:

**Each line must contain one book title that exactly matches the source `title`.** Book lists may be empty, but the three lists must not overlap and must collectively cover every book in the source exactly once. Empty lists must still be created as empty files.

# 2. Build the Dataset

Once the inputs are ready, run from the repository root:

```bash
bash run_all.sh
```

The script first validates the source texts and book lists, then runs each stage in sequence. The default model is `gpt-5.6-luna`. If your endpoint provides another model, specify it as follows:

```bash
MODEL=your-model-name bash run_all.sh
```

`run_all.sh` uses fixed, conservative concurrency and rate limits. Apart from `MODEL`, these values cannot be overridden through environment variables. To adjust them for a batch run, invoke the main pipeline script directly, for example:

```bash
python3 run_full_pipeline.py \
  --src data/src/all_books.jsonl --books-root data \
  --eval-only-books eval_only_books.txt \
  --train-eval-books train_eval_books.txt \
  --train-only-books train_only_books.txt \
  --model your-model-name \
  --parallel-books 8 --profile-workers 5 --rewash-workers 8 \
  --anonymize-workers 8 --parallel-anonymize-files 1 \
  --max-rpm 2000 --max-tpm 4000000
```

Common optional arguments are listed below. These defaults apply when **running `run_full_pipeline.py` directly**.

| Argument                     |        Default | Description                                                                                                           |
| ---------------------------- | -------------: | --------------------------------------------------------------------------------------------------------------------- |
| `--model`                    | `gpt-5.6-luna` | Model used during construction; shared by passage extraction, judgment, profile generation, rewash, and anonymization |
| `--parallel-books`           |              8 | Number of books processed concurrently during the blocks, profile, and rewash stages                                  |
| `--profile-workers`          |              5 | Number of concurrent profile workers per book                                                                         |
| `--rewash-workers`           |              8 | Number of concurrent rewash workers per book                                                                          |
| `--anonymize-workers`        |              8 | Number of concurrent workers when anonymizing a single file                                                           |
| `--parallel-anonymize-files` |              1 | Number of files anonymized concurrently; together with the preceding setting, determines actual request concurrency   |
| `--max-rpm`                  |           2000 | API requests-per-minute limit passed to batch stages that call the model                                              |
| `--max-tpm`                  |        4000000 | API tokens-per-minute limit passed to batch stages that call the model                                                |

If processing fails for some books, characters, or actions, the pipeline records the issues and allows subsequent stages to continue with the valid outputs already available. Running `bash run_all.sh` again skips completed work according to each stage's resume rules. **This does not mean that samples were successfully generated for every input.** Check the stage summaries and actual files.

The main outputs are located at:

```text
data/dataset_final/eval/              Anonymized evaluation questions (JSONL files organized by book)
data/dataset_final/train/             Anonymized training samples (JSONL files organized by book)
data/logs/full_pipeline/summary.json  Run summaries for all stages
```

For the versions before anonymization, see `data/dataset/eval/` and `data/dataset/train/`. Intermediate anonymization outputs are stored in `data/dataset_anon/`.

# 3. Evaluation

## Unified Entry Point

From the repository root, set the service endpoint and API key (local vLLM services typically use `EMPTY`), then run:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"
bash eval/run_all.sh actual-served-model-name
```

Environment variables take precedence over the root-level `config.json`. If no service endpoint is set, the default OpenAI endpoint is used. By default, the entry point merges `data/dataset_final/eval/*.jsonl`, uses **no_book, max_tokens=4, temperature=0, seed=0, no guided-choice constraints, and no changes to thinking behavior**, and resumes automatically. The 4-token limit is the setting used in our local multiple-choice experiments; it may not suit other services that require longer outputs or a larger reasoning budget.

`no_book` uses anonymized questions, while the standard `full` setting uses the corresponding non-anonymized questions. Changing only `--ablation` does not restore character names. When running `full`, you must also specify the corresponding non-anonymized question file through `EVAL_QUESTIONS` or `--questions`.

You can also invoke the same Python entry point directly to run evaluation and scoring together:

```bash
python3 eval/run_eval.py \
  --questions /path/to/questions.jsonl \
  --output data/eval_results/my-run.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model served-model \
  --ablation no_book --context-variant baseline \
  --max-tokens 4 --temperature 0 --seed 0 --resume
```

### Shared Parameters

| Wrapper Environment Variable | Python Argument           | Default / Description                                                                             |
| ---------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------- |
| `EVAL_QUESTIONS`             | `--questions`             | Automatically merges the final anonymized questions; an existing JSONL file may also be specified |
| `EVAL_OUTPUT`                | `--output`                | `data/eval_results/results.jsonl`; use a different path for each experiment                       |
| `EVAL_SCORES`                | `--scores`                | `<results-path-without-extension>.score.json`                                                     |
| `EVAL_PARTITION_SCORES`      | `--partition-scores`      | `<results-path-without-extension>.partition-score.json`                                           |
| `EVAL_WORKERS`               | `--workers`               | `4` concurrent requests                                                                           |
| `EVAL_MAX_TOKENS`            | `--max-tokens`            | `4`, sent as `max_completion_tokens`                                                              |
| `EVAL_TEMPERATURE`           | `--temperature`           | `0`; setting it to `omit` corresponds to `--omit-temperature`                                     |
| `EVAL_SEED`                  | `--seed`                  | `0`; setting it to `omit` corresponds to `--omit-seed`                                            |
| `EVAL_ABLATION`              | `--ablation`              | `no_book`; the standard protocol names are `no_book/full`                                         |
| `EVAL_CONTEXT_VARIANT`       | `--context-variant`       | `baseline`, retaining the full input context                                                      |
| `EVAL_GUIDED_CHOICE`         | `--guided-choice A B C D` | `off`; setting it to `on` enables answer-choice constraints                                       |
| `EVAL_DISABLE_THINKING`      | `--disable-thinking`      | `off`; setting it to `on` sends `chat_template_kwargs.enable_thinking=false`                      |
| `EVAL_PARTITION_MODE`        | `--partition-mode`        | `auto`; also accepts `required/off`                                                               |
| `PYTHON`                     | —                         | `python3`; specifies the interpreter used by the wrapper script                                   |

Guided-choice constraints and disabling thinking are service extensions and are not guaranteed to be supported by every service or vLLM version. Unsupported parameters produce explicit errors. **The pipeline will not automatically remove parameters and retry, silently changing the protocol.** If the API does not support temperature or seed, explicitly use the corresponding omit option; this constitutes a different protocol.

For example, enable answer-choice constraints and remove recent memory through the same entry point:

```bash
EVAL_QUESTIONS=/path/to/anonymous-questions.jsonl \
EVAL_OUTPUT=data/eval_results/no-recent-guided.jsonl \
EVAL_CONTEXT_VARIANT=no_recent EVAL_GUIDED_CHOICE=on \
  bash eval/run_all.sh served-model
```

### Input Context Ablations

| `--context-variant` | Character Profile | Long-Term Memory | Recent Memory |
| ------------------- | ----------------- | ---------------- | ------------- |
| `baseline`          | Retained          | Retained         | Retained      |
| `no_profile`        | Removed           | Retained         | Retained      |
| `no_recent`         | Retained          | Retained         | Removed       |
| `no_long`           | Retained          | Removed          | Retained      |
| `no_memory`         | Retained          | Removed          | Removed       |

## View or Recompute Scores for Existing Results

View the standard scores:

```bash
python3 -m json.tool data/eval_results/my-run.score.json
```

Compute scores for existing results:

```bash
python3 eval/score_eval.py \
  --questions /path/to/matching-questions.jsonl \
  --results /path/to/results.jsonl \
  --output /path/to/results.score.json --summary

python3 eval/score_eval_partitions.py \
  --questions /path/to/matching-questions.jsonl \
  --results /path/to/results.jsonl \
  --output /path/to/results.partition-score.json
```

# 4. RPO Training

## Environment Setup

```bash
python3 -m pip install -r train/requirements.txt
python3 -m pip install flash-attn --no-build-isolation
```

## Data Preparation

First, convert the final training JSONL files organized by book into the model training format. Replace the model and output directories in the following command with your own paths. Use a separate output directory for each base model, since token-length checks depend on that model's tokenizer:

```bash
bash train/run_prepare.sh \
  /path/to/base-model \
  data/train_prepared/my-model
```

The preparation stage generates `train.jsonl`, `validation.jsonl`, `manifest.json`, and `overlength.jsonl`. By default, it deterministically selects 1% of the training data based on `sample_id` for validation during training. This is separate from the formal evaluation set previously partitioned by book. The default total-length limit is 8192 tokens, and the prompt-length limit is 7168 tokens. If overlength samples are encountered, the script writes an audit file and stops; it does not silently truncate them. After reviewing `overlength.jsonl`, if you decide to discard these samples, run:

```bash
OVERLENGTH_POLICY=drop bash train/run_prepare.sh \
  /path/to/base-model data/train_prepared/my-model
```

## Start Training

Then start RPO training. The three positional arguments are, in order, the **base model directory, prepared data directory, and training output directory**:

```bash
bash train/run_train.sh \
  /path/to/base-model \
  data/train_prepared/my-model \
  data/train_runs/my-model-rpo
```

The defaults are a single GPU, 4-bit QLoRA, `beta=0.1`, `rpo_alpha=1.0`, and 1 epoch. RPO is implemented here using TRL's `DPOTrainer` with an additional NLL term for the chosen answer; `rpo_alpha` controls the weight of this term. Outputs include checkpoints and `final_adapter/`. The latter contains a LoRA adapter, **not full model weights**.

Common training settings can be adjusted using environment variables before the command:

| Variable                           |         Default | Description                                                                                                                                        |
| ---------------------------------- | --------------: | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TRAIN_GPUS`                       |             `1` | Number of GPUs used on a single machine; values greater than 1 launch DDP through `torchrun`                                                       |
| `MODEL_PRECISION`                  |          `4bit` | Base model loading mode: `4bit` (QLoRA) or `bf16` (LoRA)                                                                                           |
| `RPO_ALPHA`                        |           `1.0` | Weight of the chosen-answer NLL term; must remain greater than 0 for the RPO configuration used here                                               |
| `BETA`                             |           `0.1` | Beta parameter for the preference loss                                                                                                             |
| `LEARNING_RATE`                    |          `5e-6` | Learning rate                                                                                                                                      |
| `NUM_TRAIN_EPOCHS`                 |             `1` | Number of training epochs                                                                                                                          |
| `MAX_STEPS`                        |            `-1` | Maximum number of optimization steps; `-1` trains according to the epoch count                                                                     |
| `BATCH_SIZE`                       |             `1` | Batch size per GPU                                                                                                                                 |
| `GRADIENT_ACCUMULATION_STEPS`      |             `8` | Number of gradient accumulation steps                                                                                                              |
| `RESUME`                           |          `none` | `auto` resumes from the latest checkpoint in the output directory; a specific checkpoint path may also be provided                                 |
| `BNB_RESUME_FIX`                   |             `0` | Set to `1` to enable a compatibility patch for bitsandbytes paged optimizer states when resuming; use only if you encounter related restore errors |
| `MAX_LENGTH` / `MAX_PROMPT_LENGTH` | `8192` / `7168` | Must match the length limits used during data preparation                                                                                          |

For example, use BF16 on two GPUs:

```bash
TRAIN_GPUS=2 MODEL_PRECISION=bf16 \
  bash train/run_train.sh /path/to/base-model \
  data/train_prepared/my-model data/train_runs/my-model-rpo-bf16
```
