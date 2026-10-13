<h1 align="center">ACTOR</h1>

<p align="center">
  <strong>Build character behavior datasets from novels, and evaluate and train language models for character behavior prediction</strong>
</p>

<p align="center">
  <a href="#dataset-construction">Dataset Construction</a> ·
  <a href="#model-evaluation">Model Evaluation</a> ·
  <a href="#rpo-training">RPO Training</a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README_zh.md">简体中文</a>
</p>

## Introduction

ACTOR provides an end-to-end pipeline for constructing character behavior datasets from novels, together with tools for model evaluation and RPO training.

Given a character profile, historical memories, and the current scene, the model selects the candidate action that best fits the character. The repository supports three workflows:

- **Dataset construction**: Extract character behavior trajectories, generate profiles and counterfactual actions, build training and evaluation samples, and anonymize the resulting data.
- **Model evaluation**: Evaluate locally deployed or proprietary models through an OpenAI-compatible API, run context ablations, and score action-selection results.
- **RPO training**: Convert training samples into preference pairs and train with RPO using LoRA or QLoRA.

You must supply the original novels, API credentials, and model weights. Dataset construction calls model APIs and may incur costs; validate your configuration on a small number of books before running a full batch.

## Contents

- [Dataset Construction Overview](#dataset-construction-overview)
- [Environment Setup](#environment-setup)
- [Dataset Construction](#dataset-construction)
- [Model Evaluation](#model-evaluation)
- [RPO Training](#rpo-training)
- [Repository Layout](#repository-layout)

## Dataset Construction Overview

<p align="center">
  <a href="figures/pipeline.png">
    <img src="figures/pipeline.png" width="100%" alt="ACTOR core dataset construction: T-A-O extraction, behavior validation and alignment, character trajectories and profiles, and counterfactual action refinement">
  </a>
</p>

The figure shows T-A-O extraction, behavior validation and alignment, character trajectories and profiles, and counterfactual action refinement. The pipeline then builds the data partitions, anonymizes samples, and fills remaining name variants. **Anonymization follows sample construction**, and non-anonymized samples are retained. Click the image to view its original resolution.

## Environment Setup

Run all commands from the repository root after downloading or cloning the repository. The commands use Bash on Linux/macOS; Python 3.10 or later is recommended. Construction and evaluation require a working model API. RPO training additionally requires a Linux GPU environment and a base model.

```bash
python3 -m pip install -r requirements.txt
cp config.example.json config.json
```

Edit `config.json` with the API credentials for dataset construction:

```json
{
  "api_key": "your-api-key",
  "base_url": "https://your-service-address/v1"
}
```

## Dataset Construction

### Inputs and Partitions

Prepare the source novels and three book lists:

| File | Purpose and partitioning |
| --- | --- |
| `data/src/all_books.jsonl` | Original novel texts, one book per line |
| `eval_only_books.txt` | All actions enter the evaluation pool; filtering and sampling determine the questions |
| `train_eval_books.txt` | For each character, assign the first 80% of chronological actions to training and the last 20% to evaluation, then apply the respective filters |
| `train_only_books.txt` | All actions enter the training pool; filtering determines the samples |

Source record format:

```json
{ "title": "Book title", "author": "Author", "content": "Full novel text..." }
```

Each book must occupy one JSONL line. Encode line breaks in the novel text as `\n` inside the JSON string; do not format a record as multiline JSON or insert blank lines.

Each book-list line must contain a title that exactly matches `title` in the source. The three lists must be mutually disjoint and collectively cover all source books, with each book appearing exactly once. A list may be empty, but its empty file must still exist.

Assignment to a partition does not guarantee a final sample. Profile availability, action counts, and counterfactual quality affect eligibility; not every extracted action becomes a question or training sample.

### Run the Pipeline

```bash
MODEL=your-model-name bash run_all.sh
```

Replace `your-model-name` with a model exposed by your service. If `MODEL` is omitted, the default is `gpt-5.6-luna`.

The orchestrator checks inputs, API configuration, and basic dependencies, then runs all construction stages sequentially. The configuration check does not test API connectivity, model availability, or parameter compatibility.

`run_all.sh` uses conservative settings: `1` parallel book, `2` workers each for profiles, action refinement, and single-file anonymization, `1` parallel anonymization file, `60` RPM, and `200000` TPM.

### Failures and Resuming

Failures for individual books, characters, or actions are recorded while downstream stages process valid artifacts. Rerunning reuses completed work according to each stage’s resume rules. If a stage reports failure, the orchestrator returns a nonzero exit code after running subsequent stages; inspect the summary and logs.

Resume does not guarantee complete coverage or automatically rebuild artifacts after configuration changes. Review the relevant stage’s cache and output handling before rebuilding.

### Outputs

```text
data/
├── dataset/
│   ├── eval/              # Non-anonymized evaluation questions
│   └── train/             # Non-anonymized training samples
├── dataset_anon/          # Intermediate LLM-anonymized artifacts
├── dataset_final/
│   ├── eval/              # Final anonymized questions, organized by book
│   └── train/             # Final anonymized training samples, organized by book
└── logs/full_pipeline/
    └── summary.json       # Stage execution summary
```

<details>
<summary><strong>Advanced: concurrency, rate limits, and custom arguments</strong></summary>

`run_all.sh` uses fixed conservative concurrency and rate limits. Apart from `MODEL`, these settings cannot be overridden through environment variables. To adjust them, invoke the Python orchestrator directly:

```bash
python3 run_full_pipeline.py \
  --eval-only-books eval_only_books.txt \
  --train-eval-books train_eval_books.txt \
  --train-only-books train_only_books.txt \
  --model your-model-name \
  --parallel-books 2 --max-rpm 120 --max-tpm 400000
```

The defaults below apply when **running `run_full_pipeline.py` directly**, not to the conservative Bash entry point:

| Argument | Default | Description |
| --- | ---: | --- |
| `--model` | `gpt-5.6-luna` | Model used for extraction, judgment, profiles, action refinement, and anonymization |
| `--parallel-books` | `8` | Concurrent books during extraction, profile generation, and action refinement |
| `--profile-workers` | `5` | Concurrent profile workers per book |
| `--rewash-workers` | `8` | Concurrent action-refinement workers per book |
| `--anonymize-workers` | `8` | Concurrent anonymization workers per file |
| `--parallel-anonymize-files` | `1` | Concurrent files; together with per-file workers, affects request concurrency |
| `--max-rpm` | `2000` | Requests-per-minute limit passed to model-calling batch stages |
| `--max-tpm` | `4000000` | Tokens-per-minute limit passed to model-calling batch stages |

Adjust concurrency and rate limits to your actual API quota.

</details>

## Model Evaluation

### Question Illustration

<p align="center">
  <a href="figures/benchmark_example.png">
    <img src="figures/benchmark_example.png" width="100%" alt="ACTOR task illustration: select the next action from a character profile, long-term memory, recent experiences, and the current scene">
  </a>
</p>

**Task illustration, not a reproduction of a complete dataset question.** Images are for presentation only: the actual input is text, and the model answers A/B/C/D. Current Scene and Recent Memory are combined visually here but stored separately as `scene` and `recent_memory`.

<details>
<summary><strong>Evaluation JSONL fields</strong></summary>

| Field | Content |
| --- | --- |
| `profile` | Current profile; the prompt uses its `text` |
| `long_term_memory` / `recent_memory` | Key historical and recent experiences, containing Thought, Action, and Observation |
| `scene` | Scene description, time, and participant information |
| `options` | Four candidate actions, each with `id` and `text` |
| `correct_answer` | Reference answer for scoring only; not supplied to the model |
| `question_id` / `split` / `is_critical` | Question identifier, evaluation partition, and behavior type |

Each line contains one JSON object. Standard anonymized evaluation does not display original book titles or character names.

</details>

### Run Evaluation

Configure an OpenAI-compatible service and run (local vLLM example):

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"
bash eval/run_all.sh actual-served-model-name
```

The model service must already be running; this repository does not download weights or start a server. For remote or proprietary APIs, replace the endpoint, key, and model name. `EMPTY` only applies to local services accepting that value.

Environment variables take precedence over the root-level `config.json`; without a configured endpoint, the default OpenAI endpoint is used.

By default, the entry point:

- Merges `data/dataset_final/eval/*.jsonl` into `data/eval_questions_all.jsonl`.
- Uses the `no_book` protocol and the full-context `baseline` setting.
- Uses an output budget of `4`, `temperature=0`, `seed=0`, and `4` concurrent requests.
- Does not enable answer-choice constraints or change the model’s thinking behavior.
- Resumes automatically and generates overall scores, plus partition scores when applicable.

> The default 4-token output budget was used in our local multiple-choice experiments. Some services, especially those requiring a reasoning budget, may need a larger value. Set `EVAL_MAX_TOKENS` explicitly.

Default result paths:

```text
data/eval_results/
├── results.jsonl
├── results.score.json
└── results.partition-score.json   # Generated when partition metadata is available
```

Set `EVAL_OUTPUT` to a separate result path for each experiment. Resume requires the same question file, prompts, model, endpoint, and protocol settings; the evaluator checks the saved protocol manifest.

<details>
<summary><strong>Advanced: evaluation parameters and direct Python usage</strong></summary>

| Environment variable | Python argument | Default / purpose |
| --- | --- | --- |
| `EVAL_QUESTIONS` | `--questions` | Automatically merges final anonymized questions by default; may specify an existing JSONL file |
| `EVAL_OUTPUT` | `--output` | `data/eval_results/results.jsonl` |
| `EVAL_SCORES` | `--scores` | `<result-path-without-extension>.score.json` |
| `EVAL_PARTITION_SCORES` | `--partition-scores` | `<result-path-without-extension>.partition-score.json` |
| `EVAL_WORKERS` | `--workers` | `4` concurrent requests |
| `EVAL_MAX_TOKENS` | `--max-tokens` | `4`, sent as `max_completion_tokens` |
| `EVAL_TEMPERATURE` | `--temperature` | `0`; `omit` corresponds to `--omit-temperature` |
| `EVAL_SEED` | `--seed` | `0`; `omit` corresponds to `--omit-seed` |
| `EVAL_ABLATION` | `--ablation` | `no_book`; standard protocol names are `no_book/full` |
| `EVAL_CONTEXT_VARIANT` | `--context-variant` | `baseline` |
| `EVAL_GUIDED_CHOICE` | `--guided-choice A B C D` | `off`; `on` enables answer-choice constraints |
| `EVAL_DISABLE_THINKING` | `--disable-thinking` | `off`; `on` sends `chat_template_kwargs.enable_thinking=false` |
| `EVAL_PARTITION_MODE` | `--partition-mode` | `auto`; also accepts `required/off` |
| `PYTHON` | — | `python3`; interpreter used by the evaluation Bash wrapper |

You can also invoke Python directly to evaluate and score:

```bash
python3 eval/run_eval.py \
  --questions /path/to/questions.jsonl \
  --output data/eval_results/my-run.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model served-model \
  --ablation no_book \
  --context-variant baseline \
  --max-tokens 4 \
  --temperature 0 \
  --seed 0 \
  --resume
```

Direct invocation of `run_eval.py` does not read the root-level `config.json` automatically. Configure the service through command-line arguments or environment variables.

Answer-choice constraints and disabling thinking are service extensions and may not be supported by every service or vLLM version. Unsupported parameters produce explicit errors; the script does not silently remove them and retry. Explicitly omitting temperature or seed, or changing other settings, also changes the evaluation protocol.

</details>

### Context Ablations

| `--context-variant` | Character profile | Long-term memory | Recent memory |
| --- | --- | --- | --- |
| `baseline` | Retained | Retained | Retained |
| `no_profile` | Removed | Retained | Retained |
| `no_recent` | Retained | Retained | Removed |
| `no_long` | Retained | Removed | Retained |
| `no_memory` | Retained | Removed | Removed |

Select a setting through `EVAL_CONTEXT_VARIANT`, such as `no_recent` to remove recent memory, and use a new `EVAL_OUTPUT`.

<details>
<summary><strong>Anonymized versus non-anonymized evaluation</strong></summary>

The `no_book` protocol uses anonymized questions, while the standard `full` setting uses corresponding non-anonymized questions. Changing `EVAL_ABLATION` alone does not restore names. For `full`, also provide the corresponding non-anonymized question file through `EVAL_QUESTIONS`:

```bash
EVAL_QUESTIONS=/path/to/matched-non-anonymous-questions.jsonl \
EVAL_ABLATION=full \
EVAL_OUTPUT=data/eval_results/my-model-full.jsonl \
  bash eval/run_all.sh served-model
```

Match questions by ID for anonymized versus non-anonymized comparisons. Do not assume that files from the two stages contain identical question sets.

</details>

### View Scores

```bash
python3 -m json.tool data/eval_results/results.score.json
```

<details>
<summary><strong>Recompute scores for existing results</strong></summary>

```bash
python3 eval/score_eval.py \
  --questions /path/to/matching-questions.jsonl \
  --results /path/to/results.jsonl \
  --output /path/to/results.score.json \
  --summary

python3 eval/score_eval_partitions.py \
  --questions /path/to/matching-questions.jsonl \
  --results /path/to/results.jsonl \
  --output /path/to/results.partition-score.json
```

The question file must match the results. Rescoring uses existing results and does not call the model API again.

</details>

## RPO Training

### Training Sample

The same character context is paired with **chosen / rejected action texts**. Below is an excerpt from a real training sample:

| Context summary | Character 1 is an abbess visiting Character 2 after breakfast; Character 2 responds with friendly conversation |
| --- | --- |
| chosen | Listens and smiles without saying a word while Character 2 talks of Court intrigues. |
| rejected | Hastens to unseal the register, exposing each sheltered pensioner to Character 2 imprudent curiosity. |

### Environment Setup

Prepare a base model, Linux, an NVIDIA GPU supporting BF16 and FlashAttention 2, and a compatible CUDA-enabled PyTorch build, then install:

```bash
python3 -m pip install -r train/requirements.txt
python3 -m pip install flash-attn --no-build-isolation
```

FlashAttention 2 must match your GPU, PyTorch, and CUDA environment. Construction and API evaluation do not need these training dependencies.

### Prepare Training Data

Convert the final training samples into the format required by the base model:

```bash
bash train/run_prepare.sh \
  /path/to/base-model \
  data/train_prepared/my-model
```

The default input is `data/dataset_final/train/`. An alternative input directory can be provided as the third positional argument.

Successful preparation produces `train.jsonl`, `validation.jsonl`, `manifest.json`, and `overlength.jsonl`. A deterministic hash of `sample_id` assigns approximately 1% to training validation, distinct from formal evaluation; small datasets may have no validation samples.

Defaults are `8192` tokens total and `7168` for the prompt. Overlength samples cause the script to write an audit and failure-status manifest, then stop without creating or replacing training and validation files or silently truncating samples. After reviewing `overlength.jsonl`, explicitly discard them with:

```bash
OVERLENGTH_POLICY=drop bash train/run_prepare.sh \
  /path/to/base-model \
  data/train_prepared/my-model
```

Use a separate preparation directory for each base model, since token-length checks depend on its tokenizer.

### Start Training

```bash
bash train/run_train.sh \
  /path/to/base-model \
  data/train_prepared/my-model \
  data/train_runs/my-model-rpo
```

The three positional arguments are the **base model directory, prepared data directory, and training output directory**, in that order.

Defaults are a single GPU, 4-bit QLoRA, `beta=0.1`, `rpo_alpha=1.0`, and 1 epoch. RPO uses TRL’s `DPOTrainer` with an additional chosen-response NLL term weighted by `rpo_alpha`. Outputs include checkpoints and `final_adapter/`: a **LoRA adapter, not full weights**, requiring the base model.

### Improvements After Training

These are actual predictions from Qwen2.5-7B-Instruct and ACTOR-Qwen-2.5-7B on identical anonymized questions. Action descriptions summarize the options; models returned letters.

| Case | Base model | After RPO | Reference |
| --- | --- | --- | --- |
| A steward encounters resistance (*King Lear*) | D: Protect the blind man ✗ | C: Fight the intervening character ✓ | C |
| Estrangement within a household (*Jane Eyre*) | C: Seek the other character’s company ✗ | B: Minimize conversation with them ✓ | B |

<details>
<summary><strong>Case provenance</strong></summary>

Question IDs: `King Lear::oswald::p029_s000_t002_a004`, `Jane Eyre::eliza_reed::p007_s000_t000_a000`. Training sample ID: `The Three Musketeers::abbess::p129_s003_t001_a001`.

Sources are the 20260916 formal evaluation artifacts and the 20260828 rebuilt training data; predictions were verified by ID. The historical experiments selected checkpoints using a subset of the formal evaluation set, not a fully independent model-selection protocol. Only individual cases are shown here, not aggregate performance.

</details>

<details>
<summary><strong>Advanced: training parameters and resuming</strong></summary>

| Environment variable | Default | Description |
| --- | ---: | --- |
| `TRAIN_GPUS` | `1` | GPUs on a single machine; values greater than 1 launch DDP through `torchrun` |
| `MODEL_PRECISION` | `4bit` | `4bit` for QLoRA; `bf16` for LoRA |
| `RPO_ALPHA` | `1.0` | Chosen-response NLL weight; must be positive in this configuration |
| `BETA` | `0.1` | Preference-loss parameter |
| `LEARNING_RATE` | `5e-6` | Learning rate |
| `NUM_TRAIN_EPOCHS` | `1` | Training epochs |
| `MAX_STEPS` | `-1` | Maximum optimization steps; `-1` trains according to the epoch setting |
| `BATCH_SIZE` | `1` | Batch size per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Gradient accumulation steps |
| `RESUME` | `none` | `auto` resumes from the latest checkpoint; an explicit path is also accepted |
| `BNB_RESUME_FIX` | `0` | Set to `1` only for the relevant optimizer-state restore errors |
| `MAX_LENGTH` | `8192` | Must match data preparation |
| `MAX_PROMPT_LENGTH` | `7168` | Must match data preparation |
| `RUN_NAME` | `character-action-rpo-no-book` | Training run name |

Data preparation also supports `VALIDATION_RATIO`, which defaults to `0.01`.

For example, use two GPUs and BF16:

```bash
TRAIN_GPUS=2 MODEL_PRECISION=bf16 \
  bash train/run_train.sh \
  /path/to/base-model \
  data/train_prepared/my-model \
  data/train_runs/my-model-rpo-bf16
```

Resume from the latest checkpoint in the same output directory:

```bash
RESUME=auto bash train/run_train.sh \
  /path/to/base-model \
  data/train_prepared/my-model \
  data/train_runs/my-model-rpo
```

Without resume, the output directory must be empty. Enable `BNB_RESUME_FIX=1` only for bitsandbytes paged optimizer state restore issues.

</details>

## Repository Layout

```text
ACTOR/
├── actor/                       # Dataset construction modules
│   ├── trajectory_extraction/   # Extraction, validation, and alignment
│   ├── character_profiles/      # Character profile generation
│   ├── action_refinement/       # Action reprocessing and counterfactual refinement
│   ├── dataset_building/        # Training and evaluation sample construction
│   ├── name_anonymization/      # Anonymization and name-variant cleanup
│   └── shared/                  # Shared utilities and API components
├── eval/                        # Evaluation, prompt construction, and scoring
├── train/                       # Training data preparation and RPO training
├── figures/                     # README illustrations
├── config.example.json          # API configuration template
├── requirements.txt             # Basic dependencies
├── run_all.sh                   # Dataset construction with conservative settings
└── run_full_pipeline.py         # Dataset construction with configurable arguments
```
