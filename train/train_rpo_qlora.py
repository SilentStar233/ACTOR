#!/usr/bin/env python3
"""Train a causal language model with LoRA/QLoRA-RPO preference optimization."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import bitsandbytes as bnb
import datasets
import peft
import torch
import transformers
import trl
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=7168)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rpo-alpha", type=float, default=1.0,
                        help="Weight of chosen NLL in RPO (default: 1.0).")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model-precision",
        choices=["4bit", "bf16"],
        default="4bit",
        help="Load the frozen base model in 4-bit QLoRA or native BF16 LoRA mode.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--stop-at-step",
        type=int,
        default=0,
        help="Stop cleanly after this absolute optimizer step; 0 disables it.",
    )
    parser.add_argument(
        "--milestone-steps",
        type=int,
        default=0,
        help="Copy adapter weights to output_dir/milestones every N steps; 0 disables it.",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--sample-selection",
        choices=["first", "random", "longest"],
        default="first",
        help="Subset selection used with --max-*-samples; longest is for memory stress tests.",
    )
    parser.add_argument(
        "--selection-max-total-tokens",
        type=int,
        default=None,
        help="Before subsetting, keep rows whose chosen/rejected total length is at most N.",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument(
        "--resume",
        default="none",
        help="'none', 'auto', or an explicit checkpoint-* directory.",
    )
    parser.add_argument("--run-name", default="character-action-rpo-no-book")
    parser.add_argument(
        "--model-parallel",
        action="store_true",
        help="Shard model layers over every visible GPU in one process (not DDP).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade activation memory for recomputation (disable for speed when memory allows).",
    )
    args = parser.parse_args()
    if args.rpo_alpha is not None and args.rpo_alpha <= 0:
        parser.error("--rpo-alpha must be positive")
    if args.max_prompt_length > args.max_length:
        parser.error("--max-prompt-length cannot exceed --max-length")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        parser.error("--max-train-samples must be positive")
    if args.milestone_steps < 0:
        parser.error("--milestone-steps must be non-negative")
    if args.stop_at_step < 0:
        parser.error("--stop-at-step must be non-negative")
    return args


class MilestoneAdapterCallback(TrainerCallback):
    """Keep adapter-only copies of selected checkpoints after rotation."""

    def __init__(self, interval: int):
        self.interval = interval

    def on_save(self, args, state, control, **kwargs):
        if self.interval <= 0 or state.global_step <= 0:
            return control
        if state.global_step % self.interval != 0 or not state.is_world_process_zero:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            return control
        destination = Path(args.output_dir) / "milestones" / checkpoint_dir.name
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
        copied = 0
        for source in checkpoint_dir.iterdir():
            if source.is_file() and (
                source.name.startswith("adapter_")
                or source.name in {"adapter_config.json", "README.md"}
            ):
                shutil.copy2(source, temporary / source.name)
                copied += 1
        if copied:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            temporary.rename(destination)
            print(f"Milestone adapter saved: {destination}", flush=True)
        elif temporary.exists():
            shutil.rmtree(temporary)
        return control


class StopAtStepCallback(TrainerCallback):
    """Stop at an absolute optimizer step without changing scheduler length."""

    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_at_step > 0 and state.global_step >= self.stop_at_step:
            control.should_save = True
            control.should_training_stop = True
            print(f"Stopping cleanly at optimizer step {state.global_step}", flush=True)
        return control


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def resolve_resume(value: str, output_dir: Path) -> str | None:
    if value == "none":
        return None
    if value == "auto":
        checkpoints = [
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and checkpoint_step(path) >= 0
        ]
        return str(max(checkpoints, key=checkpoint_step)) if checkpoints else None
    path = Path(value).resolve()
    if not path.is_dir():
        raise SystemExit(f"Resume checkpoint not found: {path}")
    return str(path)


def max_column(dataset: datasets.Dataset, name: str) -> int:
    values = dataset[name]
    return max(int(value) for value in values) if values else 0


def select_subset(
    dataset: datasets.Dataset,
    count: int | None,
    method: str,
    seed: int,
    max_total_tokens: int | None,
) -> datasets.Dataset:
    if max_total_tokens is not None:
        original_count = len(dataset)
        dataset = dataset.filter(
            lambda row: max(
                int(row["chosen_total_tokens"]),
                int(row["rejected_total_tokens"]),
            )
            <= max_total_tokens,
            desc=f"Selecting rows up to {max_total_tokens} total tokens",
        )
        print(
            f"Length filter <= {max_total_tokens}: kept {len(dataset):,}/{original_count:,} "
            f"rows ({len(dataset) / original_count if original_count else 0.0:.2%})",
            flush=True,
        )
        if len(dataset) == 0:
            raise SystemExit(
                f"No rows have total token length <= {max_total_tokens}"
            )
    if count is None or count >= len(dataset):
        return dataset
    if method == "first":
        return dataset.select(range(count))
    if method == "random":
        return dataset.shuffle(seed=seed).select(range(count))
    chosen_lengths = dataset["chosen_total_tokens"]
    rejected_lengths = dataset["rejected_total_tokens"]
    indices = sorted(
        range(len(dataset)),
        key=lambda index: max(
            int(chosen_lengths[index]), int(rejected_lengths[index])
        ),
        reverse=True,
    )[:count]
    return dataset.select(indices)


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    manifest_path = dataset_dir / "manifest.json"

    for required in (model_path / "config.json", train_path, manifest_path):
        if not required.is_file():
            raise SystemExit(f"Required file not found: {required}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "ready":
        raise SystemExit(
            f"Prepared dataset status is {manifest.get('status')!r}, expected 'ready'"
        )
    if manifest.get("ablation") != "no_book":
        raise SystemExit("This experiment requires an ablation='no_book' dataset")
    limits = manifest.get("limits", {})
    if limits.get("max_length") != args.max_length:
        raise SystemExit(
            "--max-length differs from the prepared dataset manifest: "
            f"{args.max_length} != {limits.get('max_length')}"
        )
    if limits.get("max_prompt_length") != args.max_prompt_length:
        raise SystemExit(
            "--max-prompt-length differs from the prepared dataset manifest: "
            f"{args.max_prompt_length} != {limits.get('max_prompt_length')}"
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this script must run on the GPU node")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
    gpu_count = torch.cuda.device_count()
    if args.model_parallel and local_rank >= 0:
        raise SystemExit("--model-parallel cannot be combined with torchrun/DDP")
    if args.model_parallel and gpu_count < 2:
        raise SystemExit(
            f"--model-parallel requires at least two visible GPUs, found {gpu_count}"
        )
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 8:
        raise SystemExit(f"BF16 requires an Ampere-or-newer GPU, found capability {capability}")

    print(
        "Versions:",
        f"torch={torch.__version__}",
        f"transformers={transformers.__version__}",
        f"datasets={datasets.__version__}",
        f"peft={peft.__version__}",
        f"trl={trl.__version__}",
        f"bitsandbytes={bnb.__version__}",
        flush=True,
    )
    for device_index in range(gpu_count):
        print(
            f"GPU {device_index}: {torch.cuda.get_device_name(device_index)}; "
            f"memory={torch.cuda.get_device_properties(device_index).total_memory / 2**30:.1f} GiB",
            flush=True,
        )

    data_files = {"train": str(train_path)}
    if validation_path.is_file() and validation_path.stat().st_size > 0:
        data_files["validation"] = str(validation_path)
    loaded = load_dataset("json", data_files=data_files)
    train_dataset = loaded["train"]
    eval_dataset = loaded.get("validation")
    train_dataset = select_subset(
        train_dataset,
        args.max_train_samples,
        args.sample_selection,
        args.seed,
        args.selection_max_total_tokens,
    )
    if eval_dataset is not None:
        if args.sample_selection == "first" and args.max_eval_samples is not None:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)
        eval_dataset = select_subset(
            eval_dataset,
            args.max_eval_samples,
            args.sample_selection,
            args.seed,
            args.selection_max_total_tokens,
        )
    if len(train_dataset) == 0:
        raise SystemExit("Training dataset is empty")

    largest_prompt = max_column(train_dataset, "prompt_tokens")
    largest_chosen = max_column(train_dataset, "chosen_total_tokens")
    largest_rejected = max_column(train_dataset, "rejected_total_tokens")
    if largest_prompt > args.max_prompt_length:
        raise SystemExit(
            f"Dataset prompt needs {largest_prompt} tokens, above {args.max_prompt_length}"
        )
    if max(largest_chosen, largest_rejected) > args.max_length:
        raise SystemExit(
            "Dataset completion needs "
            f"{max(largest_chosen, largest_rejected)} total tokens, above {args.max_length}"
        )
    print(
        f"Dataset: train={len(train_dataset):,}, "
        f"validation={len(eval_dataset) if eval_dataset is not None else 0:,}; "
        f"max prompt/total tokens={largest_prompt}/{max(largest_chosen, largest_rejected)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    if args.model_precision == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    device_map: str | dict[str, int]
    max_memory = None
    if args.model_parallel:
        device_map = "balanced"
        # Keep activation headroom on every visible card; do not assume A40s.
        max_memory = {
            index: f"{max(1, int(torch.cuda.get_device_properties(index).total_memory / 2**30 * 0.88))}GiB"
            for index in range(gpu_count)
        }
    elif local_rank >= 0:
        device_map = {"": local_rank}
    else:
        device_map = {"": 0}
    model_load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "max_memory": max_memory,
        "attn_implementation": "flash_attention_2",
    }
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_load_kwargs)
    model.config.use_cache = False
    if args.model_precision == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"Model precision: {args.model_precision}", flush=True)
    print("Optimization: completion-only logits enabled", flush=True)
    print(f"Attention implementation: {model.config._attn_implementation}", flush=True)
    print(f"Gradient checkpointing: {args.gradient_checkpointing}", flush=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"World size: {world_size}", flush=True)
    print(f"Per-device train batch size: {args.batch_size}", flush=True)
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}", flush=True)
    print(
        "Effective global batch size: "
        f"{args.batch_size * world_size * args.gradient_accumulation_steps}",
        flush=True,
    )
    print(f"Seed: {args.seed}", flush=True)
    print(f"Stop-at-step: {args.stop_at_step or 'disabled'}", flush=True)
    print(f"Milestone interval: {args.milestone_steps or 'disabled'} steps", flush=True)
    print(
        "Model device map:", getattr(model, "hf_device_map", None), flush=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    training_args = DPOConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        beta=args.beta,
        loss_type="sigmoid",
        rpo_alpha=args.rpo_alpha,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_num_proc=args.dataset_num_proc,
        use_num_logits_to_keep=True,
        ddp_find_unused_parameters=False,
    )
    callbacks = []
    if args.milestone_steps:
        callbacks.append(MilestoneAdapterCallback(args.milestone_steps))
    if args.stop_at_step:
        callbacks.append(StopAtStepCallback(args.stop_at_step))
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    resume_checkpoint = resolve_resume(args.resume, output_dir)
    print(f"Resume checkpoint: {resume_checkpoint or 'none'}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    for device_index in range(gpu_count):
        print(
            f"Peak CUDA memory GPU {device_index}: "
            f"allocated={torch.cuda.max_memory_allocated(device_index) / 2**30:.2f} GiB, "
            f"reserved={torch.cuda.max_memory_reserved(device_index) / 2**30:.2f} GiB",
            flush=True,
        )
    trainer.save_model(str(output_dir / "final_adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    if trainer.is_world_process_zero():
        print(f"Training complete. Adapter: {output_dir / 'final_adapter'}", flush=True)


if __name__ == "__main__":
    main()
