"""
ri-tts training: Qwen3-0.6B -> TTS with DAC tokens.

Supports 1, 2, or 3 codebooks via --codebooks flag.
Pulls data from HuggingFace, auto-detects GPU (CUDA/MPS).

Usage:
  python train.py --codebooks 1                    # 1 codebook (fast experiment)
  python train.py --codebooks 3                    # 3 codebooks (full quality)
  python train.py --hf-repo treadon/ri-tts-model   # push checkpoints to HF
"""

import os
import re
import glob
import shutil
import argparse
import torch
import wandb
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from huggingface_hub import HfApi, create_repo

BASE_MODEL = "Qwen/Qwen3-0.6B"
HF_DATASET = "treadon/speech-dac-tokens-3cb"
SEED = 42

TEST_SENTENCES = [
    "Hello, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today?",
]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("No GPU available. Need CUDA or MPS.")


def strip_codebooks(prompt, keep_codebooks):
    """Strip c2/c3 tokens from a 3-codebook prompt to make a 1 or 2 codebook prompt."""
    if keep_codebooks >= 3:
        return prompt
    # Remove codebook tokens we don't want
    for cb in range(keep_codebooks + 1, 4):
        prompt = re.sub(rf'<\|c{cb}_\d+\|>', '', prompt)
    return prompt


class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        model = kwargs.get("model")
        if model is None:
            return

        print(f"\n  Generating test tokens at step {step}...", flush=True)
        model.eval()
        sample_dir = self.output_dir / f"step_{step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(TEST_SENTENCES):
            try:
                prompt = f"{text}<|audio_start|>"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1000,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1,
                    )

                generated = outputs[0][inputs["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(generated, skip_special_tokens=False)

                with open(sample_dir / f"sample_{i}.txt", "w") as f:
                    f.write(f"Text: {text}\n")
                    f.write(f"Generated tokens ({len(generated)}):\n")
                    f.write(decoded[:5000])

                n_audio = sum(1 for t in decoded.split("|>") if t.strip().startswith("<|c"))
                has_end = "<|audio_end|>" in decoded
                print(f"  Sample {i}: {n_audio} audio tokens, end_token={'yes' if has_end else 'no'}", flush=True)

            except Exception as e:
                print(f"  Sample {i} failed: {e}", flush=True)

        model.train()


class DiskCheckCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and state.global_step > 0:
            _, _, free = shutil.disk_usage("/")
            free_gb = free // (1024**3)
            if free_gb < 10:
                print(f"\n  DISK WARNING: {free_gb}GB free!", flush=True)
            if free_gb < 3:
                print(f"\n  CRITICAL: {free_gb}GB -- stopping!", flush=True)
                control.should_training_stop = True


class HFUploadCallback(TrainerCallback):
    def __init__(self, hf_repo, tokenizer):
        self.hf_repo = hf_repo
        self.tokenizer = tokenizer
        self.api = HfApi()
        self.uploaded = []

        try:
            create_repo(hf_repo, repo_type="model", exist_ok=True)
            print(f"  HF repo ready: {hf_repo}", flush=True)
        except Exception as e:
            print(f"  Warning: could not create HF repo: {e}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.exists(checkpoint_dir):
            return
        try:
            print(f"\n  Uploading checkpoint-{step} to {self.hf_repo}...", flush=True)
            self.api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=self.hf_repo,
                path_in_repo=f"checkpoint-{step}",
                repo_type="model",
            )
            self.uploaded.append(f"checkpoint-{step}")
            print(f"  Uploaded checkpoint-{step}", flush=True)
            while len(self.uploaded) > 2:
                old = self.uploaded.pop(0)
                try:
                    self.api.delete_folder(path_in_repo=old, repo_id=self.hf_repo, repo_type="model")
                except Exception:
                    pass
        except Exception as e:
            print(f"  Warning: HF upload failed: {e}", flush=True)


def find_latest_checkpoint(output_dir):
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
                         key=lambda x: int(x.split("-")[-1]))
    if checkpoints:
        latest = checkpoints[-1]
        step = int(latest.split("-")[-1])
        print(f"  Found checkpoint at step {step}: {latest}", flush=True)
        return latest
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="ri-tts training")
    parser.add_argument("--codebooks", type=int, default=3, choices=[1, 2, 3],
                        help="Number of DAC codebooks (1=fast experiment, 3=full quality)")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace repo for checkpoint uploads")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Filter to samples with at most this many tokens")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    n_cb = args.codebooks
    device = get_device()
    use_bf16 = device == "cuda"

    tokenizer_dir = f"tokenizer-{n_cb}cb"
    output_dir = f"checkpoints/ri-tts-{n_cb}cb"
    samples_dir = f"samples-{n_cb}cb"

    # Max sequence length scales with codebooks
    # 1cb: ~86 tokens/sec, 2cb: ~172, 3cb: ~258
    max_seq_len = args.max_tokens or (2048 if n_cb == 1 else 4096)

    print("=" * 60, flush=True)
    print(f"ri-tts Training ({n_cb} codebook{'s' if n_cb > 1 else ''})", flush=True)
    print(f"Device: {device}, bf16: {use_bf16}", flush=True)
    print(f"Context: {max_seq_len} tokens", flush=True)
    print("=" * 60, flush=True)

    _, _, free = shutil.disk_usage("/")
    print(f"Disk: {free // (1024**3)}GB free", flush=True)

    resume_checkpoint = find_latest_checkpoint(output_dir)

    # Build tokenizer if needed
    if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")):
        print(f"\nBuilding {n_cb}-codebook tokenizer...", flush=True)
        os.system(f"python build_tokenizer.py --codebooks {n_cb}")

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab: {len(tokenizer)}", flush=True)

    print("Loading model...", flush=True)
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=model_dtype, trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,}", flush=True)

    # Load data
    print(f"\nLoading data from {HF_DATASET}...", flush=True)
    hf_ds = load_dataset(HF_DATASET, split="train")
    print(f"  {len(hf_ds)} examples from HF", flush=True)

    # Pick the right pre-tokenized column
    ids_col = f"input_ids_{n_cb}cb" if n_cb < 3 else "input_ids"
    if ids_col in hf_ds.column_names:
        print(f"  Using pre-tokenized column: {ids_col} (instant)", flush=True)

        if args.max_tokens:
            before = len(hf_ds)
            hf_ds = hf_ds.filter(lambda x: len(x[ids_col]) <= args.max_tokens)
            print(f"  Filtered to <= {args.max_tokens}: {len(hf_ds)}/{before}", flush=True)

        # Rename to standard column names for the collate function
        if ids_col != "input_ids":
            hf_ds = hf_ds.rename_column(ids_col, "input_ids")
        ds = hf_ds
    else:
        # Fallback: strip codebooks and tokenize
        print(f"  Column {ids_col} not found, tokenizing...", flush=True)
        prompts = [strip_codebooks(p, n_cb) for p in hf_ds["prompt"]]
        BATCH = 1000
        all_input_ids = []
        for i in range(0, len(prompts), BATCH):
            batch = prompts[i:i + BATCH]
            toks = tokenizer(batch, max_length=max_seq_len, truncation=True, padding=False, return_tensors=None)
            all_input_ids.extend(toks["input_ids"])
            if (i // BATCH) % 50 == 0:
                print(f"    {min(i + BATCH, len(prompts))}/{len(prompts)}", flush=True)
        ds = Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": [[1] * len(ids) for ids in all_input_ids],
            "labels": all_input_ids,
        })

    split = ds.train_test_split(test_size=0.03, seed=SEED)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    def collate_fn(examples):
        max_len = min(max(len(e["input_ids"]) for e in examples), max_seq_len)
        input_ids, attention_mask, labels = [], [], []
        for e in examples:
            ids = list(e["input_ids"][:max_len])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(ids + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    n_train = len(train_ds)
    batch_size, grad_accum = 1, 16
    if use_bf16:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_mem:.0f}GB", flush=True)
    effective_batch = batch_size * grad_accum
    steps_per_epoch = n_train // effective_batch
    total_steps = steps_per_epoch * args.epochs
    print(f"  Batch: {batch_size}, Grad accum: {grad_accum}, Effective: {effective_batch}", flush=True)
    print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps} ({args.epochs} epochs)", flush=True)

    run_name = f"qwen3-0.6B-{n_cb}cb-{device}"
    wandb.init(
        project="ri-tts",
        name=run_name,
        config={
            "base_model": BASE_MODEL,
            "params": param_count,
            "codebooks": n_cb,
            "max_seq_len": max_seq_len,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "device": device,
            "bf16": use_bf16,
            "epochs": args.epochs,
        },
        tags=["tts", "dac", f"{n_cb}cb", "qwen3-0.6B"],
        resume="allow",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=False,
        max_grad_norm=1.0,
        seed=SEED,
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=4 if device == "cuda" else 0,
        remove_unused_columns=False,
    )

    callbacks = [
        GenerationCallback(tokenizer, samples_dir),
        DiskCheckCallback(),
    ]
    if args.hf_repo:
        callbacks.append(HFUploadCallback(args.hf_repo, tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    print("\nStarting training...", flush=True)
    if resume_checkpoint:
        print(f"  Resuming from {resume_checkpoint}", flush=True)
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    metrics = trainer.evaluate()
    print(f"\nFinal eval: {metrics}", flush=True)

    if args.hf_repo:
        print(f"\nUploading best model to {args.hf_repo}...", flush=True)
        api = HfApi()
        api.upload_folder(
            folder_path=best_dir,
            repo_id=args.hf_repo,
            path_in_repo="best",
            repo_type="model",
        )
        print(f"  Best model uploaded to {args.hf_repo}/best", flush=True)

    wandb.finish()
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
