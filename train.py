"""
ri-tts training: Qwen3-0.6B -> TTS with 3-codebook DAC tokens.

Pulls data from HuggingFace, auto-detects GPU (CUDA/MPS).

Features:
- Rolling checkpoints (keep last 2, optionally push to HF)
- Resume from checkpoint (local or HF)
- Periodic token generation for quality monitoring
- Disk space checks
- bf16 on CUDA, fp32 on MPS

Usage:
  python train.py                                  # local checkpoints only
  python train.py --hf-repo treadon/ri-tts-model   # push checkpoints to HF
"""

import os
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
TOKENIZER_DIR = "tokenizer"
OUTPUT_DIR = "checkpoints/ri-tts"
SAMPLES_DIR = "samples"
SEED = 42
MAX_SEQ_LEN = 4096

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


class GenerationCallback(TrainerCallback):
    """Generate test tokens at each checkpoint to monitor quality."""

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
    """Check disk space and stop if critical."""

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
    """Upload checkpoints to HuggingFace after each save. Rolling: keeps last 2."""

    def __init__(self, hf_repo, tokenizer):
        self.hf_repo = hf_repo
        self.tokenizer = tokenizer
        self.api = HfApi()
        self.uploaded = []

        # Create repo if it doesn't exist
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

            # Rolling: delete old checkpoints on HF, keep last 2
            while len(self.uploaded) > 2:
                old = self.uploaded.pop(0)
                try:
                    self.api.delete_folder(
                        path_in_repo=old,
                        repo_id=self.hf_repo,
                        repo_type="model",
                    )
                    print(f"  Deleted old HF checkpoint: {old}", flush=True)
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
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace repo for checkpoint uploads (e.g. treadon/ri-tts-model)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Filter to samples with at most this many tokens (e.g. 1024, 2048)")
    return parser.parse_args()


def main():
    args = parse_args()
    global MAX_SEQ_LEN
    device = get_device()
    use_bf16 = device == "cuda"
    if args.max_tokens:
        MAX_SEQ_LEN = args.max_tokens

    print("=" * 60, flush=True)
    print("ri-tts Training", flush=True)
    print(f"Device: {device}, bf16: {use_bf16}", flush=True)
    print(f"3 codebooks, {MAX_SEQ_LEN} context", flush=True)
    print("=" * 60, flush=True)

    _, _, free = shutil.disk_usage("/")
    print(f"Disk: {free // (1024**3)}GB free", flush=True)

    resume_checkpoint = find_latest_checkpoint(OUTPUT_DIR)

    # Load tokenizer
    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab: {len(tokenizer)}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=model_dtype, trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,}", flush=True)

    # Load data from HuggingFace
    print(f"\nLoading data from {HF_DATASET}...", flush=True)
    hf_ds = load_dataset(HF_DATASET, split="train")
    print(f"  {len(hf_ds)} examples from HF", flush=True)

    if args.max_tokens:
        before = len(hf_ds)
        hf_ds = hf_ds.filter(lambda x: x["n_tokens"] <= args.max_tokens)
        print(f"  Filtered to <= {args.max_tokens} tokens: {len(hf_ds)}/{before} samples", flush=True)

    # Use pre-tokenized columns from dataset (no tokenization or mapping)
    if all(c in hf_ds.column_names for c in ["input_ids", "attention_mask", "labels"]):
        print("  Using pre-tokenized columns from dataset (instant)", flush=True)
        ds = hf_ds
    elif "input_ids" in hf_ds.column_names:
        print("  Using pre-tokenized input_ids, adding attention_mask/labels...", flush=True)
        ds = hf_ds.select_columns(["input_ids", "n_tokens"])
        ds = ds.map(lambda x: {
            "attention_mask": [1] * len(x["input_ids"]),
            "labels": x["input_ids"],
        })
    else:
        print("  Tokenizing (no pre-tokenized column found)...", flush=True)
        prompts = list(hf_ds["prompt"])
        TOKENIZE_BATCH = 1000
        all_input_ids = []
        all_attention_mask = []
        for i in range(0, len(prompts), TOKENIZE_BATCH):
            batch = prompts[i:i + TOKENIZE_BATCH]
            batch_tok = tokenizer(
                batch, max_length=MAX_SEQ_LEN, truncation=True, padding=False, return_tensors=None
            )
            all_input_ids.extend(batch_tok["input_ids"])
            all_attention_mask.extend(batch_tok["attention_mask"])
            print(f"    {min(i + TOKENIZE_BATCH, len(prompts))}/{len(prompts)}", flush=True)
        ds = Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_input_ids,
        })

    split = ds.train_test_split(test_size=0.03, seed=SEED)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    def collate_fn(examples):
        max_len = min(max(len(e["input_ids"]) for e in examples), MAX_SEQ_LEN)
        input_ids, attention_mask, labels = [], [], []
        for e in examples:
            pad_len = max_len - len(e["input_ids"])
            input_ids.append(e["input_ids"][:max_len] + [tokenizer.pad_token_id] * max(0, pad_len))
            attention_mask.append(e["attention_mask"][:max_len] + [0] * max(0, pad_len))
            labels.append(list(e["labels"][:max_len]) + [-100] * max(0, pad_len))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    n_train = len(train_ds)
    # Auto-scale batch size based on GPU memory
    if use_bf16:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size, grad_accum = 1, 16
        print(f"  GPU: {gpu_mem:.0f}GB", flush=True)
    else:
        batch_size, grad_accum = 1, 16
    effective_batch = batch_size * grad_accum
    steps_per_epoch = n_train // effective_batch
    total_steps = steps_per_epoch * 3
    print(f"  Batch: {batch_size}, Grad accum: {grad_accum}, Effective: {effective_batch}", flush=True)
    print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps} (3 epochs)", flush=True)

    wandb.init(
        project="ri-tts",
        name=f"qwen3-0.6B-3cb-{device}",
        config={
            "base_model": BASE_MODEL,
            "params": param_count,
            "codebooks": 3,
            "max_seq_len": MAX_SEQ_LEN,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "device": device,
            "bf16": use_bf16,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
        },
        tags=["tts", "dac", "3codebook", "qwen3-0.6B"],
        resume="allow",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
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
        run_name=f"qwen3-0.6B-3cb-{device}",
        dataloader_num_workers=4 if device == "cuda" else 0,
        remove_unused_columns=False,
    )

    callbacks = [
        GenerationCallback(tokenizer, SAMPLES_DIR),
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

    best_dir = os.path.join(OUTPUT_DIR, "best")
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
