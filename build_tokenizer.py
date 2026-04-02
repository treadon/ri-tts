"""
Build ri-tts tokenizer: Qwen3-0.6B + DAC codebook tokens.

Usage:
  python build_tokenizer.py                # 3 codebooks (default)
  python build_tokenizer.py --codebooks 1  # 1 codebook
"""

import argparse
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B"
CODEBOOK_SIZE = 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebooks", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--model", type=str, default=BASE_MODEL, help="Base model")
    args = parser.parse_args()

    n_cb = args.codebooks
    model_short = args.model.split("/")[-1].lower().replace("_", "-")
    output_dir = f"tokenizer-{n_cb}cb"

    print(f"Building tokenizer with {n_cb} codebook(s) from {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    original_vocab = len(tokenizer)
    print(f"  Original vocab size: {original_vocab}", flush=True)

    new_tokens = ["<|audio_start|>", "<|audio_end|>"]
    for cb in range(1, n_cb + 1):
        for i in range(CODEBOOK_SIZE):
            new_tokens.append(f"<|c{cb}_{i}|>")

    print(f"  Adding {len(new_tokens)} tokens ({n_cb} codebooks x {CODEBOOK_SIZE} + 2 control)", flush=True)
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"  New vocab size: {len(tokenizer)}", flush=True)

    # Verify
    test_tokens = ["<|audio_start|>", "<|c1_0|>", "<|c1_1023|>", "<|audio_end|>"]
    for t in test_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        print(f"  {t} -> id {tid}", flush=True)

    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
