"""
Build ri-tts tokenizer: Qwen3-0.6B + DAC codebook tokens.

Adds:
- 3072 codebook tokens: c1_0..c1_1023, c2_0..c2_1023, c3_0..c3_1023
- 2 control tokens: <|audio_start|>, <|audio_end|>
"""

from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "tokenizer"
N_CODEBOOKS = 3
CODEBOOK_SIZE = 1024


def main():
    print(f"Loading base tokenizer from {BASE_MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    original_vocab = len(tokenizer)
    print(f"  Original vocab size: {original_vocab}", flush=True)

    # Build new special tokens
    new_tokens = []

    # Control tokens
    new_tokens.append("<|audio_start|>")
    new_tokens.append("<|audio_end|>")

    # Codebook tokens
    for cb in range(1, N_CODEBOOKS + 1):
        for i in range(CODEBOOK_SIZE):
            new_tokens.append(f"<|c{cb}_{i}|>")

    print(f"  Adding {len(new_tokens)} new tokens ({N_CODEBOOKS} codebooks x {CODEBOOK_SIZE} + 2 control)", flush=True)

    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    print(f"  New vocab size: {len(tokenizer)}", flush=True)
    print(f"  Added: {len(tokenizer) - original_vocab} tokens", flush=True)

    # Verify
    test_tokens = ["<|audio_start|>", "<|c1_0|>", "<|c1_1023|>", "<|c2_500|>", "<|c3_999|>", "<|audio_end|>"]
    for t in test_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        print(f"  {t} -> id {tid}", flush=True)

    # Test encode/decode round-trip
    test_str = "Hello world<|audio_start|><|c1_551|><|c2_118|><|c3_42|><|c1_474|><|c2_260|><|c3_99|><|audio_end|>"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    print(f"\n  Round-trip test:", flush=True)
    print(f"  Input:   {test_str}", flush=True)
    print(f"  Encoded: {encoded[-10:]}", flush=True)
    print(f"  Decoded: {decoded[-80:]}", flush=True)

    # Save
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
