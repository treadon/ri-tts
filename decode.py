"""
ri-tts inference: Generate speech from text.
Text -> Qwen3 -> DAC tokens -> DAC decoder -> WAV

Usage:
  python decode.py "Hello world" -o hello.wav --codebooks 1
  python decode.py --from-tokens samples-1cb/step_001000/sample_0.txt -o test.wav --codebooks 1
"""

import re
import argparse
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

CODEBOOK_SIZE = 1024
DAC_SAMPLE_RATE = 44100


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("No GPU available. Need CUDA or MPS.")


def parse_audio_tokens(text, n_codebooks):
    """Extract codebook indices from generated token string."""
    # Try with audio_start marker first, then fall back to scanning all tokens
    match = re.search(r'<\|audio_start\|>(.*?)(?:<\|audio_end\|>|$)', text, re.DOTALL)
    audio_str = match.group(1) if match else text

    pattern = r'<\|c(\d+)_(\d+)\|>'
    matches = re.findall(pattern, audio_str)

    if not matches:
        print("No codebook tokens found!")
        return None

    # Group into frames
    frames = []
    current_frame = [None] * n_codebooks
    for cb_str, val_str in matches:
        cb = int(cb_str) - 1
        val = int(val_str)
        if cb < n_codebooks:
            current_frame[cb] = val
            if cb == n_codebooks - 1:
                if all(v is not None for v in current_frame):
                    frames.append(list(current_frame))
                current_frame = [None] * n_codebooks

    if not frames:
        print("No complete frames found!")
        return None

    codes = torch.tensor(frames).T.unsqueeze(0).long()
    print(f"  Parsed {len(frames)} audio frames ({len(frames)/86:.1f}s)")
    return codes


def decode_to_audio(codes, n_codebooks):
    """Decode DAC codes to audio waveform."""
    import dac
    from dac.utils import load_model

    print("  Loading DAC decoder...", flush=True)
    dac_model = load_model(tag="latest", model_type="44khz")
    dac_model.eval()

    # Pad to 9 codebooks
    full_codes = torch.zeros(1, 9, codes.shape[2], dtype=torch.long)
    full_codes[:, :n_codebooks, :] = codes

    with torch.no_grad():
        z, _, _ = dac_model.quantizer.from_codes(full_codes)
        audio = dac_model.decode(z)

    return audio[0, 0].cpu().numpy()


def generate_speech(text, model_dir, output_path, n_codebooks, max_tokens=2000):
    print(f"Generating speech for: '{text}' ({n_codebooks}cb)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    device = get_device()
    model = model.to(device)

    prompt = f"{text}<|audio_start|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"  Generating tokens...", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=False)
    print(f"  Generated {len(generated)} tokens", flush=True)

    codes = parse_audio_tokens(decoded, n_codebooks)
    if codes is None:
        print("Failed to parse audio tokens!")
        return

    audio = decode_to_audio(codes, n_codebooks)
    sf.write(str(output_path), audio, DAC_SAMPLE_RATE)
    print(f"  Saved to {output_path} ({len(audio)/DAC_SAMPLE_RATE:.1f}s)", flush=True)


def decode_from_tokens_file(tokens_file, output_path, n_codebooks):
    print(f"Decoding from {tokens_file} ({n_codebooks}cb)...", flush=True)

    with open(tokens_file) as f:
        content = f.read()

    codes = parse_audio_tokens(content, n_codebooks)
    if codes is None:
        print("Failed to parse audio tokens!")
        return

    audio = decode_to_audio(codes, n_codebooks)
    sf.write(str(output_path), audio, DAC_SAMPLE_RATE)
    print(f"  Saved to {output_path} ({len(audio)/DAC_SAMPLE_RATE:.1f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ri-tts: Text to Speech")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file")
    parser.add_argument("--model", "-m", default=None, help="Model directory")
    parser.add_argument("--codebooks", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--from-tokens", help="Decode from a saved tokens file")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens to generate")
    args = parser.parse_args()

    if args.model is None:
        args.model = f"checkpoints/ri-tts-{args.codebooks}cb/best"

    if args.from_tokens:
        decode_from_tokens_file(args.from_tokens, args.output, args.codebooks)
    elif args.text:
        generate_speech(args.text, args.model, args.output, args.codebooks, args.max_tokens)
    else:
        parser.print_help()
