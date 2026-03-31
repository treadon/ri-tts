"""
ri-tts inference: Generate speech from text.
Text → Qwen3 → DAC tokens → DAC decoder → WAV

Usage:
  python decode.py "Hello world" --output hello.wav
  python decode.py --from-tokens samples/step_001000/sample_0.txt --output test.wav
"""

import re
import argparse
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

N_CODEBOOKS = 3
CODEBOOK_SIZE = 1024
DAC_SAMPLE_RATE = 44100


def parse_audio_tokens(text):
    """Extract codebook indices from generated token string."""
    # Find everything between audio_start and audio_end (or end of string)
    match = re.search(r'<\|audio_start\|>(.*?)(?:<\|audio_end\|>|$)', text, re.DOTALL)
    if not match:
        print("No audio tokens found!")
        return None

    audio_str = match.group(1)

    # Extract all codebook tokens
    pattern = r'<\|c(\d+)_(\d+)\|>'
    matches = re.findall(pattern, audio_str)

    if not matches:
        print("No codebook tokens found!")
        return None

    # Group into frames (every N_CODEBOOKS tokens = 1 frame)
    frames = []
    current_frame = [None] * N_CODEBOOKS
    for cb_str, val_str in matches:
        cb = int(cb_str) - 1  # 0-indexed
        val = int(val_str)
        if cb < N_CODEBOOKS:
            current_frame[cb] = val
            if cb == N_CODEBOOKS - 1:  # Last codebook completes a frame
                if all(v is not None for v in current_frame):
                    frames.append(list(current_frame))
                current_frame = [None] * N_CODEBOOKS

    if not frames:
        print("No complete frames found!")
        return None

    # Convert to tensor [1, n_codebooks, n_frames]
    codes = torch.tensor(frames).T.unsqueeze(0).long()
    print(f"  Parsed {len(frames)} audio frames from {len(matches)} tokens")
    return codes


def decode_to_audio(codes):
    """Decode DAC codes to audio waveform."""
    import dac
    from dac.utils import load_model

    print("  Loading DAC decoder...", flush=True)
    dac_model = load_model(tag="latest", model_type="44khz")
    dac_model.eval()

    # DAC expects codes with all codebooks (9), pad unused with zeros
    full_codes = torch.zeros(1, 9, codes.shape[2], dtype=torch.long)
    full_codes[:, :N_CODEBOOKS, :] = codes

    with torch.no_grad():
        # Get quantized representation from codes
        z = dac_model.quantizer.from_codes(full_codes)
        # Decode to audio
        audio = dac_model.decode(z)

    audio_np = audio[0, 0].cpu().numpy()
    return audio_np


def generate_speech(text, model_dir, output_path, max_tokens=2000):
    """Generate speech from text using trained model."""
    print(f"Generating speech for: '{text}'", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        raise RuntimeError("No GPU available. Need CUDA or MPS.")
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

    # Parse and decode
    codes = parse_audio_tokens(decoded)
    if codes is None:
        print("Failed to parse audio tokens!")
        return

    audio = decode_to_audio(codes)
    sf.write(str(output_path), audio, DAC_SAMPLE_RATE)
    print(f"  Saved to {output_path} ({len(audio)/DAC_SAMPLE_RATE:.1f}s)", flush=True)


def decode_from_tokens_file(tokens_file, output_path):
    """Decode audio from a saved tokens file."""
    print(f"Decoding from {tokens_file}...", flush=True)

    with open(tokens_file) as f:
        content = f.read()

    codes = parse_audio_tokens(content)
    if codes is None:
        print("Failed to parse audio tokens!")
        return

    audio = decode_to_audio(codes)
    sf.write(str(output_path), audio, DAC_SAMPLE_RATE)
    print(f"  Saved to {output_path} ({len(audio)/DAC_SAMPLE_RATE:.1f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ri-tts: Text to Speech")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file")
    parser.add_argument("--model", "-m", default="checkpoints/ri-tts/best", help="Model directory")
    parser.add_argument("--from-tokens", help="Decode from a saved tokens file instead of generating")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens to generate")
    args = parser.parse_args()

    if args.from_tokens:
        decode_from_tokens_file(args.from_tokens, args.output)
    elif args.text:
        generate_speech(args.text, args.model, args.output, args.max_tokens)
    else:
        parser.print_help()
