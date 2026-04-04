---
language:
- en
license: cc-by-4.0
task_categories:
- text-to-speech
tags:
- dac
- audio-tokens
- speech
- tts
- codebook
- descript-audio-codec
- librispeech
- 16khz
pretty_name: Speech DAC Tokens 16kHz (2 Codebooks)
size_categories:
- 10K<n<100K
---

# Speech DAC Tokens 16kHz (2 Codebooks)

Pre-tokenized speech dataset using [DAC](https://github.com/descriptinc/descript-audio-codec) at 16kHz with 2 codebooks. Optimized for speech TTS training — 16kHz captures the full speech frequency range without wasting capacity on inaudible frequencies.

## Why 16kHz?

- **Speech lives below 8kHz** — 16kHz sample rate is sufficient (Nyquist)
- **50 tokens/sec per codebook** vs 87 at 44kHz — shorter sequences, faster training
- **2 codebooks at 16kHz produce intelligible speech** — verified by listening tests
- **No resampling needed** — LibriSpeech is natively 16kHz

## Dataset Summary

| Stat | Value |
|------|-------|
| **Total samples** | 28,535 |
| **Total audio** | ~100 hours |
| **Source** | LibriSpeech clean-100 |
| **Language** | English |
| **DAC model** | 16kHz, 2 of 12 codebooks |
| **Codebook size** | 1,024 entries each |
| **Tokens per second** | 100 (50/codebook x 2) |
| **Token sequence length** | 149-2,047 (mean: 1,327) |

## Format

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Original text transcription |
| `prompt` | string | `{text}<\|audio_start\|><\|c1_X\|><\|c2_Y\|>...<\|audio_end\|>` |
| `input_ids` | list[int] | Pre-tokenized with Qwen3-0.6B + 2cb DAC tokens |
| `attention_mask` | list[int] | All 1s |
| `labels` | list[int] | Copy of input_ids |
| `n_audio_frames` | int | Number of DAC time frames |
| `n_tokens` | int | Total token count |

Audio tokens interleaved: `c1, c2, c1, c2, ...` per frame.

## Related

- **Training code:** [treadon/ri-tts](https://github.com/treadon/ri-tts) on GitHub
- **44kHz dataset (3cb):** [treadon/speech-dac-tokens-3cb](https://huggingface.co/datasets/treadon/speech-dac-tokens-3cb) (241K samples, kept for reference)

## Citation

```bibtex
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={ICASSP},
  year={2015}
}
```
