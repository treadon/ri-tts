---
language:
- en
license: cc-by-4.0
task_categories:
- text-to-speech
- audio-classification
tags:
- dac
- audio-tokens
- speech
- tts
- codebook
- descript-audio-codec
- librispeech
pretty_name: Speech DAC Tokens (3 Codebooks)
size_categories:
- 100K<n<1M
---

# Speech DAC Tokens (3 Codebooks)

Pre-tokenized speech dataset using the [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec). Each audio clip has been encoded into discrete codebook tokens from DAC's first 3 residual vector quantization codebooks, paired with its text transcription.

## Dataset Summary

| Stat | Value |
|------|-------|
| **Total samples** | 241,451 |
| **Total audio** | ~780 hours |
| **Language** | English |
| **Codebooks** | 3 (of DAC's 9) |
| **Codebook size** | 1,024 entries each |
| **DAC model** | 44kHz |
| **Tokens per second** | ~258 (86 frames x 3 codebooks) |
| **Token sequence length** | 219-4,096 (mean: 3,063) |
| **Audio duration range** | ~0.8s-15.7s |

## Data Sources

| Source | Split | Clips | License |
|--------|-------|-------|---------|
| [LibriSpeech](https://www.openslr.org/12) clean-100 | train.100 | ~24,200 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [LibriSpeech](https://www.openslr.org/12) clean-360 | train.360 | ~88,500 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [LibriSpeech](https://www.openslr.org/12) other-500 | train.500 | ~128,750 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Format

Each row contains:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Original text transcription |
| `prompt` | string | Full training prompt: `{text}<\|audio_start\|><\|c1_X\|><\|c2_Y\|><\|c3_Z\|>...<\|audio_end\|>` |
| `input_ids` | list[int] | Pre-tokenized 3-codebook prompt. Ready for training. |
| `input_ids_1cb` | list[int] | Pre-tokenized 1-codebook prompt (c1 only, shorter sequences). |
| `input_ids_2cb` | list[int] | Pre-tokenized 2-codebook prompt (c1+c2). |
| `attention_mask` | list[int] | All 1s, same length as input_ids (3cb). |
| `labels` | list[int] | Copy of input_ids (3cb). Used as training targets. |
| `n_audio_frames` | int | Number of DAC time frames |
| `n_tokens` | int | Total token count (text + audio tokens) |

Audio tokens are interleaved per time frame: `c1, c2, c3, c1, c2, c3, ...` where:
- **c1** (codebook 1): Coarse structure - pitch, rhythm, broad spectral shape
- **c2** (codebook 2): Fine detail - residual from c1
- **c3** (codebook 3): Finest detail - residual from c1+c2

## Use Cases

### Text-to-Speech Training
Train a language model to predict DAC tokens from text input. The model learns to generate the audio token sequence, which is then decoded back to audio using DAC's decoder. No spectrogram or vocoder needed - just token prediction.

```
Input:  Hello world
Output: <|audio_start|><|c1_551|><|c2_118|><|c3_42|>...<|audio_end|>
-> DAC decoder -> audio waveform
```

### Audio Language Modeling
Train unconditional or conditional audio generation models using discrete tokens, similar to how language models generate text.

### Speech Understanding
Use the tokenized representation for speech classification, speaker identification, or other downstream tasks that benefit from discrete audio representations.

### Codec Research
Study the information captured at different codebook levels, or compare DAC's tokenization against other codecs (EnCodec, SpeechTokenizer).

## How to Decode Audio

```python
import torch
import dac
from dac.utils import load_model
import re

# Load DAC decoder
dac_model = load_model(tag="latest", model_type="44khz")
dac_model.eval()

# Parse tokens from a prompt
prompt = dataset[0]["prompt"]
pattern = r'<\|c(\d+)_(\d+)\|>'
matches = re.findall(pattern, prompt)

# Group into frames (every 3 tokens = 1 frame)
frames = []
frame = [None, None, None]
for cb_str, val_str in matches:
    cb = int(cb_str) - 1
    frame[cb] = int(val_str)
    if cb == 2:
        frames.append(list(frame))
        frame = [None, None, None]

# Decode: pad to 9 codebooks (DAC expects all 9)
codes = torch.tensor(frames).T.unsqueeze(0).long()
full_codes = torch.zeros(1, 9, codes.shape[2], dtype=torch.long)
full_codes[:, :3, :] = codes

with torch.no_grad():
    z = dac_model.quantizer.from_codes(full_codes)
    audio = dac_model.decode(z)

# audio[0, 0] is the waveform at 44100 Hz
```

## Related

- **Training code:** [treadon/ri-tts](https://github.com/treadon/ri-tts) on GitHub
- **Trained model:** [treadon/ri-tts-model](https://huggingface.co/treadon/ri-tts-model) on HuggingFace (when available)

## Processing Details

- Audio resampled from 16kHz (LibriSpeech native) to 44.1kHz (DAC native)
- Clips exceeding 4,096 tokens were excluded (~17% of source data)
- DAC encoding performed on Apple MPS (M4 Max) at ~2.4 clips/sec
- No word-level alignment or prosodic features - raw text + DAC codes only
- 0 CPU fallback failures during encoding

## Citation

If you use this dataset, please cite the original data sources:

```bibtex
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={ICASSP},
  year={2015}
}

@article{kumar2024high,
  title={High-fidelity audio compression with improved RVQGAN},
  author={Kumar, Rithesh and others},
  journal={NeurIPS},
  year={2024}
}
```
