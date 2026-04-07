# ri-tts Experiment Journal

A chronological record of everything we tried, what worked, what failed, and what we learned.

---

## Phase 0: OuteTTS Replication (oute-tts/)

**Goal:** Train a TTS model using OuteTTS's pipeline and format.

**What we did:**
- Used OuteTTS's tokenizer, audio processor, prompt format, and DAC integration
- Swapped in Qwen3-0.6B as the base model (same as OuteTTS uses)
- Trained on 63K samples (LJSpeech 13K + LibriSpeech clean-360 50K)
- OuteTTS format: word-level alignment with Whisper, per-word features (timing, energy, pitch)

**Results:**
- Eval loss reached 4.25 after 1500 steps
- Model learned the format structure (word boundaries, features)
- Audio wasn't decodable — OOM'd during audio generation callback

**Why we stopped:** Realized we were just replicating OuteTTS with less data (780h vs their 20,000h). Not original enough.

**Lessons learned:**
- Disk management is critical (filled 926GB drive multiple times)
- Chunked saves + checkpoint resume saved hours of work
- Whisper-based data prep is slow (~0.5 clips/sec)
- MPS OOM is unpredictable — need `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`

---

## Phase 1: ri-tts with DAC 44kHz, 3 Codebooks

**Goal:** Build our own TTS pipeline from scratch, independent of OuteTTS.

**What we changed:**
- Our own tokenizer (Qwen3 + DAC codebook tokens)
- Our own simple format: `text<|audio_start|>c1 c2 c3 c1 c2 c3...<|audio_end|>`
- No word-level alignment, no Whisper, no per-word features
- Direct DAC encoding (not through OuteTTS wrapper)
- 3 codebooks at 44kHz DAC (258 tokens/sec)

**Dataset:** 241K samples, ~780 hours (LibriSpeech clean-100 + clean-360 + other-500)
- Uploaded to HF: `treadon/speech-dac-tokens-3cb`
- Pre-tokenized with `input_ids`, `input_ids_1cb`, `input_ids_2cb` columns

**Training runs (3cb):**
- **5090 cloud:** 3.64-4.23 s/step, reached loss 4.75 at step 17K (epoch 1.3)
- Generated tokens were mostly `c1_698` (silence code)
- Loss plateaued around 4.75 — never produced non-silent audio
- Killed after ~$20 of cloud compute

**Failure analysis:**
- 3 codebooks = 258 tokens/sec = very long sequences (avg 3,063 tokens)
- 4096 context window needed, which was slow and OOM-prone
- The model learned the format but defaulted to the most common code (silence)
- "Regression to the mean" — predicting the average token is the easiest way to minimize loss

---

## Phase 2: 1 Codebook Experiments

**Goal:** Simplify by using only codebook 1 (coarse structure). Faster training, shorter sequences.

**Why:** 1cb at 44kHz = 87 tokens/sec. Sequences 3x shorter than 3cb.

**Training runs (1cb, 44kHz):**

### 1cb on 5090 (batch 4)
- 1.62-1.66 s/step
- Loss dropped fast: 20 → 4.2 in 2000 steps
- At loss 3.7: generated diverse c1 codes (not all silence!)
- Decoded audio: "buzz" — not intelligible but not silence

### 1cb on MPS (max-tokens 512)
- 9.3 s/step, 28K samples filtered from 241K
- Reached loss 3.1 after 8500 steps (epoch ~5)
- Generated tokens had bursts of diverse codes mixed with silence/repetition
- Audio: still buzzing, occasional tone variation, no speech

**Why 1cb didn't work for intelligible speech:**
- Codebook 1 captures coarse spectral envelope but not enough detail for phoneme distinction
- Verified by reconstruction test: DAC 44kHz with 1cb produces unintelligible audio even from real speech codes
- The model was learning correctly — the codec just can't represent speech with 1 codebook at 44kHz

---

## Phase 3: 2 Codebook at 44kHz

**Training runs (2cb, 44kHz):**

### 2cb on 5090
- 4.16-4.23 s/step
- Loss: 20 → 5.5 in 1500 steps
- Killed to try GPT-2 instead

### 2cb on MPS
- 43 s/step — too slow (sequences avg 2,000 tokens)
- OOM'd repeatedly

**Why we stopped:** Too slow on MPS, and cloud GPU time was expensive for uncertain results.

---

## Phase 4: GPT-2 Experiment

**Goal:** Try a smaller model (124M params vs 600M) for faster iteration.

**What happened:**
- `vectorized_gather_kernel: index out of bounds` — crash on first step
- GPT-2 has max 1024 position embeddings, our sequences exceeded that
- The tokenizer was built on Qwen3's vocabulary, creating incompatible token IDs for GPT-2
- Would need per-model tokenizer and pre-tokenized dataset columns

**Abandoned:** Too much infrastructure work for an experiment.

---

## Phase 5: DAC 16kHz, 2 Codebooks (Current)

**Key insight:** We were using DAC 44kHz (designed for music) for speech. Speech only needs frequencies up to 8kHz. DAC 16kHz is optimized for this.

**Discovery process:**
- Tested DAC 16kHz reconstruction at 1, 2, 3, 4, 6, 8, 12 codebooks
- Created comparison WAVs from real LibriSpeech audio
- **2 codebooks at 16kHz is intelligible** (mechanical but words are clear)
- 1 codebook at 16kHz is not intelligible
- This matched our 1cb 44kHz experience

**DAC configurations compared:**

| DAC Model | Codebooks | Tokens/sec/cb | Notes |
|-----------|-----------|---------------|-------|
| 44kHz | 9 available | 87 | Designed for music |
| 24kHz | 32 available | 75 | General audio |
| 16kHz | 12 available | 50 | Speech-focused |

**Dataset:** `treadon/speech-dac-16khz-2cb`
- 132,479 samples, ~464 hours (LibriSpeech clean-100 + clean-360)
- DAC 16kHz, 2 codebooks, 100 tokens/sec
- Pre-tokenized, ready for training
- LibriSpeech is natively 16kHz — no resampling needed

**Training (16kHz 2cb, MPS, max-tokens 512):**
- ~10 s/step, ~8,500 samples after filtering
- First run (28K samples only): loss 5.87 → 5.39 in 7 epochs, plateauing
- Generated tokens: diverse codes, not stuck on silence like 44kHz was
- Audio: not a continuous buzz, varied patterns, but not intelligible yet
- Second run (132K samples): in progress

**What's different about 16kHz:**
- Model generates diverse codes instead of defaulting to silence
- `<|audio_end|>` tokens appear naturally (model learned when to stop)
- The codec entries are speech-focused, not wasting capacity on >8kHz content

---

## Infrastructure Learnings

### Cloud GPU Pain Points
- **Vast.ai/RunPod:** 32GB overlay filesystem fills up instantly. Must use `HF_HOME=/workspace/.cache` or `/etc/workspace/`. Wasted hours debugging disk issues.
- **Batch size:** Auto-scaling batch size based on GPU memory was slower, not faster. Batch 1 with grad_accum 16 was optimal.
- **HF upload callback:** Failed silently on cloud machines. Manual upload with `huggingface-hub` API worked.

### MPS (Apple Silicon) Training
- Works but slow (~10-40 s/step depending on sequence length)
- OOM kills are common and unpredictable
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` helps but doesn't prevent all OOMs
- No bf16 support — FP32 only, doubles memory usage
- The `pin_memory` warning is harmless, ignore it

### Dataset Management
- Pre-tokenizing and storing `input_ids` in the HF dataset saves minutes per run
- Adding `attention_mask` and `labels` columns avoids `.map()` which uses disk for temp files
- Multiple codebook columns (`input_ids_1cb`, `input_ids_2cb`) in one dataset works well
- Chunk-based encoding with resume is essential — processes crash frequently

### Training Stability
- Checkpoints every 500 steps (not 2000) — crashes are frequent
- Auto-resume from latest checkpoint on restart
- Interactive training wrapper (`idle_train.py`) with P/C/Q/S controls
- `nohup` + PID file for background training
- SIGINT handler for graceful checkpoint saves

---

## Key Takeaways

1. **Codec choice matters more than model size.** DAC 44kHz for speech was the wrong choice from the start. DAC 16kHz produces intelligible 2cb speech; 44kHz doesn't even with 3cb.

2. **Regression to the mean is real.** The model learns to predict the most common token (silence) because that minimizes average loss. It takes many epochs to learn conditional distributions.

3. **More codebooks = harder task, not just better quality.** 3cb has 3x the tokens per frame, 3x the sequence length, and an exponentially harder prediction problem. Start with fewer codebooks.

4. **Small dataset + many epochs > large dataset + few epochs** for initial experiments. 28K samples at 10 epochs trains in hours; 241K at 3 epochs takes days.

5. **Data prep is half the work.** Encoding, resampling, tokenizing, uploading, column management — easily 50% of total project time.

6. **Cloud GPU infrastructure is surprisingly bad.** Overlay filesystems, missing drivers, pip install times, disk issues. A local Mac with MPS is slower but more reliable for experimentation.

---

## Phase 6: Abandoning the LLM + Discrete Tokens Approach

**Realization:** After researching how small TTS models actually work (Kokoro 82M, VITS 36M, Piper 5-30M, Matcha-TTS 18M), we discovered that **none of them use discrete audio tokens**. They all generate mel spectrograms or continuous representations.

**Why our approach was fundamentally flawed:**
1. **Discrete tokens lose information.** DAC codebook entries are a lossy compression. Even with perfect prediction, 1cb at 44kHz can't represent intelligible speech.
2. **Wrong model for the job.** Qwen3-0.6B has 600M params designed for language understanding. TTS-specific models achieve better results with 36M params because they're architecturally designed for the task.
3. **Autoregressive token prediction is slow and error-prone.** One wrong token and the audio degrades. Real TTS models generate entire utterances in parallel or use attention-based alignment.
4. **Mel spectrograms are free.** We spent days encoding DAC tokens. Mel spectrograms are a simple math transform — one line of code, milliseconds to compute, no GPU needed.

**What actually works for small TTS:**
- **VITS (36M):** VAE + normalizing flow + HiFi-GAN, end-to-end, works with 24h of data
- **Tacotron 2 (13M) + HiFi-GAN (14M):** Simplest architecture that produces good speech. Encoder-attention-decoder on mel spectrograms.
- **Kokoro (82M):** Best quality but complex (two-stage training, needs pre-trained WavLM and PL-BERT)

**Decision:** Pivot to Tacotron 2 + pre-trained HiFi-GAN. Simplest possible architecture:
- Train only Tacotron 2 (~13M params) to predict mel spectrograms from text
- Use pre-trained HiFi-GAN to convert mels to audio
- No data pre-processing needed — mels computed on the fly from audio
- LibriSpeech audio + transcripts is all we need

---

## Phase 7: Tacotron 2 + HiFi-GAN (Current)

**Architecture:**
- **Tacotron 2** (encoder → attention → decoder): text → mel spectrogram
- **HiFi-GAN** (pre-trained, frozen): mel spectrogram → audio waveform

**Why Tacotron 2:**
- Simplest neural TTS that actually works
- ~13M params — trains fast
- Attention learns text-to-audio alignment automatically (no Whisper, no forced alignment)
- Well-documented, many implementations available
- Works with LJSpeech (24h) — our 464h is more than enough

**What we're training:** Only Tacotron 2. HiFi-GAN is downloaded pre-trained.

**Data:** LibriSpeech audio + text. Mel spectrograms computed on-the-fly during training. No encoding, no tokenizing, no HF dataset needed.
