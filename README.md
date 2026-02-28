# Nupe_NG_TTS

A neural text-to-speech (TTS) system for the **Nupe language** (Nupencin),
spoken in Niger State, Nigeria.  The system is built around a
**sequence-to-sequence LSTM model with Bahdanau (additive) attention**,
inspired by the Tacotron 2 architecture.

---

## Architecture

```
Input text
    ↓ Character embedding + Pre-net FC layers
Encoder  – Bidirectional LSTM
    ↓ Encoder hidden states
BahdanauAttention  – additive attention at every decoder step
    ↓ Context vector
Decoder  – Unidirectional LSTM (teacher-forced during training)
    ↓ Mel spectrogram frames + stop-token logit
PostNet  – 5-layer 1-D CNN residual refinement
    ↓
Log mel spectrogram  →  Griffin-Lim vocoder  →  WAV
```

| Component | Details |
|-----------|---------|
| Encoder | Bidirectional LSTM (256 units/direction) |
| Attention | Bahdanau additive attention (128-dim projection) |
| Decoder | Unidirectional LSTM (512 units) |
| Post-net | 5 × Conv1D (512 ch, kernel 5) + BatchNorm |
| Acoustic features | 80-bin log mel spectrogram, 22 050 Hz, 256-sample hop |
| Vocoder | Griffin-Lim (60 iterations) |

---

## Repository layout

```
Nupe_NG_TTS/
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── text_processing.py   # Nupe character set, normalisation, encode/decode
│   ├── audio_utils.py       # Mel spectrogram extraction and Griffin-Lim
│   ├── model.py             # Encoder, Attention, Decoder, PostNet, NupeTTS
│   ├── dataset.py           # PyTorch Dataset + DataLoader utilities
│   ├── train.py             # Training script
│   └── inference.py         # Inference / synthesis script
└── tests/
    ├── test_text_processing.py
    ├── test_model.py
    └── test_audio_utils.py
```

---

## Installation

```bash
git clone https://github.com/allmen/Nupe_NG_TTS.git
cd Nupe_NG_TTS
pip install -r requirements.txt
```

---

## Data preparation

The dataset loader expects the
[LJSpeech-style](https://keithito.com/LJ-Speech-Dataset/) metadata format:

```
utterance_id|raw_text|normalized_text
```

Example:

```
nupe_0001|Emi duku.|emi duku.
nupe_0002|A wuci kó.|a wuci kó.
```

Place WAV files in a `wavs/` directory next to `metadata.csv`:

```
data/
    metadata.csv
    wavs/
        nupe_0001.wav
        nupe_0002.wav
```

---

## Training

```bash
python -m src.train \
    --metadata  data/metadata.csv \
    --wavs_dir  data/wavs \
    --output    checkpoints/ \
    --epochs    200 \
    --batch_size 16 \
    --lr        1e-3
```

Checkpoints are saved as `checkpoints/nupe_tts_epoch_N.pt` after every epoch.
The best checkpoint (lowest training loss) is saved as `checkpoints/nupe_tts_best.pt`.

Run `python -m src.train --help` for a full list of options.

---

## Inference

```bash
python -m src.inference \
    --checkpoint checkpoints/nupe_tts_best.pt \
    --text       "emi duku" \
    --output     output.wav
```

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Nupe character set

The model operates at the **character level**.  The vocabulary covers:

- All 26 lower-case Latin letters (`a`–`z`).
- Nupe-specific vowels: `ɛ`, `ɔ`, `ə`, and the labio-velar stop `ɡ`.
- Tonal variants with acute (´) and grave (`) diacritics, and long vowels
  with macron (ā, ē, …).
- Common punctuation: space, apostrophe, comma, hyphen, full-stop.
- Unknown characters are mapped to a special `<unk>` token.
