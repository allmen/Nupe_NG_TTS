"""
PyTorch Dataset and DataLoader utilities for the Nupe TTS system.

Data format
-----------
The dataset expects a metadata file in the LJSpeech-style format::

    utterance_id|raw_text|normalized_text

where fields are separated by ``|``.  Only *utterance_id* and
*normalized_text* are used; *raw_text* is kept for reference.

Audio files are expected to reside in a ``wavs/`` sub-directory next to the
metadata file and to be named ``<utterance_id>.wav``.

Example metadata line::

    nupe_0001|Emi shi ma.|emi shi ma.

Directory layout::

    data_root/
        metadata.csv
        wavs/
            nupe_0001.wav
            nupe_0002.wav
            ...
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .text_processing import encode
from .audio_utils import (
    load_wav,
    wav_to_mel,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
)


class NupeTTSDataset(Dataset):
    """Dataset for the Nupe TTS system.

    Parameters
    ----------
    metadata_path:
        Path to the ``metadata.csv`` file.
    wavs_dir:
        Directory containing the WAV files.  Defaults to a ``wavs/``
        sub-directory next to *metadata_path*.
    sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max:
        Audio feature extraction parameters.
    """

    def __init__(
        self,
        metadata_path: str,
        wavs_dir: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
    ) -> None:
        self.metadata_path = metadata_path
        if wavs_dir is None:
            wavs_dir = os.path.join(os.path.dirname(metadata_path), "wavs")
        self.wavs_dir = wavs_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.items: List[Tuple[str, str]] = []  # (utterance_id, text)
        self._load_metadata()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        with open(self.metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                utterance_id = parts[0]
                # Use the normalised text field when available.
                text = parts[2] if len(parts) >= 3 else parts[1]
                self.items.append((utterance_id, text))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        utterance_id, text = self.items[index]
        wav_path = os.path.join(self.wavs_dir, f"{utterance_id}.wav")

        # Text → index sequence (includes EOS)
        text_indices = torch.tensor(encode(text), dtype=torch.long)

        # Audio → log mel spectrogram  shape: (n_mels, T)
        wav = load_wav(wav_path, self.sample_rate)
        mel = wav_to_mel(
            wav,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        # Transpose to (T, n_mels) for easier padding in collate_fn
        mel_tensor = torch.from_numpy(mel.T)

        return text_indices, mel_tensor


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length text and mel tensors.

    Parameters
    ----------
    batch:
        List of ``(text_indices, mel_tensor)`` tuples from
        :class:`NupeTTSDataset`.

    Returns
    -------
    text_padded:
        ``(B, T_text_max)`` LongTensor – zero-padded text sequences.
    text_lengths:
        ``(B,)`` LongTensor – original text lengths.
    mel_padded:
        ``(B, T_mel_max, n_mels)`` FloatTensor – zero-padded mel sequences.
    mel_lengths:
        ``(B,)`` LongTensor – original mel lengths.
    """
    text_seqs, mel_seqs = zip(*batch)

    text_lengths = torch.tensor([t.size(0) for t in text_seqs], dtype=torch.long)
    mel_lengths = torch.tensor([m.size(0) for m in mel_seqs], dtype=torch.long)

    T_text_max = int(text_lengths.max().item())
    T_mel_max = int(mel_lengths.max().item())
    n_mels = mel_seqs[0].size(1)
    B = len(batch)

    text_padded = torch.zeros(B, T_text_max, dtype=torch.long)
    mel_padded = torch.zeros(B, T_mel_max, n_mels, dtype=torch.float)

    for i, (t, m) in enumerate(zip(text_seqs, mel_seqs)):
        text_padded[i, : t.size(0)] = t
        mel_padded[i, : m.size(0)] = m

    return text_padded, text_lengths, mel_padded, mel_lengths


def build_dataloader(
    metadata_path: str,
    wavs_dir: Optional[str] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """Convenience factory for a :class:`NupeTTSDataset` DataLoader.

    Parameters
    ----------
    metadata_path:
        Path to the metadata CSV file.
    wavs_dir:
        Directory containing WAV files.
    batch_size, shuffle, num_workers:
        Standard DataLoader parameters.
    **dataset_kwargs:
        Extra keyword arguments forwarded to :class:`NupeTTSDataset`.

    Returns
    -------
    DataLoader
    """
    dataset = NupeTTSDataset(metadata_path, wavs_dir, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
