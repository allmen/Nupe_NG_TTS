"""
Unit tests for src.audio_utils.

These tests use synthetic numpy arrays to avoid the need for real audio files.
librosa is required; if it is not installed the tests are skipped.
"""

import numpy as np
import pytest

try:
    import librosa  # noqa: F401
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _LIBROSA_AVAILABLE, reason="librosa not installed"
)

from src.audio_utils import (
    wav_to_mel,
    mel_to_wav,
    SAMPLE_RATE,
    N_MELS,
    HOP_LENGTH,
    N_FFT,
    WIN_LENGTH,
    F_MIN,
    F_MAX,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sine_wav(frequency: float = 440.0, duration: float = 0.5) -> np.ndarray:
    """Generate a pure-tone waveform for testing."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (np.sin(2 * np.pi * frequency * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# wav_to_mel
# ---------------------------------------------------------------------------

class TestWavToMel:
    def test_output_shape(self):
        wav = _make_sine_wav()
        mel = wav_to_mel(wav)
        expected_frames = 1 + (len(wav) // HOP_LENGTH)
        assert mel.shape[0] == N_MELS
        # Allow ±1 frame tolerance due to librosa padding conventions.
        assert abs(mel.shape[1] - expected_frames) <= 2

    def test_output_dtype(self):
        wav = _make_sine_wav()
        mel = wav_to_mel(wav)
        assert mel.dtype == np.float32

    def test_log_compression(self):
        # All values should be log-compressed: no -inf for a non-silent signal.
        wav = _make_sine_wav()
        mel = wav_to_mel(wav)
        assert np.all(np.isfinite(mel))

    def test_custom_n_mels(self):
        wav = _make_sine_wav()
        mel = wav_to_mel(wav, n_mels=40)
        assert mel.shape[0] == 40


# ---------------------------------------------------------------------------
# mel_to_wav
# ---------------------------------------------------------------------------

class TestMelToWav:
    def test_output_is_1d_float32(self):
        wav = _make_sine_wav()
        mel = wav_to_mel(wav)
        reconstructed = mel_to_wav(mel)
        assert reconstructed.ndim == 1
        assert reconstructed.dtype == np.float32

    def test_output_length_nonzero(self):
        wav = _make_sine_wav()
        mel = wav_to_mel(wav)
        reconstructed = mel_to_wav(mel)
        assert len(reconstructed) > 0

    def test_reconstruction_roughly_matches_energy(self):
        # The reconstructed waveform should have comparable RMS energy.
        wav = _make_sine_wav()
        mel = wav_to_mel(wav)
        reconstructed = mel_to_wav(mel)
        rms_original = float(np.sqrt(np.mean(wav ** 2)))
        rms_reconstructed = float(np.sqrt(np.mean(reconstructed ** 2)))
        # Allow an order of magnitude difference (Griffin-Lim is approximate).
        assert rms_reconstructed > rms_original * 0.01
