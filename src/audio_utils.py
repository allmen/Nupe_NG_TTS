"""
Audio utilities for the Nupe TTS system.

Provides functions to:
- Load waveforms from audio files.
- Compute (log) mel spectrograms used as acoustic features.
- Convert mel spectrograms back to waveforms via the Griffin-Lim algorithm.
- Save waveforms to WAV files.
"""

import numpy as np
import soundfile as sf

try:
    import librosa
    import librosa.filters
    _LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIBROSA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 22050
N_FFT: int = 1024
HOP_LENGTH: int = 256
WIN_LENGTH: int = 1024
N_MELS: int = 80
F_MIN: float = 0.0
F_MAX: float = 8000.0
# Minimum value before taking the log; avoids log(0).
LOG_MEL_MIN: float = 1e-5
# Number of Griffin-Lim iterations for waveform reconstruction.
GRIFFIN_LIM_ITERS: int = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_librosa() -> None:
    if not _LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for audio processing. "
            "Install it with: pip install librosa"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_wav(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file and return a mono float32 waveform.

    Parameters
    ----------
    path:
        Path to the audio file (any format supported by soundfile / librosa).
    sample_rate:
        Target sample rate; the waveform is resampled if necessary.

    Returns
    -------
    np.ndarray
        1-D float32 array of waveform samples normalised to ``[-1, 1]``.
    """
    _assert_librosa()
    wav, sr = librosa.load(path, sr=sample_rate, mono=True)
    return wav.astype(np.float32)


def wav_to_mel(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
    log_mel_min: float = LOG_MEL_MIN,
) -> np.ndarray:
    """Convert a waveform to a log mel spectrogram.

    Parameters
    ----------
    wav:
        1-D float32 waveform array.
    sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max:
        Standard STFT / mel-filterbank parameters.
    log_mel_min:
        Floor value applied before taking the natural logarithm.

    Returns
    -------
    np.ndarray
        2-D float32 array of shape ``(n_mels, T)`` where *T* is the number of
        frames.
    """
    _assert_librosa()
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    log_mel = np.log(np.maximum(mel, log_mel_min))
    return log_mel.astype(np.float32)


def mel_to_wav(
    mel: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
    log_mel_min: float = LOG_MEL_MIN,
    n_iter: int = GRIFFIN_LIM_ITERS,
) -> np.ndarray:
    """Reconstruct a waveform from a log mel spectrogram via Griffin-Lim.

    Parameters
    ----------
    mel:
        2-D log mel spectrogram of shape ``(n_mels, T)``.
    n_iter:
        Number of Griffin-Lim iterations.
    Other parameters are the same as :func:`wav_to_mel`.

    Returns
    -------
    np.ndarray
        1-D float32 waveform array.
    """
    _assert_librosa()
    # Invert log compression.
    mel_linear = np.exp(mel)
    # Invert mel filterbank to obtain linear-scale spectrogram.
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    # Pseudo-inverse of mel filterbank.
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear = np.maximum(1e-10, inv_mel_basis @ mel_linear)
    wav = librosa.griffinlim(
        linear,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
    )
    return wav.astype(np.float32)


def save_wav(path: str, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a waveform to a WAV file.

    Parameters
    ----------
    path:
        Destination file path (must end with ``.wav``).
    wav:
        1-D float32 waveform array.
    sample_rate:
        Sample rate of the waveform.
    """
    sf.write(path, wav, samplerate=sample_rate, subtype="PCM_16")
