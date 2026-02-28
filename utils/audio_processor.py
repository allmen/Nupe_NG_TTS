import librosa
import numpy as np
import soundfile as sf
import torch
import os
from scipy import signal
import yaml

class AudioProcessor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        audio_config = config['audio']
        self.sr = audio_config['sample_rate']
        self.n_fft = audio_config['n_fft']
        self.hop_length = audio_config['hop_length']
        self.win_length = audio_config['win_length']
        self.n_mels = audio_config['n_mels']
        self.fmin = audio_config['fmin']
        self.fmax = audio_config['fmax']
        self.preemphasis = audio_config['preemphasis']
        self.max_db = audio_config['max_db']
        self.ref_db = audio_config['ref_db']
        self.top_db = audio_config['top_db']
        
    def load_wav(self, path):
        """Load audio file"""
        audio, sr = librosa.load(path, sr=self.sr)
        audio = self.preemphasis_wave(audio)
        return audio
    
    def preemphasis_wave(self, x):
        """Apply pre-emphasis to audio signal"""
        return signal.lfilter([1, -self.preemphasis], [1], x)
    
    def deemphasis_wave(self, x):
        """Remove pre-emphasis from audio signal"""
        return signal.lfilter([1], [1, -self.preemphasis], x)
    
    def melspectrogram(self, y):
        """Extract mel-spectrogram from audio"""
        # Compute STFT
        D = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann'
        )
        
        # Convert to magnitude
        S = np.abs(D)
        
        # Convert to mel scale
        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        mel = np.dot(mel_basis, S)
        
        # Convert to decibels
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mel = np.clip(mel, self.ref_db - self.max_db, self.ref_db)
        
        # Normalize
        mel = (mel - self.ref_db + self.max_db) / self.max_db
        
        return mel.T  # (time, n_mels)
    
    def inv_melspectrogram(self, mel, griffin_iters=30):
        """Convert mel-spectrogram back to audio (Griffin-Lim)

        This version avoids deprecated `np.complex` by working with
        magnitude and separate phase (angles) arrays.
        """
        # Denormalize
        mel = mel * self.max_db - self.max_db + self.ref_db
        
        # Convert from db to amplitude
        mel = np.power(10.0, mel * 0.05)
        
        # Inverse mel basis
        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Approximate inverse using pseudo-inverse
        inv_mel_basis = np.linalg.pinv(mel_basis)
        S = np.maximum(0, np.dot(inv_mel_basis, mel.T))  # Clamp negatives to 0
        
        # Griffin-Lim algorithm (works with magnitude S)
        y = self._griffin_lim(S, iterations=griffin_iters)
        
        # Apply de-emphasis
        y = self.deemphasis_wave(y)
        
        return y
    
    def _griffin_lim(self, S_mag, iterations=60):
        """Griffin-Lim phase reconstruction from magnitude spectrogram.

        S_mag: magnitude spectrogram (non-negative real array)
        """
        # Initialize random phase
        angles = np.exp(2j * np.pi * np.random.rand(*S_mag.shape))
        
        for _ in range(iterations):
            # Reconstruct complex spectrogram with current phase
            S_complex = S_mag * angles
            # Inverse STFT -> time signal
            full = librosa.istft(S_complex,
                                hop_length=self.hop_length,
                                win_length=self.win_length)
            # STFT of the time signal -> complex spectrogram
            est = librosa.stft(full,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length)
            # Update phase
            angles = est / (np.abs(est) + 1e-8)
        
        # Final reconstruction
        y = librosa.istft(S_mag * angles,
                          hop_length=self.hop_length,
                          win_length=self.win_length)
        return y
    
    def extract_features(self, audio_path):
        """Extract all features from audio file"""
        audio = self.load_wav(audio_path)
        mel = self.melspectrogram(audio)
        
        # Extract pitch using PyWorld (optional)
        try:
            import pyworld as pw
            _f0 = pw.dio(audio.astype(np.float64), self.sr)
            # use stonemask or cheaptrick as needed; keep it optional
            f0 = _f0 if _f0 is not None else np.zeros(len(audio))
            pitch = f0
        except Exception:
            pitch = np.zeros(len(audio))
        
        # Extract energy (scalar RMS energy)
        energy = np.sqrt(np.mean(audio ** 2))
        
        features = {
            'mel': mel,
            'audio': audio,
            'pitch': pitch,
            'energy': energy
        }
        
        return features
    
    def save_audio(self, audio, path):
        """Save audio to file"""
        sf.write(path, audio, self.sr)
