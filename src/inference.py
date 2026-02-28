"""
Inference script for the Nupe TTS system.

Given a trained model checkpoint and an input text string, this script
synthesises speech and saves the output as a WAV file.

Usage example
-------------
.. code-block:: bash

    python -m src.inference \\
        --checkpoint checkpoints/nupe_tts_best.pt \\
        --text       "emi duku" \\
        --output     output.wav
"""

import argparse
import sys

import torch
import numpy as np

from .model import NupeTTS
from .text_processing import encode, VOCAB_SIZE
from .audio_utils import mel_to_wav, save_wav, SAMPLE_RATE, N_MELS


def load_model(
    checkpoint_path: str,
    device: torch.device,
    n_mels: int = N_MELS,
    embedding_dim: int = 256,
    encoder_hidden: int = 256,
    decoder_hidden: int = 512,
    dropout: float = 0.5,
    max_decoder_steps: int = 1000,
) -> NupeTTS:
    """Load a :class:`NupeTTS` model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` file produced by :mod:`src.train`.
    device:
        Target device.
    n_mels, embedding_dim, encoder_hidden, decoder_hidden, dropout,
    max_decoder_steps:
        Model hyper-parameters – must match those used during training.

    Returns
    -------
    NupeTTS
        Model in evaluation mode.
    """
    model = NupeTTS(
        vocab_size=VOCAB_SIZE,
        n_mels=n_mels,
        embedding_dim=embedding_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        dropout=dropout,
        max_decoder_steps=max_decoder_steps,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state-dicts and training checkpoints.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def synthesize(
    model: NupeTTS,
    text: str,
    device: torch.device,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert a Nupe text string to a waveform.

    Parameters
    ----------
    model:
        Loaded :class:`NupeTTS` model (evaluation mode).
    text:
        Input Nupe text.
    device:
        Device the model lives on.
    sample_rate:
        Audio sample rate used when converting the mel to a waveform.

    Returns
    -------
    np.ndarray
        1-D float32 waveform.
    """
    indices = encode(text)
    text_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    text_lengths = torch.tensor([len(indices)], dtype=torch.long).to(device)

    with torch.no_grad():
        _, mel_postnet, _, _ = model(text_tensor, text_lengths, mel_targets=None)

    # mel_postnet: (1, T_out, n_mels) → (n_mels, T_out)
    mel_np = mel_postnet.squeeze(0).cpu().numpy().T
    wav = mel_to_wav(mel_np, sample_rate=sample_rate)
    return wav


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesise Nupe speech with TTS.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--text", required=True, help="Input Nupe text")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--n_mels", type=int, default=N_MELS)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--encoder_hidden", type=int, default=256)
    parser.add_argument("--decoder_hidden", type=int, default=512)
    parser.add_argument("--max_decoder_steps", type=int, default=1000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(
        args.checkpoint,
        device=device,
        n_mels=args.n_mels,
        embedding_dim=args.embedding_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        max_decoder_steps=args.max_decoder_steps,
    )

    wav = synthesize(model, args.text, device, sample_rate=args.sample_rate)
    save_wav(args.output, wav, sample_rate=args.sample_rate)
    print(f"Saved synthesised audio to {args.output}")
