"""
Training script for the Nupe TTS system.

Usage example
-------------
.. code-block:: bash

    python -m src.train \\
        --metadata  data/metadata.csv \\
        --wavs_dir  data/wavs \\
        --output    checkpoints/ \\
        --epochs    100 \\
        --batch_size 16

Checkpoints are saved as ``checkpoints/nupe_tts_epoch_{N}.pt`` at the end of
each epoch and whenever a new best validation loss is achieved.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataset import build_dataloader
from .model import NupeTTS
from .text_processing import VOCAB_SIZE


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    mel_outputs: torch.Tensor,
    mel_outputs_postnet: torch.Tensor,
    stop_logits: torch.Tensor,
    mel_targets: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute the combined mel + stop-token loss.

    Parameters
    ----------
    mel_outputs:
        ``(B, T_out, n_mels)`` decoder mel outputs (before post-net).
    mel_outputs_postnet:
        ``(B, T_out, n_mels)`` post-net mel outputs.
    stop_logits:
        ``(B, T_out)`` stop token logits.
    mel_targets:
        ``(B, T_out, n_mels)`` ground-truth mel frames.
    mel_lengths:
        ``(B,)`` ground-truth mel frame counts (used to build a loss mask).

    Returns
    -------
    torch.Tensor
        Scalar total loss.
    """
    B, T_out, _ = mel_targets.size()
    device = mel_targets.device

    # Mask padded positions
    mask = (
        torch.arange(T_out, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)
    ).float()  # (B, T_out)

    # MSE mel loss (decoder output)
    mel_loss = (
        nn.functional.mse_loss(mel_outputs, mel_targets, reduction="none")
        .mean(-1) * mask
    ).sum() / mask.sum()

    # MSE mel loss (post-net output)
    mel_postnet_loss = (
        nn.functional.mse_loss(mel_outputs_postnet, mel_targets, reduction="none")
        .mean(-1) * mask
    ).sum() / mask.sum()

    # BCE stop-token loss; ground truth: 0 except at the last valid frame
    stop_targets = torch.zeros(B, T_out, device=device)
    for i, length in enumerate(mel_lengths):
        stop_targets[i, length - 1] = 1.0

    stop_loss = (
        nn.functional.binary_cross_entropy_with_logits(
            stop_logits, stop_targets, reduction="none"
        ) * mask
    ).sum() / mask.sum()

    return mel_loss + mel_postnet_loss + stop_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader = build_dataloader(
        metadata_path=args.metadata,
        wavs_dir=args.wavs_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Model
    model = NupeTTS(
        vocab_size=VOCAB_SIZE,
        n_mels=args.n_mels,
        embedding_dim=args.embedding_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    os.makedirs(args.output, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (text, text_lengths, mel, mel_lengths) in enumerate(
            train_loader
        ):
            text = text.to(device)
            text_lengths = text_lengths.to(device)
            mel = mel.to(device)
            mel_lengths = mel_lengths.to(device)

            optimizer.zero_grad()
            mel_out, mel_postnet, stop_logits, _ = model(
                text, text_lengths, mel_targets=mel
            )
            loss = compute_loss(mel_out, mel_postnet, stop_logits, mel, mel_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % args.log_interval == 0:
                avg = epoch_loss / (batch_idx + 1)
                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg:.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch} complete – avg loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output, f"nupe_tts_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            ckpt_path,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output, "nupe_tts_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  → New best model saved to {best_path}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Nupe TTS model.")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--wavs_dir", default=None, help="Directory with WAV files")
    parser.add_argument("--output", default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--encoder_hidden", type=int, default=256)
    parser.add_argument("--decoder_hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args(sys.argv[1:]))
