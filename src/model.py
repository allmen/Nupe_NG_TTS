"""
Sequence-to-sequence LSTM model with Bahdanau attention for the Nupe TTS system.

Architecture overview
---------------------
Encoder
    Embedding → (optional) pre-net FC layers → Bidirectional LSTM
    Output: encoder hidden states of shape ``(B, T_in, 2*encoder_hidden)``.

BahdanauAttention
    Additive (content-based) attention that produces a context vector for each
    decoder step.

Decoder
    Unidirectional LSTM that consumes the previous mel frame (passed through a
    pre-net) concatenated with the attention context vector at every step.
    It emits:
    - One mel frame.
    - A stop-token logit.

PostNet
    Five-layer 1-D convolutional network that refines the mel spectrogram
    predicted by the decoder.

NupeTTS  (top-level module)
    Combines all of the above.  Forward pass accepts padded text indices and
    padded target mel spectrograms (teacher-forcing during training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Bidirectional LSTM encoder.

    Parameters
    ----------
    vocab_size:
        Size of the input vocabulary (number of unique characters).
    embedding_dim:
        Dimensionality of character embeddings.
    encoder_hidden:
        Number of hidden units per LSTM direction.
    prenet_dims:
        Sequence of linear layer sizes applied to the embedding before the
        LSTM.  Defaults to a single 256-unit layer.
    dropout:
        Dropout probability applied to the pre-net and between LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        encoder_hidden: int = 256,
        prenet_dims: Tuple[int, ...] = (256,),
        dropout: float = 0.5,
        n_lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Pre-net: stack of FC + ReLU + Dropout
        layers = []
        in_dim = embedding_dim
        for out_dim in prenet_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.prenet = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=encoder_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        text:
            LongTensor ``(B, T_in)`` of character indices.
        text_lengths:
            LongTensor ``(B,)`` with the un-padded length of each sequence.
            When supplied, the LSTM uses ``pack_padded_sequence``.

        Returns
        -------
        torch.Tensor
            Encoder output ``(B, T_in, 2 * encoder_hidden)``.
        """
        x = self.embedding(text)            # (B, T_in, embedding_dim)
        x = self.prenet(x)                  # (B, T_in, prenet_dims[-1])

        if text_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        out, _ = self.lstm(x)
        if text_lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out  # (B, T_in, 2*encoder_hidden)


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism.

    Parameters
    ----------
    encoder_dim:
        Dimensionality of encoder output vectors (``2 * encoder_hidden``).
    decoder_dim:
        Dimensionality of the decoder hidden state.
    attention_dim:
        Size of the internal attention projection.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and attention weights.

        Parameters
        ----------
        encoder_out:
            ``(B, T_in, encoder_dim)``.
        decoder_hidden:
            ``(B, decoder_dim)`` – the previous decoder hidden state.
        mask:
            Boolean tensor ``(B, T_in)`` where ``True`` marks valid positions.
            Padding positions are set to ``-inf`` before softmax.

        Returns
        -------
        context:
            ``(B, encoder_dim)`` weighted sum of encoder outputs.
        attention_weights:
            ``(B, T_in)`` attention distribution.
        """
        # (B, T_in, attention_dim)
        energy = torch.tanh(
            self.W_encoder(encoder_out) + self.W_decoder(decoder_hidden).unsqueeze(1)
        )
        # (B, T_in)
        scores = self.v(energy).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)  # (B, T_in)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_out).squeeze(1)
        return context, attention_weights


class DecoderPreNet(nn.Module):
    """Two-layer fully-connected pre-net applied to the previous mel frame."""

    def __init__(self, n_mels: int, prenet_dim: int = 256, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mels, prenet_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Autoregressive LSTM decoder with Bahdanau attention.

    Parameters
    ----------
    n_mels:
        Number of mel frequency bins.
    encoder_dim:
        Dimensionality of encoder output (``2 * encoder_hidden``).
    decoder_hidden:
        Number of LSTM hidden units.
    prenet_dim:
        Width of the decoder pre-net.
    attention_dim:
        Width of the attention projection layer.
    dropout:
        Dropout applied inside the decoder.
    max_decoder_steps:
        Hard upper limit on the number of frames during inference.
    """

    def __init__(
        self,
        n_mels: int,
        encoder_dim: int,
        decoder_hidden: int = 512,
        prenet_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.1,
        max_decoder_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.decoder_hidden = decoder_hidden
        self.max_decoder_steps = max_decoder_steps

        self.prenet = DecoderPreNet(n_mels, prenet_dim, dropout)
        self.attention = BahdanauAttention(encoder_dim, decoder_hidden, attention_dim)

        # LSTM input: pre-net output + context vector
        self.lstm = nn.LSTMCell(prenet_dim + encoder_dim, decoder_hidden)
        self.dropout = nn.Dropout(dropout)

        # Projection layers
        self.mel_linear = nn.Linear(decoder_hidden + encoder_dim, n_mels)
        self.stop_linear = nn.Linear(decoder_hidden + encoder_dim, 1)

    def _init_state(
        self, encoder_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise LSTM hidden and cell states to zero."""
        B = encoder_out.size(0)
        device = encoder_out.device
        h = torch.zeros(B, self.decoder_hidden, device=device)
        c = torch.zeros(B, self.decoder_hidden, device=device)
        return h, c

    def _decode_step(
        self,
        prev_mel: torch.Tensor,
        encoder_out: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoder step.

        Returns mel_out, stop_logit, h, c, attn_weights.
        """
        prenet_out = self.prenet(prev_mel)              # (B, prenet_dim)
        context, attn_w = self.attention(encoder_out, h, mask)
        lstm_in = torch.cat([prenet_out, context], dim=-1)
        h, c = self.lstm(lstm_in, (h, c))
        h = self.dropout(h)
        out = torch.cat([h, context], dim=-1)           # (B, hidden + enc_dim)
        mel_out = self.mel_linear(out)                  # (B, n_mels)
        stop_logit = self.stop_linear(out).squeeze(-1)  # (B,)
        return mel_out, stop_logit, h, c, attn_w

    def forward(
        self,
        encoder_out: torch.Tensor,
        mel_targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a full sequence (teacher-forcing if *mel_targets* given).

        Parameters
        ----------
        encoder_out:
            ``(B, T_in, encoder_dim)``.
        mel_targets:
            ``(B, T_out, n_mels)`` ground-truth mel frames for teacher forcing.
            When ``None`` inference mode is used.
        mask:
            ``(B, T_in)`` boolean mask (True = valid position).

        Returns
        -------
        mel_outputs: ``(B, T_out, n_mels)``
        stop_logits: ``(B, T_out)``
        attention_weights: ``(B, T_out, T_in)``
        """
        B = encoder_out.size(0)
        device = encoder_out.device
        h, c = self._init_state(encoder_out)

        # Start-of-sequence frame: zeros
        prev_mel = torch.zeros(B, self.n_mels, device=device)

        mel_outputs, stop_logits, attentions = [], [], []

        if mel_targets is not None:
            # Teacher-forcing: iterate over target length
            T_out = mel_targets.size(1)
            for t in range(T_out):
                mel_out, stop_logit, h, c, attn_w = self._decode_step(
                    prev_mel, encoder_out, h, c, mask
                )
                mel_outputs.append(mel_out)
                stop_logits.append(stop_logit)
                attentions.append(attn_w)
                prev_mel = mel_targets[:, t, :]  # teacher force
        else:
            # Inference: decode until stop token or max steps
            for _ in range(self.max_decoder_steps):
                mel_out, stop_logit, h, c, attn_w = self._decode_step(
                    prev_mel, encoder_out, h, c, mask
                )
                mel_outputs.append(mel_out)
                stop_logits.append(stop_logit)
                attentions.append(attn_w)
                prev_mel = mel_out
                if torch.sigmoid(stop_logit).mean().item() > 0.5:
                    break

        mel_outputs = torch.stack(mel_outputs, dim=1)   # (B, T_out, n_mels)
        stop_logits = torch.stack(stop_logits, dim=1)   # (B, T_out)
        attentions = torch.stack(attentions, dim=1)     # (B, T_out, T_in)
        return mel_outputs, stop_logits, attentions


class PostNet(nn.Module):
    """Five-layer 1-D convolutional post-net to refine mel predictions.

    Based on the Tacotron 2 post-net: five convolutional layers each followed
    by batch normalisation and a Tanh activation (except the last layer).

    Parameters
    ----------
    n_mels:
        Number of mel frequency bins.
    postnet_channels:
        Number of channels in the convolutional layers.
    postnet_kernel:
        Kernel size of the convolutional layers (must be odd).
    n_layers:
        Total number of convolutional layers.
    dropout:
        Dropout applied after each non-final layer.
    """

    def __init__(
        self,
        n_mels: int,
        postnet_channels: int = 512,
        postnet_kernel: int = 5,
        n_layers: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        padding = (postnet_kernel - 1) // 2
        layers = []
        for i in range(n_layers):
            in_ch = n_mels if i == 0 else postnet_channels
            out_ch = n_mels if i == n_layers - 1 else postnet_channels
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, postnet_kernel, padding=padding),
                    nn.BatchNorm1d(out_ch),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            ``(B, T, n_mels)`` mel spectrogram.

        Returns
        -------
        torch.Tensor
            Residual ``(B, T, n_mels)`` to be added to the decoder output.
        """
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = torch.tanh(x)
                x = self.dropout(x)
        return x.transpose(1, 2)  # (B, T, n_mels)


class NupeTTS(nn.Module):
    """Full Nupe TTS seq2seq model.

    Combines :class:`Encoder`, :class:`Decoder`, and :class:`PostNet`.

    Parameters
    ----------
    vocab_size:
        Vocabulary size (use ``text_processing.VOCAB_SIZE``).
    n_mels:
        Number of mel frequency bins.
    embedding_dim:
        Character embedding size.
    encoder_hidden:
        Per-direction hidden units in the encoder BiLSTM.
    decoder_hidden:
        Hidden units in the decoder LSTM.
    prenet_dim:
        Decoder pre-net width.
    attention_dim:
        Attention projection width.
    postnet_channels:
        Post-net convolution channels.
    postnet_kernel:
        Post-net convolution kernel size.
    dropout:
        Dropout probability used throughout the model.
    max_decoder_steps:
        Maximum number of output frames during inference.
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        embedding_dim: int = 256,
        encoder_hidden: int = 256,
        decoder_hidden: int = 512,
        prenet_dim: int = 256,
        attention_dim: int = 128,
        postnet_channels: int = 512,
        postnet_kernel: int = 5,
        dropout: float = 0.5,
        max_decoder_steps: int = 1000,
    ) -> None:
        super().__init__()
        encoder_dim = 2 * encoder_hidden
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            encoder_hidden=encoder_hidden,
            dropout=dropout,
        )
        self.decoder = Decoder(
            n_mels=n_mels,
            encoder_dim=encoder_dim,
            decoder_hidden=decoder_hidden,
            prenet_dim=prenet_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            max_decoder_steps=max_decoder_steps,
        )
        self.postnet = PostNet(
            n_mels=n_mels,
            postnet_channels=postnet_channels,
            postnet_kernel=postnet_kernel,
            dropout=dropout,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        mel_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        text:
            ``(B, T_in)`` LongTensor of character indices.
        text_lengths:
            ``(B,)`` LongTensor of unpadded text lengths.
        mel_targets:
            ``(B, T_out, n_mels)`` target mel spectrogram (training only).

        Returns
        -------
        mel_outputs:
            ``(B, T_out, n_mels)`` decoder mel outputs (before post-net).
        mel_outputs_postnet:
            ``(B, T_out, n_mels)`` refined mel outputs (after post-net).
        stop_logits:
            ``(B, T_out)`` stop token logits.
        attention_weights:
            ``(B, T_out, T_in)`` attention alignment.
        """
        # Build encoder padding mask (True = valid)
        mask: Optional[torch.Tensor] = None
        if text_lengths is not None:
            B, T_in = text.size()
            mask = (
                torch.arange(T_in, device=text.device).unsqueeze(0)
                < text_lengths.unsqueeze(1)
            )  # (B, T_in)

        encoder_out = self.encoder(text, text_lengths)
        mel_outputs, stop_logits, attention_weights = self.decoder(
            encoder_out, mel_targets, mask
        )
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        return mel_outputs, mel_outputs_postnet, stop_logits, attention_weights
