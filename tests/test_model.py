"""
Unit tests for src.model (forward-pass shapes and output consistency).

These tests use small hyper-parameters and CPU-only tensors so that they run
quickly without requiring a GPU or training data.
"""

import pytest
import torch

from src.model import Encoder, BahdanauAttention, Decoder, PostNet, NupeTTS


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50
N_MELS = 16
ENCODER_HIDDEN = 32
DECODER_HIDDEN = 64
EMBEDDING_DIM = 32
PRENET_DIM = 32
ATTENTION_DIM = 16
BATCH = 2
T_IN = 10
T_OUT = 8


@pytest.fixture
def encoder():
    return Encoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        encoder_hidden=ENCODER_HIDDEN,
        dropout=0.0,
    )


@pytest.fixture
def text_batch():
    return torch.randint(1, VOCAB_SIZE, (BATCH, T_IN))


@pytest.fixture
def encoder_output(encoder, text_batch):
    with torch.no_grad():
        return encoder(text_batch)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TestEncoder:
    def test_output_shape(self, encoder, text_batch):
        with torch.no_grad():
            out = encoder(text_batch)
        assert out.shape == (BATCH, T_IN, 2 * ENCODER_HIDDEN)

    def test_with_lengths(self, encoder, text_batch):
        lengths = torch.tensor([T_IN, T_IN - 2])
        with torch.no_grad():
            out = encoder(text_batch, text_lengths=lengths)
        assert out.shape == (BATCH, T_IN, 2 * ENCODER_HIDDEN)

    def test_output_dtype(self, encoder, text_batch):
        with torch.no_grad():
            out = encoder(text_batch)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# BahdanauAttention
# ---------------------------------------------------------------------------

class TestBahdanauAttention:
    def test_context_and_weights_shapes(self, encoder_output):
        enc_dim = 2 * ENCODER_HIDDEN
        attn = BahdanauAttention(enc_dim, DECODER_HIDDEN, ATTENTION_DIM)
        h = torch.randn(BATCH, DECODER_HIDDEN)
        with torch.no_grad():
            context, weights = attn(encoder_output, h)
        assert context.shape == (BATCH, enc_dim)
        assert weights.shape == (BATCH, T_IN)

    def test_weights_sum_to_one(self, encoder_output):
        enc_dim = 2 * ENCODER_HIDDEN
        attn = BahdanauAttention(enc_dim, DECODER_HIDDEN, ATTENTION_DIM)
        h = torch.randn(BATCH, DECODER_HIDDEN)
        with torch.no_grad():
            _, weights = attn(encoder_output, h)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_mask_applied(self, encoder_output):
        enc_dim = 2 * ENCODER_HIDDEN
        attn = BahdanauAttention(enc_dim, DECODER_HIDDEN, ATTENTION_DIM)
        h = torch.randn(BATCH, DECODER_HIDDEN)
        # Mask out the last position for all batch items
        mask = torch.ones(BATCH, T_IN, dtype=torch.bool)
        mask[:, -1] = False
        with torch.no_grad():
            _, weights = attn(encoder_output, h, mask=mask)
        # Masked position should have (near) zero weight
        assert weights[:, -1].abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TestDecoder:
    def _make_decoder(self):
        return Decoder(
            n_mels=N_MELS,
            encoder_dim=2 * ENCODER_HIDDEN,
            decoder_hidden=DECODER_HIDDEN,
            prenet_dim=PRENET_DIM,
            attention_dim=ATTENTION_DIM,
            dropout=0.0,
            max_decoder_steps=20,
        )

    def test_teacher_forced_shapes(self, encoder_output):
        decoder = self._make_decoder()
        mel_targets = torch.randn(BATCH, T_OUT, N_MELS)
        with torch.no_grad():
            mel_out, stop_logits, attns = decoder(encoder_output, mel_targets)
        assert mel_out.shape == (BATCH, T_OUT, N_MELS)
        assert stop_logits.shape == (BATCH, T_OUT)
        assert attns.shape == (BATCH, T_OUT, T_IN)

    def test_inference_shapes(self, encoder_output):
        decoder = self._make_decoder()
        with torch.no_grad():
            mel_out, stop_logits, attns = decoder(encoder_output, mel_targets=None)
        T = mel_out.shape[1]
        assert mel_out.shape == (BATCH, T, N_MELS)
        assert stop_logits.shape == (BATCH, T)
        assert attns.shape == (BATCH, T, T_IN)


# ---------------------------------------------------------------------------
# PostNet
# ---------------------------------------------------------------------------

class TestPostNet:
    def test_output_shape(self):
        postnet = PostNet(n_mels=N_MELS, postnet_channels=32, n_layers=3, dropout=0.0)
        x = torch.randn(BATCH, T_OUT, N_MELS)
        with torch.no_grad():
            out = postnet(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# NupeTTS end-to-end
# ---------------------------------------------------------------------------

class TestNupeTTS:
    def _make_model(self):
        return NupeTTS(
            vocab_size=VOCAB_SIZE,
            n_mels=N_MELS,
            embedding_dim=EMBEDDING_DIM,
            encoder_hidden=ENCODER_HIDDEN,
            decoder_hidden=DECODER_HIDDEN,
            prenet_dim=PRENET_DIM,
            attention_dim=ATTENTION_DIM,
            postnet_channels=32,
            dropout=0.0,
            max_decoder_steps=20,
        )

    def test_training_forward(self):
        model = self._make_model()
        text = torch.randint(1, VOCAB_SIZE, (BATCH, T_IN))
        mel_targets = torch.randn(BATCH, T_OUT, N_MELS)
        with torch.no_grad():
            mel_out, mel_postnet, stop_logits, attns = model(
                text, mel_targets=mel_targets
            )
        assert mel_out.shape == (BATCH, T_OUT, N_MELS)
        assert mel_postnet.shape == (BATCH, T_OUT, N_MELS)
        assert stop_logits.shape == (BATCH, T_OUT)
        assert attns.shape == (BATCH, T_OUT, T_IN)

    def test_inference_forward(self):
        model = self._make_model()
        model.eval()
        text = torch.randint(1, VOCAB_SIZE, (1, T_IN))
        with torch.no_grad():
            mel_out, mel_postnet, stop_logits, attns = model(text)
        T = mel_out.shape[1]
        assert mel_out.shape == (1, T, N_MELS)
        assert mel_postnet.shape == (1, T, N_MELS)

    def test_with_text_lengths(self):
        model = self._make_model()
        text = torch.randint(1, VOCAB_SIZE, (BATCH, T_IN))
        text_lengths = torch.tensor([T_IN, T_IN - 2])
        mel_targets = torch.randn(BATCH, T_OUT, N_MELS)
        with torch.no_grad():
            mel_out, mel_postnet, stop_logits, attns = model(
                text, text_lengths=text_lengths, mel_targets=mel_targets
            )
        assert mel_out.shape == (BATCH, T_OUT, N_MELS)

    def test_postnet_adds_residual(self):
        model = self._make_model()
        text = torch.randint(1, VOCAB_SIZE, (1, T_IN))
        mel_targets = torch.randn(1, T_OUT, N_MELS)
        with torch.no_grad():
            mel_out, mel_postnet, _, _ = model(text, mel_targets=mel_targets)
        # mel_postnet should differ from mel_out (residual added)
        assert not torch.allclose(mel_out, mel_postnet)
