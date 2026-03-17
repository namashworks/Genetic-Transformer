"""Tests for genetic_transformer.model — attention, encoder, decoder, transformer."""

import pytest
import torch

from genetic_transformer.model.attention import MultiHeadAttention
from genetic_transformer.model.positional_encoding import PositionalEncoding
from genetic_transformer.model.feed_forward import PositionWiseFeedForward
from genetic_transformer.model.encoder import Encoder, EncoderLayer
from genetic_transformer.model.decoder import Decoder, DecoderLayer
from genetic_transformer.model.transformer import GeneticTransformer


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        q = k = v = torch.randn(2, 10, 32)
        out, attn_w = mha(q, k, v)
        assert out.shape == (2, 10, 32)
        assert attn_w.shape == (2, 2, 10, 10)

    def test_cross_attention_shape(self):
        mha = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        q = torch.randn(2, 8, 32)
        k = v = torch.randn(2, 12, 32)
        out, attn_w = mha(q, k, v)
        assert out.shape == (2, 8, 32)
        assert attn_w.shape == (2, 2, 8, 12)

    def test_attention_weights_sum_to_one(self):
        mha = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        q = k = v = torch.randn(1, 5, 32)
        _, attn_w = mha(q, k, v)
        # Each row should sum to ~1
        sums = attn_w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_masking_zeros_out_future(self):
        mha = MultiHeadAttention(d_model=32, n_heads=2, dropout=0.0)
        seq_len = 5
        q = k = v = torch.randn(1, seq_len, 32)
        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        _, attn_w = mha(q, k, v, mask=mask)
        # Position 0 should only attend to position 0
        # Upper triangle of attention should be ~0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attn_w[0, :, i, j].max().item() < 1e-5

    def test_d_model_not_divisible_by_heads_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=33, n_heads=2)


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.zeros(2, 10, 32)
        out = pe(x)
        assert out.shape == (2, 10, 32)

    def test_adds_nonzero_values(self):
        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        assert out.abs().sum() > 0

    def test_different_positions_get_different_encodings(self):
        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        # Position 0 and position 1 should differ
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# PositionWiseFeedForward
# ---------------------------------------------------------------------------

class TestFeedForward:
    def test_output_shape(self):
        ff = PositionWiseFeedForward(d_model=32, d_ff=64, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = ff(x)
        assert out.shape == (2, 10, 32)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TestEncoder:
    def test_output_shape(self):
        enc = Encoder(vocab_size=100, d_model=32, n_heads=2, d_ff=64,
                      n_layers=1, max_len=32, dropout=0.0)
        src = torch.randint(0, 100, (2, 10))
        out, attn_list = enc(src)
        assert out.shape == (2, 10, 32)
        assert len(attn_list) == 1
        assert attn_list[0].shape == (2, 2, 10, 10)

    def test_with_mask(self):
        enc = Encoder(vocab_size=100, d_model=32, n_heads=2, d_ff=64,
                      n_layers=2, max_len=32, dropout=0.0)
        src = torch.randint(0, 100, (2, 10))
        mask = (src != 0).unsqueeze(1).unsqueeze(2)
        out, attn_list = enc(src, mask)
        assert out.shape == (2, 10, 32)
        assert len(attn_list) == 2


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(vocab_size=100, d_model=32, n_heads=2, d_ff=64,
                      n_layers=1, max_len=32, dropout=0.0)
        tgt = torch.randint(0, 100, (2, 8))
        enc_out = torch.randn(2, 10, 32)
        logits, attn_maps = dec(tgt, enc_out)
        assert logits.shape == (2, 8, 100)
        assert "self_attn" in attn_maps
        assert "cross_attn" in attn_maps


# ---------------------------------------------------------------------------
# GeneticTransformer (full model)
# ---------------------------------------------------------------------------

class TestGeneticTransformer:
    def test_forward_pass_shape(self, model, small_config):
        batch = 2
        src_len = 10
        tgt_len = 8
        vocab = small_config.src_vocab_size
        src = torch.randint(1, vocab, (batch, src_len))
        tgt = torch.randint(1, vocab, (batch, tgt_len))
        logits, attn_maps = model(src, tgt)
        assert logits.shape == (batch, tgt_len, vocab)
        assert "encoder_self_attn" in attn_maps
        assert "decoder_self_attn" in attn_maps
        assert "decoder_cross_attn" in attn_maps

    def test_src_mask_shape(self, model):
        src = torch.tensor([[1, 2, 3, 0, 0]])
        mask = model.make_src_mask(src)
        assert mask.shape == (1, 1, 1, 5)
        # PAD positions should be False
        assert mask[0, 0, 0, 3].item() is False
        assert mask[0, 0, 0, 0].item() is True

    def test_tgt_mask_is_causal(self, model):
        tgt = torch.tensor([[1, 2, 3, 0]])
        mask = model.make_tgt_mask(tgt)
        assert mask.shape == (1, 1, 4, 4)
        # Position 0 can only see position 0
        assert mask[0, 0, 0, 1].item() is False
        # Position 2 can see positions 0, 1, 2
        assert mask[0, 0, 2, 0].item() is True
        assert mask[0, 0, 2, 1].item() is True
        assert mask[0, 0, 2, 2].item() is True
        # PAD position 3 should be masked
        assert mask[0, 0, 2, 3].item() is False

    def test_count_parameters(self, model):
        n_params = model.count_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_gradient_flow(self, model, small_config):
        """Ensure gradients flow through the entire model."""
        model.train()
        vocab = small_config.src_vocab_size
        src = torch.randint(1, vocab, (2, 8))
        tgt = torch.randint(1, vocab, (2, 6))
        logits, _ = model(src, tgt)
        loss = logits.sum()
        loss.backward()
        # Check that encoder embedding has gradients
        assert model.encoder.embedding.weight.grad is not None
        grad_norm = model.encoder.embedding.weight.grad.norm().item()
        assert grad_norm > 0
        model.zero_grad()

    def test_shared_embedding_when_same_vocab(self, small_config):
        """When src and tgt vocab sizes match, embedding should be shared."""
        m = GeneticTransformer(small_config)
        assert m.encoder.embedding is m.decoder.embedding

    def test_eval_mode_no_error(self, model, small_config):
        model.eval()
        vocab = small_config.src_vocab_size
        src = torch.randint(1, vocab, (1, 5))
        tgt = torch.randint(1, vocab, (1, 3))
        with torch.no_grad():
            logits, _ = model(src, tgt)
        assert logits.shape[0] == 1
