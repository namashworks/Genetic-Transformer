"""Transformer decoder from 'Attention is All You Need'."""

import math

import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    """Single decoder layer: masked-self-attn -> add&norm -> cross-attn -> add&norm -> FFN -> add&norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through one decoder layer.

        Args:
            tgt: (batch, tgt_len, d_model)
            enc_out: (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len)
            tgt_mask: (batch, 1, tgt_len, tgt_len)

        Returns:
            output: (batch, tgt_len, d_model)
            self_attn_weights: (batch, n_heads, tgt_len, tgt_len)
            cross_attn_weights: (batch, n_heads, tgt_len, src_len)
        """
        # Masked self-attention
        attn_out, self_attn_weights = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))

        # Cross-attention over encoder output
        attn_out, cross_attn_weights = self.cross_attn(tgt, enc_out, enc_out, src_mask)
        tgt = self.norm2(tgt + self.dropout2(attn_out))

        # Feed-forward
        ff_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_out))

        return tgt, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    """Full decoder: Embedding + PositionalEncoding + N DecoderLayers + linear output projection."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_len: int,
        dropout: float,
        embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len * 2, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Forward pass through the full decoder stack.

        Args:
            tgt: (batch, tgt_len) token indices
            enc_out: (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len)
            tgt_mask: (batch, 1, tgt_len, tgt_len)

        Returns:
            logits: (batch, tgt_len, vocab_size)
            attention_maps: {"self_attn": [...], "cross_attn": [...]}
        """
        # Embed and scale by sqrt(d_model)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        self_attn_weights_all = []
        cross_attn_weights_all = []

        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, enc_out, src_mask, tgt_mask)
            self_attn_weights_all.append(self_attn_w)
            cross_attn_weights_all.append(cross_attn_w)

        logits = self.output_projection(x)

        attention_maps = {
            "self_attn": self_attn_weights_all,
            "cross_attn": cross_attn_weights_all,
        }

        return logits, attention_maps
