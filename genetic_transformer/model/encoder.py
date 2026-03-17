"""Transformer encoder from 'Attention is All You Need'."""

import math

import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attention -> add&norm -> FFN -> add&norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass through one encoder layer.

        Args:
            src: (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len)

        Returns:
            output: (batch, src_len, d_model)
            self_attn_weights: (batch, n_heads, src_len, src_len)
        """
        # Self-attention with residual connection and layer norm
        attn_out, self_attn_weights = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_out))

        # Feed-forward with residual connection and layer norm
        ff_out = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_out))

        return src, self_attn_weights


class Encoder(nn.Module):
    """Full encoder: Embedding + PositionalEncoding + N EncoderLayers."""

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
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self, src: Tensor, src_mask: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass through the full encoder stack.

        Args:
            src: (batch, src_len) token indices
            src_mask: (batch, 1, 1, src_len)

        Returns:
            encoder_output: (batch, src_len, d_model)
            attention_weights: list of (batch, n_heads, src_len, src_len), one per layer
        """
        # Embed and scale by sqrt(d_model)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        attention_weights = []
        for layer in self.layers:
            x, attn_w = layer(x, src_mask)
            attention_weights.append(attn_w)

        return x, attention_weights
