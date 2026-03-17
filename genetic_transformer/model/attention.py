"""Multi-head attention from 'Attention is All You Need'."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute multi-head attention.

        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)  -- seq_len_k == seq_len_v
            mask:  broadcastable to (batch, 1, seq_len_q, seq_len_k) or
                   (batch, 1, 1, seq_len_k)

        Returns:
            output: (batch, seq_len_q, d_model)
            attn_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, n_heads, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)
        # context: (batch, n_heads, seq_len_q, d_k)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output, attn_weights
