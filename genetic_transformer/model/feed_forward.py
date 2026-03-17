"""Position-wise feed-forward network from 'Attention is All You Need'."""

import torch.nn as nn
from torch import Tensor


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network.

    FFN(x) = ReLU(xW1 + b1)W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply FFN to each position independently.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
