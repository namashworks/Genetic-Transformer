"""Label-smoothed cross-entropy loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing, ignoring PAD tokens."""

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch * seq_len, vocab_size) logits
            target: (batch * seq_len,) ground truth indices
        """
        pred = pred.contiguous().view(-1, self.vocab_size)
        target = target.contiguous().view(-1)

        # Create smoothed distribution
        smooth_dist = torch.full_like(pred, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0

        # Mask out PAD positions
        pad_mask = target == self.pad_idx
        smooth_dist[pad_mask] = 0

        # KL divergence loss
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Average over non-PAD tokens
        non_pad = (~pad_mask).sum()
        if non_pad == 0:
            return loss.sum() * 0.0
        return loss.sum() / non_pad
