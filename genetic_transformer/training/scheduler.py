"""Learning rate scheduler following the original Transformer paper."""

import torch


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + inverse square root decay from 'Attention is All You Need'.

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int,
                 last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        scale = self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )
        return [scale for _ in self.base_lrs]
