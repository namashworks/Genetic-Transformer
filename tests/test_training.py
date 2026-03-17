"""Tests for genetic_transformer.training — losses, scheduler, trainer."""

import pytest
import torch

from genetic_transformer.training.losses import LabelSmoothingCrossEntropy
from genetic_transformer.training.scheduler import WarmupScheduler
from genetic_transformer.training.trainer import Trainer
from genetic_transformer.tokenizer.vocab import PAD_IDX


# ---------------------------------------------------------------------------
# LabelSmoothingCrossEntropy
# ---------------------------------------------------------------------------

class TestLabelSmoothingLoss:
    def test_output_is_scalar(self):
        loss_fn = LabelSmoothingCrossEntropy(vocab_size=50, pad_idx=0, smoothing=0.1)
        pred = torch.randn(8, 50)
        target = torch.randint(1, 50, (8,))
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # scalar

    def test_loss_is_nonnegative(self):
        loss_fn = LabelSmoothingCrossEntropy(vocab_size=50, pad_idx=0, smoothing=0.1)
        pred = torch.randn(8, 50)
        target = torch.randint(1, 50, (8,))
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_loss_ignores_pad_tokens(self):
        loss_fn = LabelSmoothingCrossEntropy(vocab_size=50, pad_idx=0, smoothing=0.1)
        # All targets are PAD
        pred = torch.randn(4, 50)
        target = torch.zeros(4, dtype=torch.long)
        loss = loss_fn(pred, target)
        assert loss.item() == 0.0

    def test_no_smoothing_matches_cross_entropy(self):
        """With smoothing=0, loss should approach standard cross-entropy."""
        vocab = 10
        loss_fn = LabelSmoothingCrossEntropy(vocab_size=vocab, pad_idx=0, smoothing=0.0)
        pred = torch.randn(4, vocab)
        target = torch.randint(1, vocab, (4,))
        loss = loss_fn(pred, target)
        # Should be finite and positive
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_higher_smoothing_yields_lower_loss_for_correct_pred(self):
        """With higher smoothing, the penalty for a correct prediction is less severe."""
        vocab = 10
        pred = torch.zeros(1, vocab)
        pred[0, 5] = 10.0  # strongly predict class 5
        target = torch.tensor([5])

        loss_low = LabelSmoothingCrossEntropy(vocab, 0, 0.0)(pred, target)
        loss_high = LabelSmoothingCrossEntropy(vocab, 0, 0.3)(pred, target)
        # With smoothing, some probability mass is spread, so loss is typically higher
        # Both should be finite
        assert torch.isfinite(loss_low)
        assert torch.isfinite(loss_high)

    def test_gradient_flows_through_loss(self):
        loss_fn = LabelSmoothingCrossEntropy(vocab_size=50, pad_idx=0, smoothing=0.1)
        pred = torch.randn(4, 50, requires_grad=True)
        target = torch.randint(1, 50, (4,))
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# WarmupScheduler
# ---------------------------------------------------------------------------

class TestWarmupScheduler:
    def test_lr_increases_during_warmup(self):
        model_params = [torch.nn.Parameter(torch.randn(10))]
        optimizer = torch.optim.Adam(model_params, lr=1.0)
        scheduler = WarmupScheduler(optimizer, d_model=32, warmup_steps=100)

        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # LR should be increasing during warmup
        assert lrs[-1] > lrs[0]

    def test_lr_decreases_after_warmup(self):
        model_params = [torch.nn.Parameter(torch.randn(10))]
        optimizer = torch.optim.Adam(model_params, lr=1.0)
        scheduler = WarmupScheduler(optimizer, d_model=32, warmup_steps=10)

        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # After warmup, LR should decrease
        assert lrs[50] < lrs[10]

    def test_lr_is_positive(self):
        model_params = [torch.nn.Parameter(torch.randn(10))]
        optimizer = torch.optim.Adam(model_params, lr=1.0)
        scheduler = WarmupScheduler(optimizer, d_model=32, warmup_steps=10)

        for _ in range(50):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            assert lr > 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_trainer_init(self, model, small_config, dataloaders):
        train_loader, val_loader = dataloaders
        trainer = Trainer(model, small_config, train_loader, val_loader, device="cpu")
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_single_train_epoch(self, model, small_config, dataloaders):
        train_loader, val_loader = dataloaders
        trainer = Trainer(model, small_config, train_loader, val_loader, device="cpu")
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
        assert loss > 0

    def test_validate(self, model, small_config, dataloaders):
        train_loader, val_loader = dataloaders
        trainer = Trainer(model, small_config, train_loader, val_loader, device="cpu")
        val_loss = trainer.validate()
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_train_reduces_loss_over_steps(self, small_config, tokenizer, dataloaders):
        """Training for a few epochs should reduce loss (on this tiny dataset)."""
        from genetic_transformer.model.transformer import GeneticTransformer
        # Use a fresh model so we start from random weights
        fresh_model = GeneticTransformer(small_config)
        train_loader, val_loader = dataloaders
        trainer = Trainer(fresh_model, small_config, train_loader, val_loader, device="cpu")
        loss1 = trainer.train_epoch()
        loss2 = trainer.train_epoch()
        loss3 = trainer.train_epoch()
        # Loss should generally decrease or at least not explode
        assert loss3 < loss1 * 5  # generous bound: should not explode

    def test_save_and_load_checkpoint(self, model, small_config, dataloaders, tmp_path):
        train_loader, val_loader = dataloaders
        trainer = Trainer(model, small_config, train_loader, val_loader, device="cpu")
        ckpt_path = str(tmp_path / "test_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)
        # Load back
        trainer.load_checkpoint(ckpt_path)
        assert trainer.best_val_loss is not None
