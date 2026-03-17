"""Training loop for the Genetic Transformer."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from genetic_transformer.config import GeneticTransformerConfig
from genetic_transformer.model.transformer import GeneticTransformer
from genetic_transformer.tokenizer.vocab import PAD_IDX
from genetic_transformer.training.losses import LabelSmoothingCrossEntropy
from genetic_transformer.training.scheduler import WarmupScheduler


class Trainer:
    """Handles training, validation, and checkpointing."""

    def __init__(self, model: GeneticTransformer, config: GeneticTransformerConfig,
                 train_loader: DataLoader, val_loader: DataLoader, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        self.scheduler = WarmupScheduler(
            self.optimizer, config.d_model, config.warmup_steps
        )
        self.criterion = LabelSmoothingCrossEntropy(
            config.tgt_vocab_size, PAD_IDX, config.label_smoothing
        )

        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    def train_epoch(self) -> float:
        """Run a single training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            tgt_y = batch["tgt_y"].to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(src, tgt)

            # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len,)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                tgt_y.view(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            tgt_y = batch["tgt_y"].to(self.device)

            logits, _ = self.model(src, tgt)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                tgt_y.view(-1)
            )

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, verbose: bool = True) -> dict:
        """Full training loop over all epochs."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            current_lr = self.scheduler.get_last_lr()[0]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            if verbose:
                print(f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {current_lr:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, "best_model.pt")
                )

        return self.history

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
