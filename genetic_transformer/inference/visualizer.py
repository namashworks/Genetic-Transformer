"""Attention visualization utilities."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure


class AttentionVisualizer:
    """Matplotlib-based attention heatmap visualizer."""

    @staticmethod
    def plot_attention_heatmap(attention_weights: torch.Tensor, src_tokens: list[str],
                                tgt_tokens: list[str], title: str = "",
                                layer: int = 0, head: int = 0) -> Figure:
        """Plot a single attention head as a heatmap.

        Args:
            attention_weights: (n_heads, tgt_len, src_len) or (batch, n_heads, tgt_len, src_len)
            src_tokens: source token labels
            tgt_tokens: target token labels
            title: plot title
            layer: which layer (for title only)
            head: which head to visualize
        """
        if attention_weights.dim() == 4:
            attn = attention_weights[0, head].cpu().numpy()
        elif attention_weights.dim() == 3:
            attn = attention_weights[head].cpu().numpy()
        else:
            attn = attention_weights.cpu().numpy()

        # Trim to actual token lengths
        tgt_len = min(len(tgt_tokens), attn.shape[0])
        src_len = min(len(src_tokens), attn.shape[1])
        attn = attn[:tgt_len, :src_len]

        fig, ax = plt.subplots(figsize=(max(8, src_len * 0.6), max(6, tgt_len * 0.5)))
        cax = ax.matshow(attn, cmap="viridis", aspect="auto")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(src_len))
        ax.set_xticklabels(src_tokens[:src_len], rotation=45, ha="left", fontsize=8)
        ax.set_yticks(range(tgt_len))
        ax.set_yticklabels(tgt_tokens[:tgt_len], fontsize=8)

        ax.set_xlabel("Source Tokens")
        ax.set_ylabel("Target Tokens")
        if not title:
            title = f"Attention — Layer {layer}, Head {head}"
        ax.set_title(title, pad=20)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_multi_head_attention(attention_weights: torch.Tensor, src_tokens: list[str],
                                   tgt_tokens: list[str], layer: int = 0,
                                   n_heads: int = 4) -> Figure:
        """Plot all attention heads for one layer in a grid."""
        if attention_weights.dim() == 4:
            attn = attention_weights[0].cpu().numpy()  # Remove batch dim
        else:
            attn = attention_weights.cpu().numpy()

        n_heads = min(n_heads, attn.shape[0])
        cols = min(n_heads, 4)
        rows = (n_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        tgt_len = min(len(tgt_tokens), attn.shape[1])
        src_len = min(len(src_tokens), attn.shape[2])

        for h in range(n_heads):
            r, c = divmod(h, cols)
            head_attn = attn[h, :tgt_len, :src_len]
            axes[r, c].matshow(head_attn, cmap="viridis", aspect="auto")
            axes[r, c].set_title(f"Head {h}", fontsize=10)
            axes[r, c].set_xticks(range(src_len))
            axes[r, c].set_xticklabels(src_tokens[:src_len], rotation=45, ha="left", fontsize=6)
            axes[r, c].set_yticks(range(tgt_len))
            axes[r, c].set_yticklabels(tgt_tokens[:tgt_len], fontsize=6)

        # Hide unused axes
        for h in range(n_heads, rows * cols):
            r, c = divmod(h, cols)
            axes[r, c].axis("off")

        fig.suptitle(f"Multi-Head Attention — Layer {layer}", fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_cross_attention(attention_maps: dict, src_tokens: list[str],
                              tgt_tokens: list[str], layer: int = -1) -> Figure:
        """Plot cross-attention (decoder attending to encoder output).

        This is the most interpretable view for understanding
        which source tokens the model uses to generate each output token.
        """
        cross_attn_list = attention_maps.get("decoder_cross", attention_maps.get("decoder_cross_attn", []))
        if not cross_attn_list:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No cross-attention data available",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        # Use last layer by default
        if layer == -1:
            layer = len(cross_attn_list) - 1
        attn = cross_attn_list[layer]

        # Average across heads for an aggregate view
        if attn.dim() == 4:
            attn_avg = attn[0].mean(dim=0).cpu().numpy()  # (tgt_len, src_len)
        elif attn.dim() == 3:
            attn_avg = attn.mean(dim=0).cpu().numpy()
        else:
            attn_avg = attn.cpu().numpy()

        tgt_len = min(len(tgt_tokens), attn_avg.shape[0])
        src_len = min(len(src_tokens), attn_avg.shape[1])
        attn_avg = attn_avg[:tgt_len, :src_len]

        fig, ax = plt.subplots(figsize=(max(8, src_len * 0.7), max(6, tgt_len * 0.5)))
        cax = ax.matshow(attn_avg, cmap="magma", aspect="auto")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(src_len))
        ax.set_xticklabels(src_tokens[:src_len], rotation=45, ha="left", fontsize=8)
        ax.set_yticks(range(tgt_len))
        ax.set_yticklabels(tgt_tokens[:tgt_len], fontsize=8)

        ax.set_xlabel("Source (Input) Tokens")
        ax.set_ylabel("Target (Output) Tokens")
        ax.set_title(f"Cross-Attention (Layer {layer}, averaged over heads)", pad=20)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_training_curves(history: dict) -> Figure:
        """Plot training and validation loss curves, plus learning rate."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss curves
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Learning rate
        ax2.plot(epochs, history["lr"], "g-", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
