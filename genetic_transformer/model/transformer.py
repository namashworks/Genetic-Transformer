"""Full Genetic Transformer model combining encoder and decoder."""

import torch
import torch.nn as nn
from torch import Tensor

from ..config import GeneticTransformerConfig
from .decoder import Decoder
from .encoder import Encoder

PAD_IDX = 0


class GeneticTransformer(nn.Module):
    """Full Transformer model for genetic sequence tasks.

    Uses shared embedding when src_vocab_size == tgt_vocab_size (unified vocab).
    """

    def __init__(self, config: GeneticTransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Shared embedding if vocabularies match (unified vocab)
        shared = config.src_vocab_size == config.tgt_vocab_size
        if shared:
            shared_embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        else:
            shared_embedding = None

        self.encoder = Encoder(
            vocab_size=config.src_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_encoder_layers,
            max_len=config.max_seq_len,
            dropout=config.dropout,
            embedding=shared_embedding,
        )

        self.decoder = Decoder(
            vocab_size=config.tgt_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_decoder_layers,
            max_len=config.max_seq_len,
            dropout=config.dropout,
            embedding=shared_embedding if shared else None,
        )

    def make_src_mask(self, src: Tensor) -> Tensor:
        """Create source padding mask.

        Args:
            src: (batch, src_len)

        Returns:
            mask: (batch, 1, 1, src_len) -- True where not padded
        """
        # (batch, 1, 1, src_len)
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: Tensor) -> Tensor:
        """Create target mask combining padding mask and causal (look-ahead) mask.

        Args:
            tgt: (batch, tgt_len)

        Returns:
            mask: (batch, 1, tgt_len, tgt_len) -- True where attention is allowed
        """
        batch_size, tgt_len = tgt.size()

        # Padding mask: (batch, 1, 1, tgt_len)
        pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)

        # Causal mask: lower triangular (1, 1, tgt_len, tgt_len)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Combine: (batch, 1, tgt_len, tgt_len)
        return pad_mask & causal_mask

    def forward(self, src: Tensor, tgt: Tensor) -> tuple[Tensor, dict]:
        """Full forward pass through encoder-decoder Transformer.

        Args:
            src: (batch, src_len) source token indices
            tgt: (batch, tgt_len) target token indices

        Returns:
            logits: (batch, tgt_len, vocab_size)
            attention_maps: dict with encoder and decoder attention weights
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out, enc_attn_weights = self.encoder(src, src_mask)
        logits, dec_attn_maps = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        attention_maps = {
            "encoder_self_attn": enc_attn_weights,
            "decoder_self_attn": dec_attn_maps["self_attn"],
            "decoder_cross_attn": dec_attn_maps["cross_attn"],
        }

        return logits, attention_maps

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
