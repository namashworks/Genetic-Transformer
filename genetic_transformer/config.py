"""Configuration for the Genetic Transformer."""

from dataclasses import dataclass, field


@dataclass
class GeneticTransformerConfig:
    # Model architecture
    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256

    # Training
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    # Data generation
    num_train_samples: int = 50_000
    num_val_samples: int = 5_000
    seed: int = 42

    # Vocab sizes (set at runtime by tokenizer)
    src_vocab_size: int = 0
    tgt_vocab_size: int = 0

    # Paths
    checkpoint_dir: str = "checkpoints"
