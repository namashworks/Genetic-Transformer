"""Shared fixtures for the Genetic Transformer test suite."""

import pytest
import torch

from genetic_transformer.config import GeneticTransformerConfig
from genetic_transformer.tokenizer.genetic_tokenizer import GeneticTokenizer
from genetic_transformer.model.transformer import GeneticTransformer
from genetic_transformer.data.synthetic_generator import SyntheticDataGenerator
from genetic_transformer.data.dataset import GeneticDataset, create_dataloaders


@pytest.fixture(scope="session")
def tokenizer():
    """Build the tokenizer once for the entire test session."""
    return GeneticTokenizer()


@pytest.fixture(scope="session")
def small_config(tokenizer):
    """A minimal config for fast tests."""
    vocab_size = tokenizer.get_vocab_size()
    return GeneticTransformerConfig(
        d_model=32,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
        max_seq_len=32,
        batch_size=4,
        num_epochs=1,
        learning_rate=1e-4,
        warmup_steps=10,
        label_smoothing=0.1,
        grad_clip=1.0,
        num_train_samples=20,
        num_val_samples=8,
        seed=42,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
    )


@pytest.fixture(scope="session")
def model(small_config):
    """Build a small model for testing."""
    return GeneticTransformer(small_config)


@pytest.fixture(scope="session")
def device():
    return "cpu"


@pytest.fixture(scope="session")
def small_pairs():
    """A small set of synthetic training pairs."""
    gen = SyntheticDataGenerator(seed=42)
    return gen.generate_all(20)


@pytest.fixture(scope="session")
def small_dataset(small_pairs, tokenizer):
    """A small GeneticDataset for testing."""
    return GeneticDataset(small_pairs, tokenizer, max_len=32)


@pytest.fixture(scope="session")
def dataloaders(small_config, tokenizer):
    """Train and val dataloaders from the small config."""
    return create_dataloaders(small_config, tokenizer)
