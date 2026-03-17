"""Tests for genetic_transformer.data — synthetic generator and dataset."""

import pytest
import torch

from genetic_transformer.data.synthetic_generator import SyntheticDataGenerator
from genetic_transformer.data.dataset import GeneticDataset, create_dataloaders
from genetic_transformer.tokenizer.vocab import PAD_IDX, SOS_IDX, EOS_IDX


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------

class TestSyntheticDataGenerator:
    def test_generate_returns_correct_count(self):
        gen = SyntheticDataGenerator(seed=42)
        pairs = gen.generate_all(50)
        assert len(pairs) == 50

    def test_each_pair_is_src_tgt_tuple(self):
        gen = SyntheticDataGenerator(seed=42)
        pairs = gen.generate_all(10)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            src, tgt = pair
            assert isinstance(src, str) and len(src) > 0
            assert isinstance(tgt, str) and len(tgt) > 0

    def test_deterministic_with_same_seed(self):
        pairs1 = SyntheticDataGenerator(seed=123).generate_all(20)
        pairs2 = SyntheticDataGenerator(seed=123).generate_all(20)
        assert pairs1 == pairs2

    def test_different_seeds_differ(self):
        pairs1 = SyntheticDataGenerator(seed=1).generate_all(20)
        pairs2 = SyntheticDataGenerator(seed=2).generate_all(20)
        assert pairs1 != pairs2

    def test_generate_single_sample(self):
        gen = SyntheticDataGenerator(seed=42)
        pairs = gen.generate_all(1)
        assert len(pairs) == 1

    def test_generate_large_batch(self):
        gen = SyntheticDataGenerator(seed=42)
        pairs = gen.generate_all(200)
        assert len(pairs) == 200


# ---------------------------------------------------------------------------
# GeneticDataset
# ---------------------------------------------------------------------------

class TestGeneticDataset:
    def test_len(self, small_dataset):
        assert len(small_dataset) == 20

    def test_getitem_returns_dict(self, small_dataset):
        item = small_dataset[0]
        assert isinstance(item, dict)
        assert "src" in item
        assert "tgt" in item
        assert "tgt_y" in item

    def test_item_tensors_are_long(self, small_dataset):
        item = small_dataset[0]
        assert item["src"].dtype == torch.long
        assert item["tgt"].dtype == torch.long
        assert item["tgt_y"].dtype == torch.long

    def test_src_has_correct_length(self, small_dataset):
        item = small_dataset[0]
        assert item["src"].shape == (32,)  # max_len=32

    def test_tgt_starts_with_sos(self, small_dataset):
        item = small_dataset[0]
        assert item["tgt"][0].item() == SOS_IDX

    def test_tgt_y_contains_eos(self, small_dataset):
        item = small_dataset[0]
        assert EOS_IDX in item["tgt_y"].tolist()

    def test_tgt_and_tgt_y_same_length(self, small_dataset):
        item = small_dataset[0]
        assert item["tgt"].shape == item["tgt_y"].shape

    def test_all_indices_valid(self, small_dataset, tokenizer):
        vocab_size = tokenizer.get_vocab_size()
        for i in range(min(5, len(small_dataset))):
            item = small_dataset[i]
            assert item["src"].max().item() < vocab_size
            assert item["tgt"].max().item() < vocab_size
            assert item["tgt_y"].max().item() < vocab_size


# ---------------------------------------------------------------------------
# create_dataloaders
# ---------------------------------------------------------------------------

class TestDataLoaders:
    def test_returns_two_loaders(self, dataloaders):
        train_loader, val_loader = dataloaders
        assert train_loader is not None
        assert val_loader is not None

    def test_train_loader_yields_batches(self, dataloaders):
        train_loader, _ = dataloaders
        batch = next(iter(train_loader))
        assert "src" in batch
        assert "tgt" in batch
        assert "tgt_y" in batch

    def test_batch_dimensions(self, dataloaders, small_config):
        train_loader, _ = dataloaders
        batch = next(iter(train_loader))
        # batch_size=4, max_seq_len=32
        assert batch["src"].shape[1] == small_config.max_seq_len
        assert batch["tgt"].shape[1] <= small_config.max_seq_len
        assert batch["src"].shape[0] <= small_config.batch_size

    def test_val_loader_yields_batches(self, dataloaders):
        _, val_loader = dataloaders
        batch = next(iter(val_loader))
        assert batch["src"].dim() == 2
