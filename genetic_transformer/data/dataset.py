"""PyTorch Dataset and DataLoader factory for genetic transformer training."""

import torch
from torch.utils.data import Dataset, DataLoader

from genetic_transformer.config import GeneticTransformerConfig
from genetic_transformer.tokenizer.genetic_tokenizer import GeneticTokenizer
from genetic_transformer.tokenizer.vocab import SOS_IDX, EOS_IDX, PAD_IDX
from genetic_transformer.data.synthetic_generator import SyntheticDataGenerator


class GeneticDataset(Dataset):
    """Dataset of (source, target) pairs for transformer training."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer: GeneticTokenizer,
                 max_len: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        src_text, tgt_text = self.pairs[idx]

        # Encode source
        src = self.tokenizer.encode(src_text, max_len=self.max_len)

        # Encode target - we need tgt (decoder input with SOS) and tgt_y (labels with EOS)
        tgt_tokens = self.tokenizer.tokenize(tgt_text)
        tgt_indices = [self.tokenizer.vocab.encode(t) for t in tgt_tokens]

        # Decoder input: SOS + tokens (truncate if needed)
        tgt_input = [SOS_IDX] + tgt_indices
        if len(tgt_input) > self.max_len - 1:
            tgt_input = tgt_input[:self.max_len - 1]

        # Labels: tokens + EOS
        tgt_label = tgt_indices + [EOS_IDX]
        if len(tgt_label) > self.max_len - 1:
            tgt_label = tgt_label[:self.max_len - 1]
            tgt_label[-1] = EOS_IDX

        # Ensure same length
        max_tgt = max(len(tgt_input), len(tgt_label))
        while len(tgt_input) < max_tgt:
            tgt_input.append(PAD_IDX)
        while len(tgt_label) < max_tgt:
            tgt_label.append(PAD_IDX)

        # Pad to max_len
        while len(tgt_input) < self.max_len:
            tgt_input.append(PAD_IDX)
            tgt_label.append(PAD_IDX)

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt_input, dtype=torch.long),
            "tgt_y": torch.tensor(tgt_label, dtype=torch.long),
        }


def create_dataloaders(config: GeneticTransformerConfig,
                       tokenizer: GeneticTokenizer) -> tuple[DataLoader, DataLoader]:
    """Generate synthetic data and create train/val DataLoaders."""
    generator = SyntheticDataGenerator(seed=config.seed)

    train_pairs = generator.generate_all(config.num_train_samples)
    val_pairs = generator.generate_all(config.num_val_samples)

    train_dataset = GeneticDataset(train_pairs, tokenizer, max_len=config.max_seq_len)
    val_dataset = GeneticDataset(val_pairs, tokenizer, max_len=config.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader
