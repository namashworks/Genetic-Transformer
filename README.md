# :dna: Genetic Transformer

[![CI](https://github.com/InnovationLabsAnthopic/genetic-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/InnovationLabsAnthopic/genetic-transformer/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Transformer architecture inspired by *Attention Is All You Need* (Vaswani et al., 2017), specialized for **bidirectional translation between natural language and genetic sequences**. The model encodes and decodes across DNA, RNA, codons, amino acids, and protein representations, bridging the gap between human-readable descriptions and the language of molecular biology.

---

## Architecture

```
                        Genetic Transformer
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │   ┌──────────────┐    Genetic Knowledge    ┌─────────┐  │
  │   │              │    ┌───────────────┐    │         │  │
  │   │   ENCODER    │    │ Codon Table   │    │ DECODER │  │
  │   │              │    │ Amino Acids   │    │         │  │
  │   │ ┌──────────┐ │    │ DNA/RNA Rules │    │ ┌─────┐ │  │
  │   │ │ Multi-   │ │    └───────┬───────┘    │ │Cross│ │  │
  │   │ │ Head     │ │            │            │ │Attn │ │  │
  │   │ │ Self-    │◄├────────────┘   ┌───────►├─┤     │ │  │
  │   │ │ Attention│ │                │        │ │     │ │  │
  │   │ └──────────┘ │                │        │ └─────┘ │  │
  │   │ ┌──────────┐ │                │        │ ┌─────┐ │  │
  │   │ │ Feed     │ │   Context      │        │ │Feed │ │  │
  │   │ │ Forward  │ ├────Vector──────┘        │ │Fwd  │ │  │
  │   │ └──────────┘ │                         │ └─────┘ │  │
  │   │              │                         │         │  │
  │   │  x4 layers   │                         │x4 layers│  │
  │   └──────┬───────┘                         └────┬────┘  │
  │          │                                      │       │
  │   ┌──────┴───────┐                         ┌────┴────┐  │
  │   │  Positional  │                         │  Linear │  │
  │   │  Encoding +  │                         │  Head + │  │
  │   │  Embedding   │                         │ Softmax │  │
  │   └──────┬───────┘                         └────┬────┘  │
  │          │                                      │       │
  └──────────┼──────────────────────────────────────┼───────┘
             │                                      │
        ┌────┴────┐                            ┌────┴────┐
        │  Input  │                            │ Output  │
        │ "ATG    │                            │ "Met -  │
        │  codon" │                            │  Start" │
        └─────────┘                            └─────────┘
```

---

## Features

- **Genetic Tokenization** -- Custom tokenizer handling DNA bases, RNA bases, codons, amino acid codes, and natural language tokens in a unified vocabulary
- **Codon Table Lookup** -- Integrated genetic knowledge module with the standard codon-to-amino-acid mapping and reverse-complement logic
- **Multi-Head Self-Attention** -- Visualizable attention weights to inspect what the model attends to during translation
- **Label-Smoothed Training** -- Cross-entropy loss with configurable label smoothing for improved generalization
- **Warmup Learning Rate Schedule** -- Linear warmup followed by inverse-square-root decay, following the original Transformer recipe
- **Greedy Decoding** -- Autoregressive inference with greedy token selection
- **Synthetic Data Generation** -- Built-in generator producing paired (natural language, genetic sequence) training examples

---

## Project Structure

```
genetic-transformer/
├── genetic_transformer/
│   ├── __init__.py
│   ├── config.py                  # Dataclass configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # PyTorch Dataset wrapper
│   │   └── synthetic_generator.py # Paired training data generator
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── translator.py          # Greedy decoding translator
│   │   └── visualizer.py          # Attention heatmap visualization
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── amino_acids.py         # Amino acid properties
│   │   ├── codon_table.py         # Standard genetic code
│   │   └── constants.py           # Biological constants
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py           # Multi-head attention
│   │   ├── decoder.py             # Transformer decoder stack
│   │   ├── encoder.py             # Transformer encoder stack
│   │   ├── feed_forward.py        # Position-wise FFN
│   │   ├── positional_encoding.py # Sinusoidal positional encoding
│   │   └── transformer.py         # Full encoder-decoder model
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── genetic_tokenizer.py   # Unified genetic tokenizer
│   │   └── vocab.py               # Vocabulary management
│   └── training/
│       ├── __init__.py
│       ├── losses.py              # Label-smoothed cross-entropy
│       ├── scheduler.py           # Warmup LR scheduler
│       └── trainer.py             # Training loop
├── notebooks/
│   └── genetic_transformer_demo.ipynb
├── checkpoints/                   # Saved model weights (git-ignored)
├── tests/                         # Unit tests
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI pipeline
├── pyproject.toml                 # Build & dependency config
├── requirements.txt               # Pinned dependencies
├── LICENSE
└── README.md
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/InnovationLabsAnthopic/genetic-transformer.git
cd genetic-transformer

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e ".[dev]"
```

### Training

```python
from genetic_transformer.config import GeneticTransformerConfig
from genetic_transformer.tokenizer import GeneticTokenizer
from genetic_transformer.model import GeneticTransformer
from genetic_transformer.data import create_dataloaders
from genetic_transformer.training.trainer import Trainer

# Initialize
tokenizer = GeneticTokenizer()
config = GeneticTransformerConfig(
    src_vocab_size=tokenizer.get_vocab_size(),
    tgt_vocab_size=tokenizer.get_vocab_size(),
)

# Create dataloaders from synthetic data
train_loader, val_loader = create_dataloaders(config, tokenizer)

# Build model and train
model = GeneticTransformer(config)
trainer = Trainer(model, config, train_loader, val_loader)
history = trainer.train()
```

### Translation

```python
from genetic_transformer.inference.translator import GeneticTranslator

translator = GeneticTranslator(model, tokenizer, config)

# Natural language to genetic sequence
result = translator.translate("what amino acid does AUG encode")
print(result)

# Reverse complement
result = translator.translate("reverse complement of ATGCGA")
print(result)

# RNA sequence explanation
result = translator.translate("translate the rna sequence AUG GCU UAA")
print(result)
```

---

## Model Architecture

| Hyperparameter       | Value |
|----------------------|-------|
| `d_model`            | 256   |
| `n_heads`            | 4     |
| `n_encoder_layers`   | 4     |
| `n_decoder_layers`   | 4     |
| `d_ff`               | 512   |
| `dropout`            | 0.1   |
| `max_seq_len`        | 256   |
| `label_smoothing`    | 0.1   |
| `warmup_steps`       | 500   |
| `learning_rate`      | 1e-4  |

---

## Training

The model is trained on synthetically generated pairs of natural language queries and genetic sequence answers. The synthetic data generator covers:

- **Codon lookups** -- "What amino acid does GCU encode?" -> "Alanine (Ala / A)"
- **Reverse translation** -- "What codons code for Leucine?" -> "UUA, UUG, CUU, CUC, CUA, CUG"
- **Complement / reverse complement** -- "complement of ATGC" -> "TACG"
- **Transcription** -- "transcribe ATGCGA to mRNA" -> "AUGCGA"
- **Sequence properties** -- "GC content of ATGCGC" -> "66.7%"

Training uses the Adam optimizer with a warmup schedule (linear warmup for 500 steps, then inverse-square-root decay) and gradient clipping at 1.0.

---

## Example Outputs

**Codon Lookup:**
```
Input:  "What amino acid does UGG encode?"
Output: "Tryptophan (Trp / W)"
```

**Sequence Translation:**
```
Input:  "Translate the mRNA sequence AUGUUUAAA"
Output: "Met-Phe-Lys (MFK)"
```

**Reverse Complement:**
```
Input:  "reverse complement of AATTGGCC"
Output: "GGCCAATT"
```

---

## Developer

Developed by **Namash Aggarwal**

---

## Citation

If you use this software in your research, please cite it using the [CITATION.cff](CITATION.cff) file, or reference the original Transformer paper:

```bibtex
@article{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
               Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
               Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}
```
If you use this software in academic work, please cite this repository.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
