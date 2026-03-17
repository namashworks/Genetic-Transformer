"""Dual-domain tokenizer handling both natural language and genetic sequences."""

import re
from genetic_transformer.tokenizer.vocab import Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX
from genetic_transformer.knowledge.codon_table import CODON_TO_AMINO, AMINO_TO_CODONS
from genetic_transformer.knowledge.amino_acids import AMINO_ACID_PROPERTIES
from genetic_transformer.knowledge.constants import DNA_BASES, RNA_BASES


class GeneticTokenizer:
    """Unified tokenizer for natural language and genetic sequences."""

    def __init__(self):
        self.vocab = Vocabulary()
        self._build_vocab()

    def _build_vocab(self):
        """Build the unified vocabulary from all domains."""
        tokens = []

        # Domain markers
        tokens.extend(["<NL>", "<DNA>", "<RNA>", "<PROTEIN>", "<CODON>"])

        # Nucleotide bases (as individual tokens for non-codon-aligned sequences)
        tokens.extend(["_A", "_T", "_G", "_C", "_U"])

        # All 64 codons as single tokens
        for codon in CODON_TO_AMINO:
            tokens.append(codon)

        # Amino acid single-letter codes (prefixed to avoid collision with bases)
        for aa_code in AMINO_ACID_PROPERTIES:
            tokens.append(f"<AA_{aa_code}>")
        tokens.append("<AA_*>")  # Stop codon symbol

        # Natural language vocabulary - biology terms
        nl_tokens = self._build_nl_vocabulary()
        tokens.extend(nl_tokens)

        self.vocab.build_from_tokens(tokens)

    def _build_nl_vocabulary(self) -> list[str]:
        """Build the natural language portion of the vocabulary."""
        tokens = []

        # All amino acid full names and three-letter codes
        for aa_code, props in AMINO_ACID_PROPERTIES.items():
            tokens.append(props["full_name"].lower())
            tokens.append(props["three_letter"].lower())

        # Biology-specific terms
        bio_terms = [
            "codon", "codons", "amino", "acid", "acids", "protein", "proteins",
            "sequence", "sequences", "gene", "genes", "genetic", "code",
            "dna", "rna", "mrna", "trna", "rrna",
            "nucleotide", "nucleotides", "base", "bases", "pair", "pairs",
            "adenine", "thymine", "guanine", "cytosine", "uracil",
            "purine", "purines", "pyrimidine", "pyrimidines",
            "encode", "encodes", "encoded", "encoding",
            "decode", "decodes", "decoded", "decoding",
            "translate", "translates", "translated", "translation",
            "transcribe", "transcribes", "transcribed", "transcription",
            "complement", "complementary", "reverse",
            "start", "stop", "reading", "frame", "frames",
            "open", "orf",
            "strand", "template", "sense", "antisense",
            "peptide", "peptides", "polypeptide",
            "residue", "residues", "chain",
            "methionine", "tryptophan",
            "polar", "nonpolar", "hydrophobic", "hydrophilic",
            "charged", "uncharged", "positive", "negative", "neutral",
            "aromatic", "aliphatic",
            "essential", "nonessential",
            "molecular", "weight", "mass",
            "bond", "bonds", "peptide",
            "folding", "structure", "primary", "secondary", "tertiary",
            "mutation", "mutations", "substitution",
            "silent", "missense", "nonsense",
            "degenerate", "degeneracy", "redundant", "redundancy",
            "wobble", "position",
        ]
        tokens.extend(bio_terms)

        # Common English words for sentence construction
        common_words = [
            "the", "a", "an", "is", "are", "was", "were",
            "this", "that", "these", "those",
            "what", "which", "how", "does", "do", "did",
            "of", "to", "for", "in", "on", "at", "by", "from", "with",
            "and", "or", "but", "not", "no", "yes",
            "it", "its", "they", "them", "their",
            "has", "have", "had", "been", "being",
            "can", "could", "will", "would", "may", "might",
            "into", "onto", "than", "then", "also",
            "codes", "produces", "results", "converts", "forms",
            "contains", "consists", "includes", "represents",
            "followed", "following", "starts", "ends", "begins",
            "first", "second", "third", "last", "next",
            "one", "two", "three", "four", "five", "six",
            "called", "known", "named",
            "type", "group", "class", "category", "family",
            "function", "role", "property", "properties",
            "number", "position", "length", "size",
            "same", "different", "similar",
            "found", "located", "present",
            "between", "among", "within", "across",
            "each", "every", "all", "both", "many",
            "specific", "particular", "general",
            "about", "there", "here",
        ]
        tokens.extend(common_words)

        # Additional descriptive terms
        descriptors = [
            "small", "large", "long", "short",
            "simple", "complex",
            "single", "double", "triple",
            "letter", "symbol",
            "table", "list", "map",
            "standard", "universal", "common",
            "biological", "chemical", "physical",
            "important", "necessary", "required",
        ]
        tokens.extend(descriptors)

        return tokens

    def detect_domain(self, text: str) -> str:
        """Auto-detect the domain of the input text."""
        text_clean = text.strip().upper()
        # Check if it looks like a protein sequence (single amino acid letters)
        if re.match(r'^[ACDEFGHIKLMNPQRSTVWY\s\*]+$', text_clean) and len(text_clean.replace(" ", "")) <= 50:
            # Could be protein if it has non-nucleotide amino acid letters
            non_nucleotide = set(text_clean.replace(" ", "").replace("*", "")) - {"A", "C", "G", "U", "T"}
            if non_nucleotide:
                return "protein"
        # Check if it looks like DNA
        if re.match(r'^[ATGC\s]+$', text_clean):
            return "dna"
        # Check if it looks like RNA
        if re.match(r'^[AUGC\s]+$', text_clean):
            return "rna"
        # Check if it looks like codon-spaced RNA
        words = text_clean.split()
        if all(len(w) == 3 and re.match(r'^[AUGC]+$', w) for w in words) and len(words) > 0:
            return "rna"
        return "nl"

    def tokenize(self, text: str, mode: str = "auto") -> list[str]:
        """Tokenize input text into a list of token strings."""
        if mode == "auto":
            mode = self.detect_domain(text)

        if mode in ("dna", "rna"):
            return self._tokenize_nucleotide(text, mode)
        elif mode == "protein":
            return self._tokenize_protein(text)
        else:
            return self._tokenize_nl(text)

    def _tokenize_nucleotide(self, text: str, seq_type: str) -> list[str]:
        """Tokenize a DNA or RNA sequence."""
        marker = "<DNA>" if seq_type == "dna" else "<RNA>"
        tokens = [marker]
        seq = text.strip().upper().replace(" ", "")

        # Try codon grouping (multiples of 3)
        if len(seq) % 3 == 0 and len(seq) >= 3:
            for i in range(0, len(seq), 3):
                codon = seq[i:i+3]
                # For DNA, convert to RNA codon for lookup
                if seq_type == "dna":
                    rna_codon = codon.replace("T", "U")
                else:
                    rna_codon = codon
                if rna_codon in CODON_TO_AMINO:
                    tokens.append(rna_codon)
                else:
                    # Fall back to individual bases
                    for base in codon:
                        tokens.append(f"_{base}")
        else:
            # Not codon-aligned, use individual bases
            for base in seq:
                if base in ("A", "T", "G", "C", "U"):
                    tokens.append(f"_{base}")

        return tokens

    def _tokenize_protein(self, text: str) -> list[str]:
        """Tokenize a protein sequence (amino acid one-letter codes)."""
        tokens = ["<PROTEIN>"]
        seq = text.strip().upper().replace(" ", "")
        for char in seq:
            if char == "*":
                tokens.append("<AA_*>")
            elif char in AMINO_ACID_PROPERTIES:
                tokens.append(f"<AA_{char}>")
        return tokens

    def _tokenize_nl(self, text: str) -> list[str]:
        """Tokenize natural language text."""
        tokens = ["<NL>"]
        # Lowercase and split on whitespace/punctuation
        text = text.lower().strip()
        # Split keeping simple word boundaries
        words = re.findall(r'[a-z]+|[0-9]+|\*', text)
        for word in words:
            if word in self.vocab:
                tokens.append(word)
            elif word == "*":
                tokens.append("<AA_*>")
            else:
                # Check if it's an uppercase genetic sequence embedded in NL
                if re.match(r'^[augc]+$', word) and len(word) == 3:
                    tokens.append(word.upper())
                else:
                    tokens.append("<UNK>")
        return tokens

    def encode(self, text: str, mode: str = "auto", max_len: int = 256) -> list[int]:
        """Tokenize and convert to padded index sequence with SOS/EOS."""
        tokens = self.tokenize(text, mode)
        indices = [SOS_IDX]
        for token in tokens:
            indices.append(self.vocab.encode(token))
        indices.append(EOS_IDX)

        # Truncate if needed
        if len(indices) > max_len:
            indices = indices[:max_len - 1] + [EOS_IDX]

        # Pad
        while len(indices) < max_len:
            indices.append(PAD_IDX)

        return indices

    def decode(self, indices: list[int], skip_special: bool = True) -> str:
        """Convert index sequence back to readable string."""
        special = {"<PAD>", "<SOS>", "<EOS>", "<UNK>", "<SEP>"}
        tokens = []
        for idx in indices:
            token = self.vocab.decode(idx)
            if skip_special and token in special:
                continue
            if token == "<PAD>":
                break
            tokens.append(token)
        return self._detokenize(tokens)

    def _detokenize(self, tokens: list[str]) -> str:
        """Convert token list back to a human-readable string."""
        parts = []
        for token in tokens:
            if token in ("<NL>", "<DNA>", "<RNA>", "<PROTEIN>", "<CODON>"):
                continue  # Skip domain markers
            elif token.startswith("<AA_") and token.endswith(">"):
                # Extract amino acid letter
                parts.append(token[4:-1])
            elif token.startswith("_") and len(token) == 2:
                # Single nucleotide base
                parts.append(token[1])
            elif token in CODON_TO_AMINO:
                # Codon token
                parts.append(token)
            else:
                parts.append(token)
        return " ".join(parts)

    def get_vocab_size(self) -> int:
        """Return the total vocabulary size."""
        return len(self.vocab)

    def get_token_list(self) -> list[str]:
        """Return all tokens in the vocabulary."""
        return list(self.vocab.token2idx.keys())
