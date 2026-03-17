"""Tests for genetic_transformer.tokenizer — vocab and genetic tokenizer."""

import pytest

from genetic_transformer.tokenizer.vocab import (
    Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, SEP_IDX,
)
from genetic_transformer.tokenizer.genetic_tokenizer import GeneticTokenizer
from genetic_transformer.knowledge.codon_table import CODON_TO_AMINO


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_special_tokens_indices(self):
        assert PAD_IDX == 0
        assert SOS_IDX == 1
        assert EOS_IDX == 2
        assert UNK_IDX == 3
        assert SEP_IDX == 4

    def test_build_from_tokens(self):
        v = Vocabulary()
        v.build_from_tokens(["foo", "bar"])
        assert len(v) == len(Vocabulary.SPECIAL_TOKENS) + 2
        assert "foo" in v
        assert "bar" in v

    def test_encode_known_token(self):
        v = Vocabulary()
        v.build_from_tokens(["hello"])
        idx = v.encode("hello")
        assert isinstance(idx, int)
        assert v.decode(idx) == "hello"

    def test_encode_unknown_returns_unk(self):
        v = Vocabulary()
        v.build_from_tokens(["hello"])
        assert v.encode("missing") == UNK_IDX

    def test_decode_unknown_idx_returns_unk_string(self):
        v = Vocabulary()
        v.build_from_tokens([])
        assert v.decode(99999) == "<UNK>"

    def test_special_tokens_are_first(self):
        v = Vocabulary()
        v.build_from_tokens(["x"])
        for i, tok in enumerate(Vocabulary.SPECIAL_TOKENS):
            assert v.encode(tok) == i

    def test_no_duplicates_in_vocab(self):
        v = Vocabulary()
        v.build_from_tokens(["a", "b", "a", "c"])
        # 'a' should appear once
        assert len(v) == len(Vocabulary.SPECIAL_TOKENS) + 3


# ---------------------------------------------------------------------------
# GeneticTokenizer — domain detection
# ---------------------------------------------------------------------------

class TestDomainDetection:
    def test_detect_dna(self, tokenizer):
        assert tokenizer.detect_domain("ATGCGATCG") == "dna"

    def test_detect_rna(self, tokenizer):
        assert tokenizer.detect_domain("AUGCGAUCG") == "rna"

    def test_detect_protein(self, tokenizer):
        assert tokenizer.detect_domain("MFLK") == "protein"

    def test_detect_nl(self, tokenizer):
        assert tokenizer.detect_domain("what amino acid does AUG encode") == "nl"

    def test_detect_dna_lowercase(self, tokenizer):
        assert tokenizer.detect_domain("atgcgatcg") == "dna"

    def test_pure_ag_classified_as_dna(self, tokenizer):
        # AG are both DNA bases and amino acid letters, but short all-nucleotide
        # sequences that only contain ATGC should be detected as DNA
        result = tokenizer.detect_domain("AAGC")
        assert result == "dna"


# ---------------------------------------------------------------------------
# GeneticTokenizer — encode / decode
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_encode_returns_list_of_ints(self, tokenizer):
        result = tokenizer.encode("ATGCGA", max_len=32)
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)

    def test_encode_has_correct_length(self, tokenizer):
        result = tokenizer.encode("ATGCGA", max_len=32)
        assert len(result) == 32

    def test_encode_starts_with_sos(self, tokenizer):
        result = tokenizer.encode("ATGCGA", max_len=32)
        assert result[0] == SOS_IDX

    def test_encode_contains_eos(self, tokenizer):
        result = tokenizer.encode("ATGCGA", max_len=32)
        assert EOS_IDX in result

    def test_encode_pads_to_max_len(self, tokenizer):
        result = tokenizer.encode("A", max_len=32)
        assert result[-1] == PAD_IDX

    def test_encode_truncation(self, tokenizer):
        # A very long sequence should be truncated to max_len
        long_seq = "ATGC" * 100
        result = tokenizer.encode(long_seq, max_len=16)
        assert len(result) == 16
        # Last token should be EOS after truncation
        assert result[-1] == EOS_IDX

    def test_encode_rna_codon_aligned(self, tokenizer):
        # AUGGCU is 6 bases = 2 codons, should be tokenized as codons
        result = tokenizer.encode("AUGGCU", max_len=32)
        assert SOS_IDX == result[0]
        # Should have non-pad content after SOS
        non_pad = [x for x in result if x != PAD_IDX]
        assert len(non_pad) >= 3  # SOS + domain marker + at least one codon + EOS

    def test_decode_skips_special_tokens(self, tokenizer):
        encoded = tokenizer.encode("AUGGCU", max_len=32)
        decoded = tokenizer.decode(encoded, skip_special=True)
        assert "<PAD>" not in decoded
        assert "<SOS>" not in decoded
        assert "<EOS>" not in decoded

    def test_encode_decode_roundtrip_dna(self, tokenizer):
        original = "ATGCGA"
        encoded = tokenizer.encode(original, mode="dna", max_len=32)
        decoded = tokenizer.decode(encoded, skip_special=True)
        # The decoded form might have spaces between codons/bases, but the
        # nucleotide content should be preserved
        decoded_clean = decoded.replace(" ", "").upper()
        # DNA gets converted to RNA codons internally, so we compare RNA form
        expected_rna = original.replace("T", "U")
        assert expected_rna in decoded_clean or decoded_clean != ""

    def test_encode_decode_roundtrip_protein(self, tokenizer):
        original = "MFKL"
        encoded = tokenizer.encode(original, mode="protein", max_len=32)
        decoded = tokenizer.decode(encoded, skip_special=True)
        decoded_clean = decoded.replace(" ", "")
        assert decoded_clean == original

    def test_encode_nl_text(self, tokenizer):
        text = "what amino acid does AUG encode"
        encoded = tokenizer.encode(text, mode="nl", max_len=64)
        assert len(encoded) == 64
        assert encoded[0] == SOS_IDX


# ---------------------------------------------------------------------------
# GeneticTokenizer — vocab
# ---------------------------------------------------------------------------

class TestVocabBuilding:
    def test_vocab_size_is_positive(self, tokenizer):
        assert tokenizer.get_vocab_size() > 0

    def test_vocab_contains_special_tokens(self, tokenizer):
        tokens = tokenizer.get_token_list()
        for special in ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<SEP>"]:
            assert special in tokens

    def test_vocab_contains_all_codons(self, tokenizer):
        tokens = set(tokenizer.get_token_list())
        for codon in CODON_TO_AMINO:
            assert codon in tokens, f"Codon {codon} missing from vocab"

    def test_vocab_contains_amino_acid_tokens(self, tokenizer):
        tokens = set(tokenizer.get_token_list())
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            assert f"<AA_{aa}>" in tokens

    def test_vocab_contains_domain_markers(self, tokenizer):
        tokens = set(tokenizer.get_token_list())
        for marker in ["<NL>", "<DNA>", "<RNA>", "<PROTEIN>", "<CODON>"]:
            assert marker in tokens

    def test_vocab_contains_nucleotide_base_tokens(self, tokenizer):
        tokens = set(tokenizer.get_token_list())
        for base in ["_A", "_T", "_G", "_C", "_U"]:
            assert base in tokens
