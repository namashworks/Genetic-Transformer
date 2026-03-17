"""Tests for genetic_transformer.knowledge — constants, codon table, amino acids."""

import pytest

from genetic_transformer.knowledge.constants import (
    DNA_BASES, RNA_BASES, START_CODONS, STOP_CODONS,
    COMPLEMENT_DNA, COMPLEMENT_RNA,
)
from genetic_transformer.knowledge.codon_table import (
    CODON_TO_AMINO, AMINO_TO_CODONS,
    translate_rna_to_protein, translate_dna_to_rna, get_reverse_complement,
)
from genetic_transformer.knowledge.amino_acids import AMINO_ACID_PROPERTIES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_dna_bases(self):
        assert set(DNA_BASES) == {"A", "T", "G", "C"}

    def test_rna_bases(self):
        assert set(RNA_BASES) == {"A", "U", "G", "C"}

    def test_start_codons(self):
        assert "AUG" in START_CODONS

    def test_stop_codons(self):
        assert set(STOP_CODONS) == {"UAA", "UAG", "UGA"}

    def test_complement_dna_is_involution(self):
        for base, comp in COMPLEMENT_DNA.items():
            assert COMPLEMENT_DNA[comp] == base

    def test_complement_rna_is_involution(self):
        for base, comp in COMPLEMENT_RNA.items():
            assert COMPLEMENT_RNA[comp] == base


# ---------------------------------------------------------------------------
# Codon table
# ---------------------------------------------------------------------------

class TestCodonTable:
    def test_codon_table_has_64_entries(self):
        assert len(CODON_TO_AMINO) == 64

    def test_all_codons_are_rna_triplets(self):
        valid_bases = set("AUGC")
        for codon in CODON_TO_AMINO:
            assert len(codon) == 3
            assert set(codon).issubset(valid_bases), f"Invalid codon: {codon}"

    def test_stop_codons_in_table(self):
        for stop in STOP_CODONS:
            entry = CODON_TO_AMINO[stop]
            assert entry["type"] == "stop"
            assert entry["one_letter"] == "*"

    def test_start_codon_is_methionine(self):
        entry = CODON_TO_AMINO["AUG"]
        assert entry["one_letter"] == "M"
        assert entry["type"] == "start"

    def test_each_entry_has_required_fields(self):
        required = {"three_letter", "one_letter", "full_name", "type"}
        for codon, entry in CODON_TO_AMINO.items():
            assert required.issubset(entry.keys()), f"Missing fields for {codon}"

    def test_amino_to_codons_reverse_mapping(self):
        """Every codon in the reverse map should map back correctly."""
        for letter, codons in AMINO_TO_CODONS.items():
            for codon in codons:
                assert CODON_TO_AMINO[codon]["one_letter"] == letter

    def test_all_20_amino_acids_represented(self):
        """The standard 20 amino acids plus stop should be present."""
        one_letters = {entry["one_letter"] for entry in CODON_TO_AMINO.values()}
        standard_20 = set("ACDEFGHIKLMNPQRSTVWY")
        assert standard_20.issubset(one_letters)

    def test_degeneracy_leucine_has_six_codons(self):
        assert len(AMINO_TO_CODONS["L"]) == 6

    def test_degeneracy_methionine_has_one_codon(self):
        assert len(AMINO_TO_CODONS["M"]) == 1

    def test_degeneracy_tryptophan_has_one_codon(self):
        assert len(AMINO_TO_CODONS["W"]) == 1


# ---------------------------------------------------------------------------
# Translation functions
# ---------------------------------------------------------------------------

class TestTranslationFunctions:
    def test_translate_rna_to_protein_basic(self):
        # AUG GCU UUU -> M A F
        assert translate_rna_to_protein("AUGGCUUUU") == "MAF"

    def test_translate_rna_stops_at_stop_codon(self):
        # AUG UAA GCU -> M (stops at UAA)
        assert translate_rna_to_protein("AUGUAAGCU") == "M"

    def test_translate_rna_handles_spaces(self):
        assert translate_rna_to_protein("AUG GCU") == "MA"

    def test_translate_rna_case_insensitive(self):
        assert translate_rna_to_protein("auggcu") == "MA"

    def test_translate_rna_empty_returns_empty(self):
        assert translate_rna_to_protein("") == ""

    def test_translate_rna_incomplete_last_codon_ignored(self):
        # AUG GC -> only AUG is complete -> M
        assert translate_rna_to_protein("AUGGC") == "M"

    def test_translate_dna_to_rna(self):
        assert translate_dna_to_rna("ATGCGT") == "AUGCGU"

    def test_translate_dna_to_rna_case_insensitive(self):
        assert translate_dna_to_rna("atgcgt") == "AUGCGU"

    def test_get_reverse_complement_dna(self):
        # ATGC -> reverse CGTA -> complement GCAT
        assert get_reverse_complement("ATGC", "dna") == "GCAT"

    def test_get_reverse_complement_rna(self):
        # AUGC -> reverse CGUA -> complement GCAU
        assert get_reverse_complement("AUGC", "rna") == "GCAU"

    def test_get_reverse_complement_single_base(self):
        assert get_reverse_complement("A", "dna") == "T"

    def test_get_reverse_complement_invalid_type_raises(self):
        with pytest.raises(ValueError, match="seq_type must be"):
            get_reverse_complement("ATGC", "protein")

    def test_reverse_complement_is_involution_dna(self):
        seq = "ATGCGTA"
        assert get_reverse_complement(get_reverse_complement(seq, "dna"), "dna") == seq

    def test_reverse_complement_is_involution_rna(self):
        seq = "AUGCGUA"
        assert get_reverse_complement(get_reverse_complement(seq, "rna"), "rna") == seq


# ---------------------------------------------------------------------------
# Amino acid properties
# ---------------------------------------------------------------------------

class TestAminoAcidProperties:
    def test_has_20_entries(self):
        assert len(AMINO_ACID_PROPERTIES) == 20

    def test_all_standard_amino_acids_present(self):
        expected = set("ACDEFGHIKLMNPQRSTVWY")
        assert set(AMINO_ACID_PROPERTIES.keys()) == expected

    def test_each_entry_has_required_fields(self):
        required = {"full_name", "three_letter", "molecular_weight",
                     "charge", "polarity", "hydrophobicity", "classification"}
        for aa, props in AMINO_ACID_PROPERTIES.items():
            assert required.issubset(props.keys()), f"Missing fields for {aa}"

    def test_molecular_weights_are_positive(self):
        for aa, props in AMINO_ACID_PROPERTIES.items():
            assert props["molecular_weight"] > 0, f"{aa} has non-positive MW"

    def test_glycine_is_lightest(self):
        glycine_mw = AMINO_ACID_PROPERTIES["G"]["molecular_weight"]
        for aa, props in AMINO_ACID_PROPERTIES.items():
            assert props["molecular_weight"] >= glycine_mw

    def test_tryptophan_is_heaviest(self):
        trp_mw = AMINO_ACID_PROPERTIES["W"]["molecular_weight"]
        for aa, props in AMINO_ACID_PROPERTIES.items():
            assert props["molecular_weight"] <= trp_mw

    def test_charge_values_valid(self):
        valid_charges = {"positive", "negative", "neutral"}
        for aa, props in AMINO_ACID_PROPERTIES.items():
            assert props["charge"] in valid_charges, f"{aa} has invalid charge"

    def test_polarity_values_valid(self):
        valid = {"polar", "nonpolar"}
        for aa, props in AMINO_ACID_PROPERTIES.items():
            assert props["polarity"] in valid, f"{aa} has invalid polarity"

    def test_known_properties_arginine(self):
        r = AMINO_ACID_PROPERTIES["R"]
        assert r["full_name"] == "Arginine"
        assert r["charge"] == "positive"
        assert r["polarity"] == "polar"
