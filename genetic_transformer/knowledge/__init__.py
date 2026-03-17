"""Genetic knowledge base: constants, codon tables, and amino acid properties."""

from .constants import (
    COMPLEMENT_DNA,
    COMPLEMENT_RNA,
    DNA_BASES,
    RNA_BASES,
    START_CODONS,
    STOP_CODONS,
)
from .codon_table import (
    AMINO_TO_CODONS,
    CODON_TO_AMINO,
    get_reverse_complement,
    translate_dna_to_rna,
    translate_rna_to_protein,
)
from .amino_acids import AMINO_ACID_PROPERTIES

__all__ = [
    "DNA_BASES",
    "RNA_BASES",
    "START_CODONS",
    "STOP_CODONS",
    "COMPLEMENT_DNA",
    "COMPLEMENT_RNA",
    "CODON_TO_AMINO",
    "AMINO_TO_CODONS",
    "translate_rna_to_protein",
    "translate_dna_to_rna",
    "get_reverse_complement",
    "AMINO_ACID_PROPERTIES",
]
