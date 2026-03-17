"""Standard genetic code: codon table and translation utilities."""

from typing import Dict, List

from .constants import COMPLEMENT_DNA, COMPLEMENT_RNA


def _entry(three_letter: str, one_letter: str, full_name: str, codon_type: str) -> Dict:
    return {
        "three_letter": three_letter,
        "one_letter": one_letter,
        "full_name": full_name,
        "type": codon_type,
    }


# Complete standard genetic code — all 64 RNA codons
CODON_TO_AMINO: Dict[str, Dict] = {
    # Phenylalanine (F)
    "UUU": _entry("Phe", "F", "Phenylalanine", "standard"),
    "UUC": _entry("Phe", "F", "Phenylalanine", "standard"),
    # Leucine (L)
    "UUA": _entry("Leu", "L", "Leucine", "standard"),
    "UUG": _entry("Leu", "L", "Leucine", "standard"),
    "CUU": _entry("Leu", "L", "Leucine", "standard"),
    "CUC": _entry("Leu", "L", "Leucine", "standard"),
    "CUA": _entry("Leu", "L", "Leucine", "standard"),
    "CUG": _entry("Leu", "L", "Leucine", "standard"),
    # Isoleucine (I)
    "AUU": _entry("Ile", "I", "Isoleucine", "standard"),
    "AUC": _entry("Ile", "I", "Isoleucine", "standard"),
    "AUA": _entry("Ile", "I", "Isoleucine", "standard"),
    # Methionine / Start (M)
    "AUG": _entry("Met", "M", "Methionine", "start"),
    # Valine (V)
    "GUU": _entry("Val", "V", "Valine", "standard"),
    "GUC": _entry("Val", "V", "Valine", "standard"),
    "GUA": _entry("Val", "V", "Valine", "standard"),
    "GUG": _entry("Val", "V", "Valine", "standard"),
    # Serine (S) — UCN block
    "UCU": _entry("Ser", "S", "Serine", "standard"),
    "UCC": _entry("Ser", "S", "Serine", "standard"),
    "UCA": _entry("Ser", "S", "Serine", "standard"),
    "UCG": _entry("Ser", "S", "Serine", "standard"),
    # Proline (P)
    "CCU": _entry("Pro", "P", "Proline", "standard"),
    "CCC": _entry("Pro", "P", "Proline", "standard"),
    "CCA": _entry("Pro", "P", "Proline", "standard"),
    "CCG": _entry("Pro", "P", "Proline", "standard"),
    # Threonine (T)
    "ACU": _entry("Thr", "T", "Threonine", "standard"),
    "ACC": _entry("Thr", "T", "Threonine", "standard"),
    "ACA": _entry("Thr", "T", "Threonine", "standard"),
    "ACG": _entry("Thr", "T", "Threonine", "standard"),
    # Alanine (A)
    "GCU": _entry("Ala", "A", "Alanine", "standard"),
    "GCC": _entry("Ala", "A", "Alanine", "standard"),
    "GCA": _entry("Ala", "A", "Alanine", "standard"),
    "GCG": _entry("Ala", "A", "Alanine", "standard"),
    # Tyrosine (Y)
    "UAU": _entry("Tyr", "Y", "Tyrosine", "standard"),
    "UAC": _entry("Tyr", "Y", "Tyrosine", "standard"),
    # Stop codons (*)
    "UAA": _entry("Stop", "*", "Stop", "stop"),
    "UAG": _entry("Stop", "*", "Stop", "stop"),
    "UGA": _entry("Stop", "*", "Stop", "stop"),
    # Histidine (H)
    "CAU": _entry("His", "H", "Histidine", "standard"),
    "CAC": _entry("His", "H", "Histidine", "standard"),
    # Glutamine (Q)
    "CAA": _entry("Gln", "Q", "Glutamine", "standard"),
    "CAG": _entry("Gln", "Q", "Glutamine", "standard"),
    # Asparagine (N)
    "AAU": _entry("Asn", "N", "Asparagine", "standard"),
    "AAC": _entry("Asn", "N", "Asparagine", "standard"),
    # Lysine (K)
    "AAA": _entry("Lys", "K", "Lysine", "standard"),
    "AAG": _entry("Lys", "K", "Lysine", "standard"),
    # Aspartic acid (D)
    "GAU": _entry("Asp", "D", "Aspartic acid", "standard"),
    "GAC": _entry("Asp", "D", "Aspartic acid", "standard"),
    # Glutamic acid (E)
    "GAA": _entry("Glu", "E", "Glutamic acid", "standard"),
    "GAG": _entry("Glu", "E", "Glutamic acid", "standard"),
    # Cysteine (C)
    "UGU": _entry("Cys", "C", "Cysteine", "standard"),
    "UGC": _entry("Cys", "C", "Cysteine", "standard"),
    # Tryptophan (W)
    "UGG": _entry("Trp", "W", "Tryptophan", "standard"),
    # Arginine (R) — CGN block
    "CGU": _entry("Arg", "R", "Arginine", "standard"),
    "CGC": _entry("Arg", "R", "Arginine", "standard"),
    "CGA": _entry("Arg", "R", "Arginine", "standard"),
    "CGG": _entry("Arg", "R", "Arginine", "standard"),
    # Serine (S) — AGU/AGC
    "AGU": _entry("Ser", "S", "Serine", "standard"),
    "AGC": _entry("Ser", "S", "Serine", "standard"),
    # Arginine (R) — AGA/AGG
    "AGA": _entry("Arg", "R", "Arginine", "standard"),
    "AGG": _entry("Arg", "R", "Arginine", "standard"),
    # Glycine (G)
    "GGU": _entry("Gly", "G", "Glycine", "standard"),
    "GGC": _entry("Gly", "G", "Glycine", "standard"),
    "GGA": _entry("Gly", "G", "Glycine", "standard"),
    "GGG": _entry("Gly", "G", "Glycine", "standard"),
}

# Reverse mapping: amino acid one-letter code → list of codons
AMINO_TO_CODONS: Dict[str, List[str]] = {}
for _codon, _info in CODON_TO_AMINO.items():
    _letter = _info["one_letter"]
    AMINO_TO_CODONS.setdefault(_letter, []).append(_codon)


def translate_rna_to_protein(rna_seq: str) -> str:
    """Translate an RNA sequence to a protein string (one-letter amino acid codes).

    Reads the sequence in codons of 3 nucleotides. Translation stops when a
    stop codon is encountered or the sequence is exhausted.

    Args:
        rna_seq: RNA nucleotide string (e.g. "AUGGCUUAA").

    Returns:
        Protein string using one-letter amino acid codes.
    """
    rna_seq = rna_seq.upper().replace(" ", "")
    protein: list[str] = []
    for i in range(0, len(rna_seq) - 2, 3):
        codon = rna_seq[i : i + 3]
        entry = CODON_TO_AMINO.get(codon)
        if entry is None:
            break
        if entry["type"] == "stop":
            break
        protein.append(entry["one_letter"])
    return "".join(protein)


def translate_dna_to_rna(dna_seq: str) -> str:
    """Convert a DNA sequence to RNA by replacing T with U.

    Args:
        dna_seq: DNA nucleotide string.

    Returns:
        Corresponding RNA string.
    """
    return dna_seq.upper().replace("T", "U")


def get_reverse_complement(seq: str, seq_type: str = "dna") -> str:
    """Return the reverse complement of a nucleotide sequence.

    Args:
        seq: Nucleotide string.
        seq_type: Either ``"dna"`` or ``"rna"``.

    Returns:
        Reverse complement string.

    Raises:
        ValueError: If *seq_type* is not ``"dna"`` or ``"rna"``.
    """
    seq = seq.upper()
    if seq_type.lower() == "dna":
        complement_map = COMPLEMENT_DNA
    elif seq_type.lower() == "rna":
        complement_map = COMPLEMENT_RNA
    else:
        raise ValueError(f"seq_type must be 'dna' or 'rna', got '{seq_type}'")

    return "".join(complement_map[base] for base in reversed(seq))
