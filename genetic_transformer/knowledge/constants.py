"""Biological constants for genetic sequence processing."""

# DNA and RNA base alphabets
DNA_BASES = ["A", "T", "G", "C"]
RNA_BASES = ["A", "U", "G", "C"]

# Translation control codons
START_CODONS = ["AUG"]
STOP_CODONS = ["UAA", "UAG", "UGA"]

# Watson-Crick base pairing
COMPLEMENT_DNA = {"A": "T", "T": "A", "G": "C", "C": "G"}
COMPLEMENT_RNA = {"A": "U", "U": "A", "G": "C", "C": "G"}
