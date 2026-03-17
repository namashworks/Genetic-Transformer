"""Synthetic training data generator for NL <-> Genetic sequence pairs."""

import random
from genetic_transformer.knowledge.codon_table import (
    CODON_TO_AMINO, AMINO_TO_CODONS, translate_rna_to_protein, translate_dna_to_rna
)
from genetic_transformer.knowledge.amino_acids import AMINO_ACID_PROPERTIES
from genetic_transformer.knowledge.constants import (
    DNA_BASES, RNA_BASES, STOP_CODONS, COMPLEMENT_DNA, COMPLEMENT_RNA
)


class SyntheticDataGenerator:
    """Generates paired (source, target) training examples for both translation directions."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        # Non-stop codons for generating valid protein-coding sequences
        self.coding_codons = [c for c in CODON_TO_AMINO if CODON_TO_AMINO[c]["type"] != "stop"]

    def generate_all(self, n_samples: int) -> list[tuple[str, str]]:
        """Generate n_samples paired (source, target) examples, balanced across categories."""
        all_pairs = []

        # Gather all pair generators with their approximate proportions
        generators = [
            (self._gen_codon_lookup_pairs, 0.12),
            (self._gen_sequence_translation_pairs, 0.20),
            (self._gen_dna_to_rna_pairs, 0.10),
            (self._gen_reverse_complement_pairs, 0.08),
            (self._gen_amino_acid_property_pairs, 0.10),
            (self._gen_sequence_explain_pairs, 0.20),
            (self._gen_protein_describe_pairs, 0.10),
            (self._gen_reading_frame_pairs, 0.05),
            (self._gen_start_stop_pairs, 0.05),
        ]

        for gen_func, proportion in generators:
            count = int(n_samples * proportion)
            pairs = gen_func(count)
            all_pairs.extend(pairs)

        # Fill remaining with mixed
        while len(all_pairs) < n_samples:
            gen_func, _ = self.rng.choice(generators)
            pairs = gen_func(1)
            all_pairs.extend(pairs)

        self.rng.shuffle(all_pairs)
        return all_pairs[:n_samples]

    def _gen_codon_lookup_pairs(self, n: int) -> list[tuple[str, str]]:
        """NL question about what a codon encodes -> answer."""
        templates_q = [
            "what amino acid does the codon {codon} encode",
            "what does {codon} code for",
            "which amino acid is encoded by {codon}",
            "{codon} encodes what amino acid",
            "what is the amino acid for codon {codon}",
            "translate the codon {codon}",
            "what protein residue does {codon} specify",
            "the codon {codon} codes for which amino acid",
            "identify the amino acid encoded by {codon}",
            "what amino acid corresponds to {codon}",
        ]
        templates_a = [
            "{codon} encodes {full_name} {one_letter}",
            "the codon {codon} codes for {full_name} {one_letter}",
            "{codon} translates to {full_name} which is represented by {one_letter}",
            "{full_name} {one_letter} is encoded by {codon}",
            "the amino acid is {full_name} {one_letter}",
        ]
        pairs = []
        codons = list(CODON_TO_AMINO.keys())
        for _ in range(n):
            codon = self.rng.choice(codons)
            info = CODON_TO_AMINO[codon]
            if info["type"] == "stop":
                q = self.rng.choice(templates_q).format(codon=codon)
                a = f"{codon} is a stop codon it signals the end of translation"
            else:
                data = {"codon": codon, "full_name": info["full_name"].lower(),
                        "one_letter": info["one_letter"], "three_letter": info["three_letter"].lower()}
                q = self.rng.choice(templates_q).format(**data)
                a = self.rng.choice(templates_a).format(**data)
            pairs.append((q, a))
        return pairs

    def _gen_sequence_translation_pairs(self, n: int) -> list[tuple[str, str]]:
        """NL request to translate RNA sequence -> protein sequence."""
        templates_q = [
            "translate the rna sequence {seq}",
            "what protein does {seq} encode",
            "convert {seq} to amino acids",
            "translate {seq} to protein",
            "what is the protein sequence for {seq}",
            "decode the rna {seq}",
            "what amino acid sequence does {seq} produce",
        ]
        templates_a = [
            "the protein sequence is {protein}",
            "{seq} translates to {protein}",
            "the amino acids are {protein}",
            "{protein}",
        ]
        pairs = []
        for _ in range(n):
            # Random sequence length 1-8 codons + stop
            length = self.rng.randint(1, 8)
            codons = [self.rng.choice(self.coding_codons) for _ in range(length)]
            # Optionally add a stop codon
            if self.rng.random() < 0.5:
                codons.append(self.rng.choice(STOP_CODONS))
            seq = " ".join(codons)
            protein = translate_rna_to_protein("".join(codons))
            data = {"seq": seq, "protein": protein}
            q = self.rng.choice(templates_q).format(**data)
            a = self.rng.choice(templates_a).format(**data)
            pairs.append((q, a))
        return pairs

    def _gen_dna_to_rna_pairs(self, n: int) -> list[tuple[str, str]]:
        """DNA to RNA transcription pairs."""
        templates_q = [
            "transcribe {dna} to rna",
            "what is the rna for dna sequence {dna}",
            "convert {dna} from dna to rna",
            "transcribe the dna sequence {dna}",
            "what rna sequence does {dna} produce",
        ]
        templates_a = [
            "the rna sequence is {rna}",
            "{dna} transcribes to {rna}",
            "{rna}",
        ]
        pairs = []
        for _ in range(n):
            length = self.rng.randint(3, 18)
            # Make length a multiple of 3 sometimes
            if self.rng.random() < 0.7:
                length = (length // 3) * 3
                if length == 0:
                    length = 3
            dna = "".join(self.rng.choice(DNA_BASES) for _ in range(length))
            rna = translate_dna_to_rna(dna)
            data = {"dna": dna, "rna": rna}
            q = self.rng.choice(templates_q).format(**data)
            a = self.rng.choice(templates_a).format(**data)
            pairs.append((q, a))
        return pairs

    def _gen_reverse_complement_pairs(self, n: int) -> list[tuple[str, str]]:
        """Reverse complement pairs."""
        templates_q = [
            "what is the reverse complement of {seq}",
            "reverse complement {seq}",
            "find the complementary strand of {seq}",
            "what is the complement of {seq} in reverse",
        ]
        pairs = []
        for _ in range(n):
            # DNA or RNA
            if self.rng.random() < 0.6:
                bases = DNA_BASES
                comp = COMPLEMENT_DNA
                seq_type = "dna"
            else:
                bases = RNA_BASES
                comp = COMPLEMENT_RNA
                seq_type = "rna"
            length = self.rng.randint(3, 15)
            seq = "".join(self.rng.choice(bases) for _ in range(length))
            rev_comp = "".join(comp[b] for b in reversed(seq))
            q = self.rng.choice(templates_q).format(seq=seq)
            a = f"the reverse complement is {rev_comp}"
            pairs.append((q, a))
        return pairs

    def _gen_amino_acid_property_pairs(self, n: int) -> list[tuple[str, str]]:
        """Questions about amino acid properties."""
        aa_list = list(AMINO_ACID_PROPERTIES.keys())
        templates = [
            ("is {name} polar", "{name} is {polarity}"),
            ("is {name} hydrophobic", "{name} is {hydrophobicity}"),
            ("what is the charge of {name}", "{name} has {charge} charge"),
            ("what type of amino acid is {name}", "{name} is a {classification} amino acid"),
            ("what is the molecular weight of {name}", "{name} has a molecular weight of {mw}"),
            ("is {name} charged", "{name} is {charge}"),
            ("what group does {name} belong to", "{name} belongs to the {classification} group"),
            ("describe {name}", "{name} is a {polarity} {classification} amino acid with {charge} charge"),
        ]
        pairs = []
        for _ in range(n):
            aa = self.rng.choice(aa_list)
            props = AMINO_ACID_PROPERTIES[aa]
            data = {
                "name": props["full_name"].lower(),
                "polarity": props["polarity"],
                "hydrophobicity": props["hydrophobicity"],
                "charge": props["charge"],
                "classification": props["classification"].replace("_", " "),
                "mw": str(round(props["molecular_weight"], 1)),
            }
            q_template, a_template = self.rng.choice(templates)
            q = q_template.format(**data)
            a = a_template.format(**data)
            pairs.append((q, a))
        return pairs

    def _gen_sequence_explain_pairs(self, n: int) -> list[tuple[str, str]]:
        """Genetic sequence -> NL explanation."""
        pairs = []
        for _ in range(n):
            length = self.rng.randint(1, 6)
            codons = [self.rng.choice(self.coding_codons) for _ in range(length)]
            add_stop = self.rng.random() < 0.4
            if add_stop:
                codons.append(self.rng.choice(STOP_CODONS))

            seq = " ".join(codons)
            amino_acids = []
            for c in codons:
                info = CODON_TO_AMINO[c]
                if info["type"] == "stop":
                    amino_acids.append(("stop codon", info["full_name"].lower()))
                else:
                    amino_acids.append((info["full_name"].lower(), info["one_letter"]))

            # Build NL explanation
            explanations = []
            if self.rng.random() < 0.5:
                prefix = "this rna sequence encodes"
            else:
                prefix = "the sequence translates to"

            aa_descriptions = []
            for i, (name, code) in enumerate(amino_acids):
                if name == "stop codon":
                    aa_descriptions.append("then ends with a stop codon")
                elif i == 0:
                    aa_descriptions.append(name)
                else:
                    aa_descriptions.append(f"followed by {name}")

            explanation = prefix + " " + " ".join(aa_descriptions)
            pairs.append((seq, explanation))
        return pairs

    def _gen_protein_describe_pairs(self, n: int) -> list[tuple[str, str]]:
        """Protein sequence -> NL description."""
        aa_list = [aa for aa in AMINO_ACID_PROPERTIES.keys()]
        pairs = []
        for _ in range(n):
            length = self.rng.randint(1, 6)
            aas = [self.rng.choice(aa_list) for _ in range(length)]
            seq = " ".join(aas)

            names = [AMINO_ACID_PROPERTIES[aa]["full_name"].lower() for aa in aas]
            if len(names) == 1:
                desc = f"this protein contains {names[0]}"
            elif len(names) == 2:
                desc = f"this protein contains {names[0]} and {names[1]}"
            else:
                desc = f"this protein starts with {names[0]}"
                for name in names[1:-1]:
                    desc += f" followed by {name}"
                desc += f" and ends with {names[-1]}"

            pairs.append((seq, desc))
        return pairs

    def _gen_reading_frame_pairs(self, n: int) -> list[tuple[str, str]]:
        """Reading frame analysis pairs."""
        pairs = []
        for _ in range(n):
            # Generate an RNA sequence that's at least 9 bases
            n_codons = self.rng.randint(3, 7)
            codons = [self.rng.choice(self.coding_codons) for _ in range(n_codons)]
            seq = "".join(codons)

            frame = self.rng.randint(0, 2)
            q = f"what is reading frame {frame + 1} of {seq}"

            # Extract codons in the given reading frame
            frame_seq = seq[frame:]
            frame_codons = []
            for i in range(0, len(frame_seq) - 2, 3):
                frame_codons.append(frame_seq[i:i+3])

            if frame_codons:
                protein = "".join(
                    CODON_TO_AMINO.get(c, {"one_letter": "?"})["one_letter"]
                    for c in frame_codons
                )
                codon_str = " ".join(frame_codons)
                a = f"reading frame {frame + 1} gives codons {codon_str} which translates to {protein}"
            else:
                a = f"reading frame {frame + 1} does not contain complete codons"

            pairs.append((q, a))
        return pairs

    def _gen_start_stop_pairs(self, n: int) -> list[tuple[str, str]]:
        """Questions about start and stop codons."""
        templates = [
            ("what is the start codon", "the start codon is AUG which encodes methionine"),
            ("what are the stop codons", "the stop codons are UAA UAG and UGA"),
            ("does {seq} contain a start codon", "{has_start}"),
            ("does {seq} contain a stop codon", "{has_stop}"),
            ("is {codon} a start codon", "{is_start}"),
            ("is {codon} a stop codon", "{is_stop}"),
            ("what does the start codon encode", "the start codon AUG encodes methionine M"),
            ("how many stop codons are there", "there are three stop codons UAA UAG and UGA"),
        ]
        pairs = []
        codons_list = list(CODON_TO_AMINO.keys())
        for _ in range(n):
            template_q, template_a = self.rng.choice(templates)
            if "{seq}" in template_q:
                length = self.rng.randint(2, 6)
                codons = [self.rng.choice(codons_list) for _ in range(length)]
                seq = " ".join(codons)
                has_start = "yes it contains the start codon AUG" if "AUG" in codons else "no it does not contain the start codon AUG"
                has_stop = "yes" if any(c in STOP_CODONS for c in codons) else "no it does not contain a stop codon"
                q = template_q.format(seq=seq)
                a = template_a.format(has_start=has_start, has_stop=has_stop)
            elif "{codon}" in template_q:
                codon = self.rng.choice(codons_list)
                is_start = "yes AUG is the start codon" if codon == "AUG" else f"no {codon} is not a start codon"
                is_stop = f"yes {codon} is a stop codon" if codon in STOP_CODONS else f"no {codon} is not a stop codon"
                q = template_q.format(codon=codon)
                a = template_a.format(is_start=is_start, is_stop=is_stop)
            else:
                q = template_q
                a = template_a
            pairs.append((q, a))
        return pairs
