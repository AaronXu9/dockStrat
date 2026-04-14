"""Unit tests for the shared protein-sequence utility."""

import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PROTEIN = os.path.join(
    PROJECT_ROOT, "data", "runsNposes", "8gkf__1__1.A__1.J",
    "8gkf__1__1.A__1.J_protein.pdb",
)


class TestExtractProteinSequence:
    def test_returns_list_of_nonempty_sequences(self):
        from dockstrat.utils.sequence import extract_protein_sequence

        if not os.path.exists(FIXTURE_PROTEIN):
            pytest.skip(f"Fixture not found: {FIXTURE_PROTEIN}")

        seqs = extract_protein_sequence(FIXTURE_PROTEIN)
        assert isinstance(seqs, list)
        assert len(seqs) >= 1
        for s in seqs:
            assert isinstance(s, str)
            assert len(s) > 0
            # All characters must be in the canonical 1-letter AA alphabet (or X for unknown)
            assert set(s).issubset(set("ACDEFGHIKLMNPQRSTVWYUX"))

    def test_aa_3to1_covers_standard_residues(self):
        from dockstrat.utils.sequence import AA_3TO1

        for three, one in [
            ("ALA", "A"), ("ARG", "R"), ("GLY", "G"), ("LYS", "K"),
            ("VAL", "V"), ("HIS", "H"), ("HSD", "H"), ("MSE", "M"),
        ]:
            assert AA_3TO1[three] == one
