"""Shared helpers for extracting protein sequences from PDB files.

Used by methods (chai, alphafold3, ...) that need the per-chain amino-acid
sequence as input. Extracted from chai_inference.py to avoid duplication.
"""

from typing import List

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HSD": "H", "HSE": "H", "HSP": "H", "HIE": "H", "HID": "H",
    "MSE": "M", "SEC": "U",
}


def extract_protein_sequence(pdb_path: str) -> List[str]:
    """Return one amino-acid sequence string per chain in the PDB file.

    Sequences are read from the CA atoms (one residue per CA), with insertions
    de-duplicated. Unknown residue codes are encoded as ``X``.
    """
    from biopandas.pdb import PandasPdb

    ppdb = PandasPdb().read_pdb(pdb_path)
    atoms = ppdb.df["ATOM"]
    ca = atoms[atoms["atom_name"] == "CA"].drop_duplicates(
        subset=["chain_id", "residue_number", "insertion"]
    )
    sequences: List[str] = []
    for _chain_id, group in ca.groupby("chain_id", sort=False):
        seq = "".join(AA_3TO1.get(r, "X") for r in group["residue_name"])
        if seq:
            sequences.append(seq)
    return sequences
