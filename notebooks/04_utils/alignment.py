#!/usr/bin/env python

import os
import numpy as np
import tempfile

import MDAnalysis as mda
from MDAnalysis.analysis import align

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdFMCS
import numpy as np

import subprocess

def align_protein_ligand_complex(ref_prot_pdb, pred_prot_pdb,
                                 pred_lig_sdf, out_prot_pdb, out_lig_sdf):
    """
    1. Attempt direct rotation_matrix() on CA atoms to align predicted => reference.
       - If we catch a shape mismatch error, do a fallback:
         a) Use align.alignto() with a sequence/backbone-based alignment.
         b) Compute a 4x4 transform from the pre- vs. post-alignment coordinates.
    2. Apply the resulting transform to *all* predicted protein atoms and the predicted ligand.
    3. Write aligned results to PDB/SDF.
    """

    # --- Load proteins into MDAnalysis ---
    ref_u = mda.Universe(ref_prot_pdb)
    pred_u = mda.Universe(pred_prot_pdb)

    # Selections for alignment
    ref_ca = ref_u.select_atoms("name CA and protein")
    pred_ca = pred_u.select_atoms("name CA and protein")

    # We'll store the original predicted coords so that if we do a fallback alignment
    # (alignto), we can figure out how the structure moved overall.
    old_coords_all = pred_u.atoms.positions.copy()

    # Try direct rotation_matrix with CA atoms
    try:
        R, rmsd_value = align.rotation_matrix(pred_ca.positions, ref_ca.positions)
        print(f"[Direct CA-based alignment] RMSD: {rmsd_value:.3f} Å")

        # Compute centers of mass
        ref_com = ref_ca.center_of_mass()
        pred_com = pred_ca.center_of_mass()

    except ValueError as e:
        if "must have same shape" in str(e):
            print("[Fallback] Shape mismatch in CA sets. Attempting sequence-based alignment via align.alignto()...")
            # # Perform sequence/backbone alignment. This modifies pred_u in memory.
            # # align.alignto() returns RMSD as a float.
            # rmsd_value = align.alignto(mobile=pred_u,
            #                            reference=ref_u,
            #                            select='backbone')
            # print(f"[Sequence-based alignment] RMSD: {rmsd_value:.3f} Å")

            # # Now pred_u is aligned in memory. We need to figure out the global transform.
            # # We can do that by comparing old_coords_all to new_coords_all for the same atom indices.
            # new_coords_all = pred_u.atoms.positions.copy()

            # # Let's pick an arbitrary set of atoms (all protein) to compute a best-fit rotation.
            # # We'll do the same approach with rotation_matrix:
            # R, rmsd_val2 = align.rotation_matrix(old_coords_all, new_coords_all)
            # print(f"[Post sequence-based transform] RMSD(whole-protein): {rmsd_val2:.3f} Å")

            # # We also compute COM shift for these atoms (old vs. new).
            # pred_com = old_coords_all.mean(axis=0)  # or center_of_mass via a smaller selection
            # ref_com = new_coords_all.mean(axis=0)
        else:
            # Some other ValueError
            raise e

    # Build a 4×4 homogeneous transform
    #   final_pos = (pos - pred_com) * R + ref_com
    t = ref_com - np.dot(pred_com, R)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3]  = t

    # Now we apply 'transform_matrix' to *all* atoms in pred_u
    # But note: if we used align.alignto in fallback, pred_u might already be in the aligned position.
    # We'll re-apply it only if we used the direct approach or we want a consistent approach.
    # For clarity, let's always apply the matrix to the original coords.
    pred_u.atoms.positions = (np.hstack([old_coords_all, 
                                         np.ones((old_coords_all.shape[0], 1))]) @ transform_matrix.T)[:, :3]

    # Write out aligned predicted protein
    pred_u.atoms.write(out_prot_pdb)

    # --- Apply to the predicted ligand (RDKit) ---
    lig_mol = Chem.MolFromMolFile(pred_lig_sdf, removeHs=False)
    if lig_mol is None:
        print(f"ERROR: Could not read ligand SDF: {pred_lig_sdf}")
        return

    conf = lig_mol.GetConformer()
    if not conf.Is3D():
        print(f"WARNING: {pred_lig_sdf} doesn't appear to have 3D coords.")

    for i in range(lig_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords_hom = np.array([pos.x, pos.y, pos.z, 1.0])
        new_coords_hom = coords_hom @ transform_matrix.T
        conf.SetAtomPosition(
            i,
            Chem.rdGeometry.Point3D(new_coords_hom[0], 
                                    new_coords_hom[1],
                                    new_coords_hom[2])
        )

    # Write out the newly aligned ligand
    Chem.MolToMolFile(lig_mol, out_lig_sdf, includeStereo=True)

def align_ligand_to_ref(pred_lig_sdf, ref_lig_sdf, out_aligned_sdf):
    # Load reference ligand
    ref_mol = Chem.MolFromMolFile(ref_lig_sdf, removeHs=False)
    # Load predicted ligand
    pred_mol = Chem.MolFromMolFile(pred_lig_sdf, removeHs=False)

    if ref_mol is None or pred_mol is None:
        print("Error: Could not read input SDF(s).")
        return

    # Usually, you'd want to ensure both have 3D coordinates
    # For best results, they should be the same compound or at least share a substructure.
    # RDKit needs matching atom order or a substructure match to do the alignment well.

    # Generate an MMFF or UFF force field for the reference if needed
    # e.g., AllChem.EmbedMolecule(ref_mol)

    # The simplest approach: AlignMol (pred, ref) modifies pred in place.
    # This returns the RMSD (float)
    rmsd_value = AllChem.AlignMol(pred_mol, ref_mol)  
    print(f"Ligand alignment RMSD: {rmsd_value:.3f} Å")

    # Write out the newly aligned predicted ligand
    Chem.MolToMolFile(pred_mol, out_aligned_sdf, includeStereo=True)

def compute_ligand_rmsd(ref_lig_sdf, aligned_pred_lig_sdf):
    ref_mol = Chem.MolFromMolFile(ref_lig_sdf, removeHs=False)
    pred_mol = Chem.MolFromMolFile(aligned_pred_lig_sdf, removeHs=False)
    
    if ref_mol is None or pred_mol is None:
        print("Error reading SDF(s).")
        return None
    
    # If both are the same molecule (same connectivity, same # atoms),
    # we can do direct RMSD. If not, you might need substructure matching.
    rmsd_val = AllChem.GetBestRMS(ref_mol, pred_mol)
    print(f"Ligand RMSD (same frame): {rmsd_val:.3f} Å")
    return rmsd_val

def align_with_pymol(ref_prot, ref_lig, pred_prot, pred_lig, out_prot, out_lig):
    """
    Generate a PyMOL script on the fly to:
      1) Load reference protein (optionally ref ligand if present).
      2) Load predicted protein & ligand as separate objects.
      3) Align predicted protein to reference protein (typically by CA atoms).
      4) Copy the same transformation matrix from predicted protein to the ligand.
      5) Save the aligned protein and ligand.
    Then run the script in headless PyMOL.
    """

    # Build the PyMOL commands as a multi-line string
    # Note: we use triple quotes for easy multi-line
    pymol_script = f"""
load {ref_prot}, refProt
"""

    # If the reference ligand file actually exists, load it
    if os.path.isfile(ref_lig):
        pymol_script += f"load {ref_lig}, refLig\n"

    pymol_script += f"""
load {pred_prot}, predProt
load {pred_lig}, predLig

# Align predicted protein to reference protein (focusing on CA atoms)
align predProt and name CA, refProt and name CA

# Apply the same transform from predProt to predLig
matrix_copy predProt, predLig

# Save the aligned protein and ligand
save {out_prot}, predProt
save {out_lig}, predLig

quit
"""
    with tempfile.NamedTemporaryFile(suffix=".pml", delete=False, mode="w") as script_file:
        script_file.write(pymol_script)
        script_path = script_file.name

    # Now we run PyMOL in headless mode with the above script (via stdin).
    pymol_exec = "/Applications/PyMOL.app/Contents/MacOS/PyMOL"
    process = subprocess.Popen(
        [pymol_exec, "-cq", script_path],             # 'pymol' must be on your PATH or replace with full path
        # stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True                     # So we can pass a string instead of bytes
    )

    # Send the script into PyMOL
    out, err = process.communicate(pymol_script)
    
    # Check for errors
    if process.returncode != 0:
        print(f"[ERROR] PyMOL alignment failed for {pred_prot} and {pred_lig}:\n{err}")
    else:
        print(f"[INFO] Aligned {pred_prot} => {out_prot}\n        {pred_lig} => {out_lig}")
        if out.strip():
            print("=== PyMOL STDOUT ===")
            print(out.strip())
        if err.strip():
            print("=== PyMOL STDERR ===")
            print(err.strip())

def load_first_mol(sdf_file):
    """Return the first molecule in an SDF (with hydrogens removed)."""
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    if not suppl or suppl[0] is None:
        raise ValueError(f"Could not read a molecule from {sdf_file}")
    return Chem.RemoveHs(suppl[0])

def rmsd_on_mcs(mol1, mol2):
    """Compute RMSD using Maximum Common Substructure alignment."""
    mcs = rdFMCS.FindMCS(
        [mol1, mol2],
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10)
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    idx1 = mol1.GetSubstructMatch(mcs_mol)
    idx2 = mol2.GetSubstructMatch(mcs_mol)
    if not idx1 or not idx2:
        raise RuntimeError("No common substructure found.")
    atom_map = list(zip(idx2, idx1))  # (probe, ref)
    rmsd_val = rdMolAlign.AlignMol(mol2, mol1, atomMap=atom_map)
    return rmsd_val, len(atom_map)

print("Imports and utility functions loaded successfully!")

def main_mda_loop():
    """
    Loop over a directory structure, performing alignment on predicted
    protein-ligand pairs. Adjust directory paths as needed.
    """
    base_dir = "/Users/aoxu/projects/DrugDiscovery/PoseBench/forks/chai-lab/inference/chai-lab_plinder_outputs_0"
    data_dir = "/Users/aoxu/projects/DrugDiscovery/PoseBench/data/plinder_set"

    for protein_name in os.listdir(base_dir):
        protein_dir = os.path.join(base_dir, protein_name)
        if not os.path.isdir(protein_dir):
            continue

        ref_prot_pdb = os.path.join(data_dir, protein_name, f"{protein_name}_protein.pdb")
        ref_lig_sdf = os.path.join(data_dir, protein_name, f"{protein_name}_ligand.sdf")

        for model_idx in range(0, 5):
            pred_prot_pdb = os.path.join(protein_dir, f"pred.model_idx_{model_idx}_protein.pdb")
            pred_lig_sdf = os.path.join(protein_dir, f"pred.model_idx_{model_idx}_ligand.sdf")
            out_prot_pdb = os.path.join(protein_dir, f"pred.model_idx_{model_idx}_protein_aligned.pdb")
            out_lig_sdf = os.path.join(protein_dir, f"pred.model_idx_{model_idx}_ligand_aligned.sdf")

            if not (os.path.exists(pred_prot_pdb) and os.path.exists(pred_lig_sdf)):
                continue
            # # 1) Align entire complex
            # align_protein_ligand_complex(
            #     ref_prot_pdb, pred_prot_pdb,
            #     pred_lig_sdf, out_prot_pdb, out_lig_sdf
            # )

            # # 2) Compute ligand RMSD
            # compute_ligand_rmsd(ref_lig_sdf, out_lig_sdf)

            # 3) Align ligand to reference
            align_with_pymol(
                ref_prot=ref_prot_pdb,
                ref_lig=ref_lig_sdf,
                pred_prot=pred_prot_pdb,
                pred_lig=pred_lig_sdf,
                out_prot=out_prot_pdb,
                out_lig=out_lig_sdf
            )

if __name__ == "__main__":
    main_mda_loop()