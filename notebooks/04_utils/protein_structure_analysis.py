"""
Structure-based clustering:
1) For each pair of PDB files, run TM-align (or an equivalent) to get RMSD or TM-score.
2) Build an NxN distance matrix.
3) Perform hierarchical clustering or another method.
"""

import os
import subprocess
import glob
import numpy as np
import re
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def run_tmalign(pdbA, pdbB, tmalign_bin="TMalign"):
    """
    Run TM-align and parse RMSD or TM-score from stdout.
    Returns (rmsd, tm_score).
    """
    cmd = [tmalign_bin, pdbA, pdbB]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()

    rmsd_val = None
    tm_val = None
    for line in lines:
        if line.startswith("Aligned length"):
            # e.g. "RMSD of the common residues=   2.12"
            match = re.search(r"RMSD=\s*([\d\.]+)", line)
            if match:
                rmsd_str = match.group(1)  # e.g. "4.60"
                rmsd_val = float(rmsd_str)
                print("RMSD value is:", rmsd_val)
            else:
                print("No RMSD value found in the line.")
        elif line.startswith("TM-score="):
            # e.g. "TM-score= 0.45"
            parts = line.split("=")
            tm_val = float(parts[1].split()[0])
    return rmsd_val, tm_val

def load_pdb(pdb_dir, selected_proteins=None, cropped=''):
    """
    retrieve the path for the pdb files in the pdb directory
    """
    protein_names = os.listdir(pdb_dir)
    if selected_proteins is not None:
        protein_paths = [os.path.join(pdb_dir, protein_name, f"{protein_name}_protein_{cropped}.pdb") for protein_name in selected_proteins]
    else:
        protein_paths = [os.path.join(pdb_dir, protein_name, f"{protein_name}_protein_{cropped}.pdb") for protein_name in protein_names]

    return protein_paths

def build_structure_distance_matrix(pdb_dir, selected_proteins=None, cropped=''):
    """
    For each PDB in pdb_dir, run pairwise TMalign, store RMSD as distance.
    """
    names = os.listdir(pdb_dir)
    if selected_proteins is not None:
        pdb_paths = [os.path.join(pdb_dir, protein_name, f"{protein_name}_protein{cropped}.pdb") for protein_name in selected_proteins]
        names = selected_proteins
    else:
        pdb_paths = [os.path.join(pdb_dir, protein_name, f"{protein_name}_protein{cropped}.pdb") for protein_name in names]
    # names = [os.path.splitext(os.path.basename(p))[0] for p in pdb_paths]
    n = len(pdb_paths)

    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            print(f"Comparing {pdb_paths[i]} vs {pdb_paths[j]}")
            rmsd, tm = run_tmalign(pdb_paths[i], pdb_paths[j])
            dist = rmsd  # or (1 - tm), if you prefer to cluster by TM-score
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return names, dist_mat

def cluster_structures(pdb_dir, selected_proteins=None, cropped=''):
    names, dist_mat = build_structure_distance_matrix(pdb_dir, selected_proteins, cropped)

    dist_condensed = squareform(dist_mat, checks=False)
    Z = linkage(dist_condensed, method="average")
    plt.figure(figsize=(8, 4))
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.title("Structure-based clustering (RMSD average linkage)")
    plt.show()


# Example usage:
failed_proteins = ['7WL4_JFU', '7WPW_F15', '7XFA_D9J', '7NU0_DCL', '7VB8_STL',
       '8C5M_MTA', '7Z1Q_NIO', '7QE4_NGA', '7BJJ_TVW', '7UYB_OK0',
       '7PL1_SFG', '7SDD_4IP', '6YR2_T1C', '8B8H_OJQ', '7NP6_UK8',
       '7UQ3_O2U', '6TW7_NZB', '7ZOC_T8E', '8BOM_QU6', '7MY1_IPE']
pdb_dir = "/Users/aoxu/projects/DrugDiscovery/PoseBench/data/posebusters_benchmark_set/"
cluster_structures(pdb_dir, selected_proteins=failed_proteins, cropped='')