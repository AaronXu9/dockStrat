"""
Helper script for SurfDock surface computation.

Replicates computeTargetMesh_test_samples.py logic but uses
computeAPBS_plinder.py instead of computeAPBS.py to fix the
temp-directory handling bug in the original APBS wrapper.

Usage:
    python _surfdock_surface_helper.py \
        --data_dir /path/to/data_dir \
        --out_dir  /path/to/surface_out_dir \
        [--surfdock_dir /path/to/SurfDock]
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",     required=True)
parser.add_argument("--out_dir",      required=True)
parser.add_argument("--surfdock_dir", default="/home/aoxu/projects/SurfDock")
args = parser.parse_args()

# Make the SurfDock surface module importable
surface_code_dir = os.path.join(args.surfdock_dir, "comp_surface", "prepare_target")
sys.path.insert(0, surface_code_dir)
os.chdir(surface_code_dir)   # masif_opts does os.environ lookups from here

import numpy as np
import shutil
import glob
import pymesh
import Bio.PDB
from Bio.PDB import *
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")
from IPython.utils import io
from sklearn.neighbors import KDTree
from scipy.spatial import distance
from tqdm import tqdm
from joblib import delayed, Parallel

from default_config.masif_opts import masif_opts
from compute_normal import compute_normal
from computeAPBS_plinder import computeAPBS   # <-- fixed version
from computeCharges import computeCharges, assignChargesToNewMesh
from computeHydrophobicity import computeHydrophobicity
from computeMSMS import computeMSMS
from fixmesh import fix_mesh
from save_ply import save_ply
from mol2graph import mol_to_nx


def compute_inp_surface(target_filename, ligand_filename, out_dir=None, dist_threshold=8):
    try:
        sufix = "_" + str(dist_threshold) + "A.pdb"
        if out_dir is not None:
            out_filename = os.path.join(out_dir, ligand_filename.split("/")[-2])
            os.makedirs(out_filename, exist_ok=True)
            sufix = "/" + os.path.splitext(target_filename)[0].split("/")[-1] + "_" + str(dist_threshold) + "A.pdb"
        else:
            out_filename = os.path.splitext(ligand_filename)[0]

        if os.path.exists(out_filename + f"/{sufix.split('.')[0]}.ply"):
            print("Already done, skipping:", out_filename)
            return 0

        input_filename = os.path.splitext(target_filename)[0]

        if ligand_filename.endswith(".sdf"):
            mol = Chem.SDMolSupplier(ligand_filename, sanitize=False)[0]
        elif ligand_filename.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(ligand_filename, sanitize=False)
        g = mol_to_nx(mol)
        atomCoords = np.array([g.nodes[i]["pos"].tolist() for i in g.nodes])

        parser_pdb = Bio.PDB.PDBParser(QUIET=True)
        structures = parser_pdb.get_structure("target", input_filename + ".pdb")
        structure = structures[0]
        atoms = Bio.PDB.Selection.unfold_entities(structure, "A")
        ns = Bio.PDB.NeighborSearch(atoms)

        close_residues = []
        for a in atomCoords:
            close_residues.extend(ns.search(a, dist_threshold, level="R"))
        close_residues = Bio.PDB.Selection.uniqueify(close_residues)

        class SelectNeighbors(Select):
            def accept_residue(self, residue):
                if residue in close_residues:
                    if all(a in [i.get_name() for i in residue.get_unpacked_list()]
                           for a in ["N", "CA", "C", "O"]) or residue.resname == "HOH":
                        return True
                return False

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(out_filename + sufix, SelectNeighbors())

        with io.capture_output():
            vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename + sufix)

        # computeCharges appends ".pdb" internally, so strip extension first
        pocket_pdb_base = os.path.splitext(out_filename + sufix)[0]
        vertex_hbond = computeCharges(pocket_pdb_base, vertices1, names1)
        # Hydrophobicity
        vertex_hphobicity = computeHydrophobicity(names1)

        if len(vertices1) == 0:
            return target_filename

        # Fix mesh
        mesh = pymesh.form_mesh(vertices1, faces1)
        regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])

        # Check mesh has enough connectivity
        try:
            mesh2, info = pymesh.remove_isolated_vertices(mesh)
            vertices2 = mesh2.vertices
            faces2 = mesh2.faces
        except Exception:
            vertices2 = vertices1
            faces2 = faces1

        mesh = pymesh.form_mesh(vertices2, faces2)
        faces_to_keep = np.arange(len(faces2))
        mesh = pymesh.submesh(mesh, faces_to_keep, 0)
        with io.capture_output():
            regular_mesh = fix_mesh(mesh, masif_opts["mesh_res"])

        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

        if masif_opts["use_hbond"]:
            vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)
        if masif_opts["use_hphob"]:
            vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)
        if masif_opts["use_apbs"]:
            vertex_charges = computeAPBS(regular_mesh.vertices, out_filename + sufix, out_filename + "_temp")

        regular_mesh.add_attribute("vertex_mean_curvature")
        H = regular_mesh.get_attribute("vertex_mean_curvature")
        regular_mesh.add_attribute("vertex_gaussian_curvature")
        K = regular_mesh.get_attribute("vertex_gaussian_curvature")
        elem = np.square(H) - K
        elem[elem < 0] = 1e-8
        k1 = H + np.sqrt(elem)
        k2 = H - np.sqrt(elem)
        si = (k1 + k2) / (k1 - k2)
        si = np.arctan(si) * (2 / np.pi)

        ply_path = out_filename + f"/{sufix.split('.')[0]}.ply"
        save_ply(ply_path, regular_mesh.vertices, regular_mesh.faces,
                 normals=vertex_normal, charges=vertex_charges,
                 normalize_charges=True, hbond=vertex_hbond,
                 hphob=vertex_hphobicity, si=si)
        return 0

    except Exception as e:
        print(f"[WARN] Surface computation failed for {target_filename}: {e}")
        return target_filename


if __name__ == "__main__":
    os.makedirs(args.out_dir, exist_ok=True)

    args_list = []
    for protein_dir in tqdm(os.listdir(args.data_dir)):
        target = os.path.join(args.data_dir, protein_dir, f"{protein_dir}_protein_processed.pdb")
        ligand = os.path.join(args.data_dir, protein_dir, f"{protein_dir}_ligand.sdf")
        if not os.path.exists(ligand):
            ligand = os.path.join(args.data_dir, protein_dir, f"{protein_dir}_ligand.mol2")
        args_list.append((target, ligand))

    print(f"Processing {len(args_list)} systems")
    results = Parallel(n_jobs=1, backend="multiprocessing")(
        delayed(compute_inp_surface)(t, l, args.out_dir, dist_threshold=8)
        for t, l in tqdm(args_list)
    )
    failed = [r for r in results if r != 0]
    if failed:
        print(f"[WARN] Failed for {len(failed)} systems:", failed)

    # Clean up temp files
    for f in glob.glob(os.path.join(args.out_dir, "*_temp*")) + \
             glob.glob(os.path.join(args.out_dir, "*msms*")):
        try:
            os.remove(f)
        except Exception:
            pass
