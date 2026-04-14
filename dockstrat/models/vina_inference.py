import os
import subprocess
import sys
import time
import logging
from omegaconf import OmegaConf
from rdkit import Chem
import pandas as pd
import numpy as np
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from dockstrat.utils.log import get_custom_logger

# # Configure logging: logs will be written to 'docking.log'
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.FileHandler('gnina_timing.log'),
#         logging.StreamHandler()  # optional: this sends log output to the console as well
#     ]
# )

# def get_custom_logger(logger_name: str, log_filename: str) -> logging.Logger:
#     """
#     Returns a logger configured with a FileHandler that writes to log_filename.
#     Any existing handlers will be removed.
#     """
#     logger = logging.getLogger(logger_name)
#     # Remove any existing handlers attached to this logger
#     if logger.hasHandlers():
#         logger.handlers.clear()
    
#     logger.setLevel(logging.INFO)
    
#     # Create file handler with the custom filename
#     file_handler = logging.FileHandler(log_filename)
#     file_handler.setLevel(logging.INFO)
    
#     # Define a formatter and set it for the file handler
#     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
#     file_handler.setFormatter(formatter)
    
#     logger.addHandler(file_handler)
    
#     # Optionally, also log to the console by adding a StreamHandler
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(formatter)
#     logger.addHandler(console_handler)
    
#     return logger


def compute_ligand_center_and_size(ligand_sdf, protein_pdb=None):
    """
    IMPROVED VERSION: Uses protein-ligand spatial relationship for better binding site detection.
    
    Parses ligand coordinates from an SDF file and optionally uses protein structure
    to find a more accurate binding site center. This fixes the issue where docked
    ligands were positioned 20+ Å away from reference ligands.
    
    Returns the center (center_x, center_y, center_z)
    and size (size_x, size_y, size_z) suitable for defining
    the AutoDock Vina box.
    
    Args:
        ligand_sdf: Path to ligand SDF file
        protein_pdb: Optional path to protein PDB file for improved binding site detection
    """
    # Use RDKit for reliable SDF parsing (similar to sdf_to_inchikey_molbin approach)
    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
    mol = next(iter(suppl))  # Get first molecule
    if mol is None:
        raise ValueError(f"Could not load molecule from {ligand_sdf}")
    
    # Get conformer and extract coordinates (similar to conformer_centroid approach)
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=float)
    
    # Extract x, y, z coordinates for size calculation
    x_coords = coords[:, 0]
    y_coords = coords[:, 1] 
    z_coords = coords[:, 2]
    
    # Calculate ligand center of mass (mean of all atom positions)
    ligand_center = coords.mean(0)  # Same as conformer_centroid
    ligand_center_x, ligand_center_y, ligand_center_z = ligand_center
    ligand_coords = coords

    # Calculate bounding box for size calculation
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)

    # IMPROVED: Use protein-ligand spatial relationship if protein is provided
    if protein_pdb is not None:
        try:
            import MDAnalysis as mda
            u = mda.Universe(protein_pdb)
            protein_coords = u.atoms.positions
            
            # Find closest protein atoms to reference ligand (midpoint method - most accurate)
            distances_to_protein = []
            for atom_pos in protein_coords:
                min_dist = np.min(np.linalg.norm(ligand_coords - atom_pos, axis=1))
                distances_to_protein.append(min_dist)
            
            # Get 50 closest protein atoms and find their center
            closest_indices = np.argsort(distances_to_protein)[:50]
            closest_protein_atoms = protein_coords[closest_indices]
            closest_protein_center = np.mean(closest_protein_atoms, axis=0)
            
            # Use midpoint between ligand and closest protein atoms as binding site
            center_x = (ligand_center[0] + closest_protein_center[0]) / 2
            center_y = (ligand_center[1] + closest_protein_center[1]) / 2  
            center_z = (ligand_center[2] + closest_protein_center[2]) / 2
            
            improvement_distance = np.linalg.norm(np.array([center_x, center_y, center_z]) - ligand_center)
            print(f"[INFO] Improved binding site: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
            print(f"[INFO] Reference ligand: ({ligand_center[0]:.3f}, {ligand_center[1]:.3f}, {ligand_center[2]:.3f})")
            print(f"[INFO] Binding site adjustment: {improvement_distance:.3f} Å")
            
        except ImportError:
            print("[WARNING] MDAnalysis not available, using original ligand-based method")
            center_x, center_y, center_z = ligand_center
        except Exception as e:
            print(f"[WARNING] Protein analysis failed: {e}, using original ligand-based method")
            center_x, center_y, center_z = ligand_center
    else:
        # Fallback to original method
        center_x, center_y, center_z = ligand_center

    # Calculate box size with minimum size to ensure adequate search space
    size_x = max(20.0, (max_x - min_x) + 10.0)  # Minimum 20Å box with 10Å margin
    size_y = max(20.0, (max_y - min_y) + 10.0)
    size_z = max(20.0, (max_z - min_z) + 10.0)

    return (center_x, center_y, center_z), (size_x, size_y, size_z)


def prepare_and_run_vina(protein_pdb, ligand_sdf, out_dir, exhaustiveness=8):
    """
    Converts the protein PDB and ligand SDF to PDBQT, computes the
    box, and runs AutoDock Vina. Docked pose is written to out_dir/vina_out.pdbqt.
    Returns the path to the Vina output file for further processing.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    protein_pdbqt = os.path.join(out_dir, "protein.pdbqt")
    ligand_pdbqt  = os.path.join(out_dir, "ligand.pdbqt")
    vina_out      = os.path.join(out_dir, "vina_out.pdbqt")

    # 1) Convert protein from PDB -> PDBQT
    cmd_prot = f"obabel -i pdb {protein_pdb} -o pdbqt -O {protein_pdbqt} --partialcharge gasteiger -xhk -r"
    subprocess.run(cmd_prot, shell=True, check=True)
    # 2) remove ROOT/ENDROOT tags 
    clean_pdbqt = protein_pdbqt.replace(".pdbqt", "_clean.pdbqt")
    with open(protein_pdbqt) as inp, open(clean_pdbqt, "w") as out:
        for line in inp:
            # skip any AutoDock4 torsion-tree markers
            if line.startswith((
                "ROOT", "ENDROOT", 
                "BRANCH", "ENDBRANCH", 
                "TORSDOF"
            )):
                continue
            out.write(line)
    os.replace(clean_pdbqt, protein_pdbqt)
    # Convert ligand from SDF -> PDBQT
    cmd_lig = f"obabel -i sdf {ligand_sdf} -o pdbqt -O {ligand_pdbqt} --partialcharge gasteiger -xh"
    subprocess.run(cmd_lig, shell=True, check=True)

    # Compute the docking box from ligand with improved binding site detection
    center, size = compute_ligand_center_and_size(ligand_sdf, protein_pdb)
    cx, cy, cz = center
    sx, sy, sz = size

    # Run AutoDock Vina
    cmd_vina = (
        f"vina --receptor {protein_pdbqt} --ligand {ligand_pdbqt} "
        f"--center_x {cx:.3f} --center_y {cy:.3f} --center_z {cz:.3f} "
        f"--size_x {sx:.3f} --size_y {sy:.3f} --size_z {sz:.3f} "
        f"--exhaustiveness {exhaustiveness} --out {vina_out}"
    )
    subprocess.run(cmd_vina, shell=True, check=True)
    
    return vina_out

def parse_vina_poses(
    pdbqt_file: str,
    out_dir: str,
    top_n: int = 10,
    prefix: str = "pose",
    remove_hs: bool = True
):
    """
    Parses a multi-model PDBQT from AutoDock Vina, selects the top N poses
    by docking score, and writes each pose to its own SDF file.

    Args:
        pdbqt_file (str): Path to the Vina output PDBQT (multi-model).
        out_dir (str): Directory to write individual SDF files.
        top_n (int): Number of best-scoring poses to extract.
        prefix (str): Prefix for output filenames.
        remove_hs (bool): If True, removes hydrogens from the output.

    The function:
      1. Reads the PDBQT and splits it into MODEL blocks.
      2. Extracts the score from each "REMARK VINA RESULT:" line.
      3. Ranks poses by score (lowest = best).
      4. Uses Open Babel (pybel) to load each top pose.
      5. Optionally removes hydrogens and writes each as SDF.
    """
    import tempfile
    from openbabel import pybel

    os.makedirs(out_dir, exist_ok=True)

    # Read all lines
    lines = open(pdbqt_file).read().splitlines()

    # Split into MODEL blocks and collect scores
    models, scores = [], []
    current = []
    for line in lines:
        if line.startswith("MODEL"):
            current = [line]
        elif line.startswith("ENDMDL"):
            current.append(line)
            models.append(current)
            # parse score
            score = next(
                (float(l.split()[3]) for l in current if l.startswith("REMARK VINA RESULT:")),
                float("inf")
            )
            scores.append(score)
        else:
            if current:
                current.append(line)

    # Get indices of top N poses
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i])[:top_n]

    for rank, idx in enumerate(top_indices, start=1):
        block = models[idx]
        score = scores[idx]

        # write a tiny PDBQT for this single pose
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt")
        tmp.write("\n".join(block).encode())
        tmp.close()

        # load with pybel
        mol = next(pybel.readfile("pdbqt", tmp.name))
        if remove_hs:
            mol.removeh()

        # write out SDF
        out_path = os.path.join(out_dir, f"{prefix}_pose{rank}_score{score:.2f}.sdf")
        mol.write("sdf", out_path, overwrite=True)

        # cleanup
        os.unlink(tmp.name)

    print(f"Extracted top-{top_n} poses to {out_dir}")

def extract_and_write_top_poses(vina_pdbqt_file: str, out_dir: str, prefix: str = "docked", 
                                remove_hs=True, top_n=10):
    """
    Parse the Vina output (PDBQT) which may contain multiple poses.
    Write the top N poses to individual .sdf files in out_dir.

    :param vina_pdbqt_file: Path to the .pdbqt file produced by Vina, containing multiple poses.
    :param out_dir: Where to write the .sdf files.
    :param prefix: A prefix to use in naming the .sdf files.
    :param remove_hs: Whether to remove hydrogens in final .sdf files.
    :param top_n: Number of top conformations to extract.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # -- Example parse of Vina scores from REMARK lines
    scores = []
    with open(vina_pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith('REMARK VINA RESULT:'):
                parts = line.strip().split()
                if len(parts) >= 4:
                    score = float(parts[3])
                    scores.append(score)

    # scores[i] corresponds to pose i (the order they appear in the file)
    # We assume we have at least as many poses as lines with "REMARK VINA RESULT".
    # In practice, you would parse the actual poses out carefully.
    
    # -- Here we assume you have some code to convert each conformation from the
    #    multi-POSE PDBQT into an RDKit molecule. For example:
    #    1. Use a custom parser that returns a list of RDKit molecules (one per pose).
    #    2. Or run "obabel -ipdbqt vina_out.pdbqt -osdf multi.sdf --uniq" and then read them with RDKit.
    # For demonstration, let's assume you have a function get_rdkit_poses(vina_pdbqt_file)
    # that returns (list_of_molecules, list_of_scores). We'll show a placeholder:

    # Placeholder: you MUST implement something like this to extract actual poses
    # from your .pdbqt or from a converted .sdf. We’ll keep this super-simplified:
    def get_rdkit_poses(pdbqt_path: str):
        """
        Parse a multi-POSE PDBQT file into a list of RDKit Mol objects.

        Returns:
            (all_mols, all_scores_placeholder)
            - all_mols: A list of RDKit Mol objects (one for each pose or ligand).
            Each Mol may have multiple conformers if the PDBQT had multiple poses.
            - all_scores_placeholder: Return an empty list or None here,
            since we parse scores separately from REMARK lines.
        """
        # 1) Parse the multi-pose PDBQT file into a PDBQTMolecule object
        # pdbqt_mol = PDBQTMolecule.from_file(pdbqt_path, skip_typing=True)
        pdbqt_mol = PDBQTMolecule.from_file(pdbqt_path)   # default is skip_typing=False
        # 2) Convert the PDBQTMolecule into RDKit Mol objects
        #    Typically returns a list of Mols. 
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        
        # 1) Load your cleaned PDBQT
        pdbqt_mol = PDBQTMolecule.from_file(pdbqt_path)

        # 2) Use the newer `run` method instead of `prepare`
        prep = MoleculePreparation()
        prepared = prep.run(pdbqt_mol)      # returns a prepared Meeko molecule

        # 3) Then convert that to RDKit
        rdkit_mols = RDKitMolCreate().from_pdbqt_mol(prepared)

        # 3) You can do any additional filtering or rearranging as needed.
        #    For example, skip any None entries.
        all_mols = []
        for mol in rdkit_mols:
            if mol is not None:
                all_mols.append(mol)
        
        # 4) We return an empty list for 'all_scores' because 
        #    in the main code, we parse the scores from the REMARK lines.
        return all_mols, []
    
    # Actually retrieve the poses. In real usage, each pose might be an RDKit Mol.
    # For demonstration, assume it returns a list of length = len(scores).
    all_poses, all_scores = get_rdkit_poses(vina_pdbqt_file)

    # If your 'get_rdkit_poses' doesn’t already parse scores, use the 'scores' list we collected from remarks
    # to keep them consistent. Otherwise, you can rely on all_scores if it’s accurate.
    if not all_scores and len(scores) == len(all_poses):
        all_scores = scores

    if not all_poses:
        print(f"[WARNING] No poses found in {vina_pdbqt_file}. Skipping extraction.")
        return

    # Sort the poses by score if needed (lowest score is best for Vina).
    # We'll create an index-based sort:
    # pairs like (pose_index, score)
    indexed_scores = list(enumerate(all_scores))
    # Sort by the score value
    indexed_scores.sort(key=lambda x: x[1])

    # Now keep only top_n
    top_indices = indexed_scores[:top_n]

    for rank, (pose_idx, pose_score) in enumerate(top_indices, start=1):
        rdkit_mol = all_poses[pose_idx]

        if remove_hs:
            rdkit_mol = Chem.RemoveHs(rdkit_mol)

        # Write out as .sdf
        out_sdf = os.path.join(out_dir, f"{prefix}_pose{rank}_score{pose_score:.2f}.sdf")
        # RDKit needs the conformer. If your mol has multiple conformers, ensure
        # you select the correct one. For example:
        # conformer_id = 0
        # tmp_mol = Chem.Mol(rdkit_mol)
        # tmp_mol.RemoveAllConformers()
        # tmp_mol.AddConformer(rdkit_mol.GetConformer(conformer_id))

        # For now, just write it directly if each RDKit Mol is a single conformer:
        Chem.MolToMolFile(rdkit_mol, out_sdf)

    print(f"[INFO] Wrote top-{top_n} poses to SDF in {out_dir}")




def run_dataset(config: dict):
    """Run Vina docking over a full benchmark dataset via inputs CSV."""
    logger = get_custom_logger("vina", config, f"vina_timing_{config['repeat_index']}.log")
    inputs_df = pd.read_csv(config['inputs_csv'])

    for _, row in inputs_df.iterrows():
        protein_dir = row['complex_name']
        protein_pdb = row['protein_path'].replace("receptor.pdb", f"{protein_dir}_protein.pdb")
        ligand_sdf  = row['ligand_path']

        if not (os.path.exists(protein_pdb) and os.path.exists(ligand_sdf)):
            print(f"[WARNING] Missing files for {protein_dir}. Skipping.")
            continue

        out_dir = os.path.join(config['output_dir'], protein_dir)
        os.makedirs(out_dir, exist_ok=True)

        if config.get('skip_existing', False):
            existing = [f for f in os.listdir(out_dir) if f.endswith('.sdf')]
            if existing:
                print(f"[INFO] Skipping {protein_dir} — already done.")
                continue

        print(f"\n[INFO] Processing: {protein_dir}")
        start_time = time.time()
        try:
            vina_out = prepare_and_run_vina(protein_pdb, ligand_sdf, out_dir)
            parse_vina_poses(
                pdbqt_file=vina_out,
                out_dir=out_dir,
                top_n=config.get('top_n', 10),
                prefix=protein_dir,
                remove_hs=True,
            )
        except Exception as e:
            print(f"[ERROR] Vina failed for {protein_dir}: {e}")
        elapsed_time = time.time() - start_time
        print(f"[INFO] Done in {elapsed_time:.2f}s")
        logger.info(f"{protein_dir},{elapsed_time:.2f}")


def run_single(
    protein: str,
    ligand: str,
    output_dir: str,
    config: dict = None,
    prefix: str = None,
    **kwargs,
) -> str:
    """Run Vina on a single protein-ligand pair.

    Results are written to output_dir as {prefix}_pose{rank}_score{score}.sdf files.
    """
    config = config or {}
    prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
    os.makedirs(output_dir, exist_ok=True)

    exhaustiveness = kwargs.get('exhaustiveness', config.get('exhaustiveness', 8))
    top_n = kwargs.get('top_n', config.get('top_n', 10))

    try:
        vina_out = prepare_and_run_vina(protein, ligand, output_dir, exhaustiveness=exhaustiveness)
        parse_vina_poses(
            pdbqt_file=vina_out,
            out_dir=output_dir,
            top_n=top_n,
            prefix=prefix,
            remove_hs=True,
        )
        print(f"[INFO] Vina docking complete. Results in {output_dir}")
    except Exception as e:
        print(f"[ERROR] Vina docking failed: {e}")
        raise
    return output_dir


def parse_config_file_with_omegaconf(config_path):
    """
    Parse a configuration file using OmegaConf and resolve variables.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Resolved configuration as a dictionary
    """
    # Make sure the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Set up environment variables for resolution
    # In a real application, these should be set before calling this function
    os.environ["PROJECT_ROOT"] = "/mnt/katritch_lab2/aoxu/CogLigandBench"  # Example value
    
    # Load the configuration from file
    cfg = OmegaConf.load(config_path)
    
    # Resolve interpolation
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    
    return resolved_cfg


if __name__ == "__main__":
    config = parse_config_file_with_omegaconf(sys.argv[1])
    run_dataset(config)
