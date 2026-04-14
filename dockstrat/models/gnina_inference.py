import gzip
import os
import subprocess
import sys
import time
import logging

from omegaconf import OmegaConf
from rdkit import Chem
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from dockstrat.utils.log import get_custom_logger

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(rootutils.find_root(search_from=__file__, indicator=".project-root")))


# ---------------------------------------------------------------------------
# Core docking helpers
# ---------------------------------------------------------------------------

def run_gnina_docking(protein_pdb: str, ligand_sdf: str, out_dir: str) -> str:
    """Run GNINA with autobox on the reference ligand. Output: out_dir/docked.sdf.gz."""
    os.makedirs(out_dir, exist_ok=True)
    gnina_exec = os.path.join(PROJECT_ROOT, "forks", "GNINA", "gnina")
    out_file = os.path.join(out_dir, "docked.sdf.gz")
    cmd = (
        f"{gnina_exec} --receptor {protein_pdb} --ligand {ligand_sdf} "
        f"--autobox_ligand {ligand_sdf} --out {out_file}"
    )
    subprocess.run(cmd, shell=True, check=True)
    return out_file


def decompress_file(file_path: str) -> str:
    """Decompress a .gz file in place, return decompressed path."""
    decompressed = file_path[:-3]
    with gzip.open(file_path, "rb") as fin, open(decompressed, "wb") as fout:
        fout.write(fin.read())
    return decompressed


def load_sdf(file_path: str):
    suppl = Chem.ForwardSDMolSupplier(file_path, removeHs=False)
    return [mol for mol in suppl if mol is not None]


def extract_scores(mols):
    results = []
    for mol in mols:
        cnn_score = float(mol.GetProp("CNNscore")) if mol.HasProp("CNNscore") else None
        results.append({"mol": mol, "cnn_score": cnn_score})
    return results


def rank_and_save_poses(results, output_dir: str, prefix: str = "docked", top_n: int = 10):
    """Sort by CNNscore (higher = better), write top_n poses as individual SDF files."""
    results = [r for r in results if r["cnn_score"] is not None]
    results.sort(key=lambda x: x["cnn_score"], reverse=True)
    os.makedirs(output_dir, exist_ok=True)
    for rank, result in enumerate(results[:top_n], start=1):
        out_path = os.path.join(output_dir, f"{prefix}_pose{rank}_score{result['cnn_score']:.2f}.sdf")
        writer = Chem.SDWriter(out_path)
        writer.write(result["mol"])
        writer.close()


def _process_system(protein_pdb: str, ligand_sdf: str, out_dir: str, prefix: str, top_n: int):
    """Run GNINA + post-process for a single system."""
    run_gnina_docking(protein_pdb, ligand_sdf, out_dir)
    sdf_gz = os.path.join(out_dir, "docked.sdf.gz")
    sdf_path = decompress_file(sdf_gz)
    mols = load_sdf(sdf_path)
    results = extract_scores(mols)
    rank_and_save_poses(results, out_dir, prefix=prefix, top_n=top_n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dataset(config: dict):
    """Run GNINA docking over a full benchmark dataset via inputs CSV."""
    logger = get_custom_logger("gnina", config, f"gnina_timing_{config['repeat_index']}.log")
    inputs_df = pd.read_csv(config["inputs_csv"])
    top_n = config.get("top_n", 10)

    for _, row in inputs_df.iterrows():
        system_id = row["complex_name"]
        protein_pdb = row["protein_path"].replace("receptor.pdb", f"{system_id}_protein.pdb")
        ligand_sdf = row["ligand_path"]

        if not (os.path.exists(protein_pdb) and os.path.exists(ligand_sdf)):
            print(f"[WARNING] Missing files for {system_id}. Skipping.")
            continue

        out_dir = os.path.join(config["output_dir"], system_id)
        os.makedirs(out_dir, exist_ok=True)

        if config.get("skip_existing", False):
            if any(f.endswith(".sdf") for f in os.listdir(out_dir)):
                print(f"[INFO] Skipping {system_id} — already done.")
                continue

        print(f"\n[INFO] Processing: {system_id}")
        start = time.time()
        try:
            _process_system(protein_pdb, ligand_sdf, out_dir, prefix=system_id, top_n=top_n)
        except Exception as e:
            print(f"[ERROR] GNINA failed for {system_id}: {e}")
        logger.info(f"{system_id},{time.time() - start:.2f}")


def run_single(
    protein: str,
    ligand: str,
    output_dir: str,
    config: dict = None,
    prefix: str = None,
    **kwargs,
) -> str:
    """Run GNINA on a single protein-ligand pair.

    Results are written to output_dir as {prefix}_pose{rank}_score{score}.sdf files.
    """
    config = config or {}
    prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
    top_n = kwargs.get("top_n", config.get("top_n", 10))
    os.makedirs(output_dir, exist_ok=True)

    try:
        _process_system(protein, ligand, output_dir, prefix=prefix, top_n=top_n)
        print(f"[INFO] GNINA docking complete. Results in {output_dir}")
    except Exception as e:
        print(f"[ERROR] GNINA docking failed: {e}")
        raise
    return output_dir


# ---------------------------------------------------------------------------
# Config loading + CLI entry point
# ---------------------------------------------------------------------------

def parse_config_file_with_omegaconf(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    config = parse_config_file_with_omegaconf(sys.argv[1])
    run_dataset(config)
