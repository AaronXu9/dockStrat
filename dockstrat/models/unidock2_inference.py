"""UniDock2 inference for dockstrat."""
import os
import re
import subprocess
import sys
import tempfile
import time
import logging
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from dockstrat.utils.log import get_custom_logger

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(rootutils.find_root(search_from=__file__, indicator=".project-root")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_ligand_center(ligand_sdf: str):
    """Compute ligand centroid from SDF using RDKit."""
    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
    mol = next(iter(suppl), None)
    if mol is None:
        raise ValueError(f"Cannot read molecule from {ligand_sdf}")
    conf = mol.GetConformer()
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())]
    )
    return tuple(float(v) for v in coords.mean(0))


def _run_unidock2(protein: str, ligand: str, output_dir: str, center, config: dict) -> list:
    """Run a single UniDock2 docking job. Returns paths to ranked SDF files."""
    box_size = config.get("box_size", [15.0, 15.0, 15.0])
    num_poses = config.get("num_poses", 10)
    search_mode = config.get("search_mode", "detail")
    energy_range = config.get("energy_range", 15.0)
    timeout = config.get("timeout", 1800)

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="unidock2_") as tmp:
        posed_sdf = os.path.join(tmp, "docked_poses.sdf")
        dock_config = {
            "Required": {"receptor": protein, "ligand": ligand, "center": list(center)},
            "Settings": {
                "size": list(box_size),
                "search_mode": search_mode,
                "num_poses": num_poses,
                "energy_range": energy_range,
            },
            "Preprocessing": {"output_docking_pose_sdf_file_name": posed_sdf},
        }
        config_file = os.path.join(tmp, "config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(dock_config, f)

        result = subprocess.run(
            ["conda", "run", "-n", "unidock2", "unidock2", "docking", "-cf", config_file],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"UniDock2 failed: {result.stderr[:500]}")

        if not os.path.exists(posed_sdf):
            raise RuntimeError("UniDock2 produced no output file")

        # Copy raw poses SDF to output dir before tmp is cleaned up
        raw_poses_sdf = os.path.join(output_dir, f"{os.path.basename(output_dir)}_poses.sdf")
        with open(posed_sdf) as f:
            content = f.read()
        with open(raw_poses_sdf, "w") as f:
            f.write(content)

    # Parse and rank poses by binding free energy.
    # split('$$$$')[:-1] drops the trailing empty element after the last delimiter.
    poses = content.split("$$$$")[:-1]
    pose_data = []
    for pose in poses:
        m = re.search(r"<vina_binding_free_energy>\s*\(\d+\)\s*\n([-\d.]+)", pose)
        energy = float(m.group(1)) if m else 999.0
        pose_data.append((energy, pose))
    pose_data.sort(key=lambda x: x[0])

    rank_files = []
    for rank, (energy, pose) in enumerate(pose_data[:num_poses], 1):
        rank_file = os.path.join(output_dir, f"rank{rank}.sdf")
        with open(rank_file, "w") as f:
            f.write(pose.lstrip("\n") + "$$$$\n")
        rank_files.append(rank_file)

    return rank_files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dataset(config: dict):
    """Run UniDock2 over all systems in a data directory."""
    data_dir = Path(config["data_dir"])
    output_dir = config["output_dir"]
    skip_existing = config.get("skip_existing", True)

    os.makedirs(output_dir, exist_ok=True)
    logger = get_custom_logger("unidock2", config, f"unidock2_timing_{config['repeat_index']}.log")

    system_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(system_dirs)} systems in {data_dir}")

    for system_dir in system_dirs:
        system_id = system_dir.name
        sys_out = os.path.join(output_dir, system_id)

        if skip_existing and list(Path(sys_out).glob("rank*.sdf")) if os.path.isdir(sys_out) else False:
            logger.info(f"Skipping {system_id} — already done.")
            continue

        protein_files = list(system_dir.glob("*_protein.pdb"))
        ligand_files = list(system_dir.glob("*_ligand.sdf"))
        if not protein_files or not ligand_files:
            logger.warning(f"Missing protein/ligand for {system_id}. Skipping.")
            continue

        protein = str(protein_files[0])
        ligand = str(ligand_files[0])

        try:
            center = _compute_ligand_center(ligand)
        except Exception as e:
            logger.error(f"Center calculation failed for {system_id}: {e}")
            continue

        start = time.time()
        try:
            _run_unidock2(protein, ligand, sys_out, center, config)
            logger.info(f"{system_id},{time.time() - start:.2f}")
        except Exception as e:
            logger.error(f"UniDock2 failed for {system_id}: {e}")


def run_single(
    protein: str,
    ligand: str,
    output_dir: str,
    config: dict = None,
    prefix: str = None,
    **kwargs,
) -> str:
    """Run UniDock2 on a single protein-ligand pair.

    Results are written as rank1.sdf, rank2.sdf, ... in output_dir.
    """
    config = config or {}
    prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
    os.makedirs(output_dir, exist_ok=True)

    center = _compute_ligand_center(ligand)

    # Allow kwargs to override config
    merged = {**config, **{k: v for k, v in kwargs.items() if v is not None}}

    try:
        _run_unidock2(protein, ligand, output_dir, center, merged)
        print(f"[INFO] UniDock2 docking complete. Results in {output_dir}")
    except Exception as e:
        print(f"[ERROR] UniDock2 docking failed: {e}")
        raise
    return output_dir


# ---------------------------------------------------------------------------
# Config loading + CLI entry point
# ---------------------------------------------------------------------------

def parse_config_file_with_omegaconf(config_path: str) -> dict:
    from omegaconf import OmegaConf
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    config = parse_config_file_with_omegaconf(sys.argv[1])
    run_dataset(config)
