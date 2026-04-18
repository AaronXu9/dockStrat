"""Chai-1 docking wrapper.

Runs Chai-1 via a subprocess in a dedicated conda env, then extracts
ligand SDFs from the predicted PDB/CIF files using the known input
SMILES for bond-order recovery.

Public surface (used by dockstrat.engine):
  - run_single(protein, ligand, output_dir, config=None, prefix=None, **kwargs)
  - run_dataset(config)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def _smiles_from_sdf(sdf_path: str) -> str:
    """Read the first molecule from an SDF and return its canonical SMILES."""
    from rdkit import Chem

    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
    except OSError as exc:
        raise ValueError(f"Cannot open SDF file {sdf_path}: {exc}") from exc
    mol = next(iter(suppl), None)
    if mol is None:
        raise ValueError(f"Cannot read molecule from {sdf_path}")
    return Chem.MolToSmiles(mol)


def _write_fasta(fasta_path: str, system_id: str, sequences: List[str], smiles: str):
    """Write a Chai-1 compatible FASTA file."""
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">protein|{system_id}-chain-{i}\n{seq}\n")
        f.write(f">ligand|{system_id}-chain-{len(sequences) + 1}\n{smiles}\n")


def _extract_ligand_from_chai_output(
    output_path: str | Path,
    template_mol: "Chem.Mol",
) -> "Chem.Mol":
    """Extract ligand from a Chai-1 prediction file and recover bond orders.

    Chai-1 labels ligand atoms with residue name ``LIG``. The output may be
    PDB or mmCIF depending on the Chai-1 version.  All atoms whose residue
    name is ``LIG`` are collected into a synthetic PDB block, parsed by
    RDKit, and then bond orders are recovered via
    ``AssignBondOrdersFromTemplate`` with the known SMILES template.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    output_path = Path(output_path)
    ext = output_path.suffix.lower()

    if ext == ".cif":
        pdb_lines = _extract_lig_atoms_from_cif(output_path)
    else:
        pdb_lines = _extract_lig_atoms_from_pdb(output_path)

    if not pdb_lines:
        raise ValueError(f"No ligand atoms (residue LIG) found in {output_path}")

    pdb_block = "\n".join(pdb_lines) + "\nEND\n"
    raw_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if raw_mol is None:
        raise ValueError(f"Failed to parse synthesized PDB block for ligand from {output_path}")

    try:
        Chem.SanitizeMol(raw_mol)
    except Exception:
        pass

    annotated = AllChem.AssignBondOrdersFromTemplate(template_mol, raw_mol)
    return annotated


def _extract_lig_atoms_from_pdb(pdb_path: Path) -> List[str]:
    """Collect LIG-residue atoms from a PDB file as HETATM lines."""
    pdb_lines = []
    atom_idx = 1
    with open(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip()
            if resname != "LIG":
                continue
            name = line[12:16]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = line[76:78].strip() if len(line) >= 78 and line[76:78].strip() else name.strip()[0]
            pdb_lines.append(
                f"HETATM{atom_idx:>5} {name} LIG L{1:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
                f"{elem.rjust(2)}"
            )
            atom_idx += 1
    return pdb_lines


def _extract_lig_atoms_from_cif(cif_path: Path) -> List[str]:
    """Collect LIG-residue atoms from an mmCIF file as HETATM lines."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    mmcif_dict = MMCIF2Dict(str(cif_path))

    resname_list = mmcif_dict.get("_atom_site.label_comp_id", [])
    atom_name_list = mmcif_dict.get("_atom_site.label_atom_id", [])
    elem_list = mmcif_dict.get("_atom_site.type_symbol", [])
    x_list = mmcif_dict.get("_atom_site.Cartn_x", [])
    y_list = mmcif_dict.get("_atom_site.Cartn_y", [])
    z_list = mmcif_dict.get("_atom_site.Cartn_z", [])

    n_atoms = len(x_list)

    pdb_lines = []
    atom_idx = 1
    for i in range(n_atoms):
        resname = (resname_list[i] if i < len(resname_list) else "").strip()
        if resname != "LIG":
            continue
        name = (atom_name_list[i] if i < len(atom_name_list) else "X")[:4].ljust(4)
        elem = (elem_list[i] if i < len(elem_list) else name.strip()[0]).strip()
        x = float(x_list[i])
        y = float(y_list[i])
        z = float(z_list[i])
        pdb_lines.append(
            f"HETATM{atom_idx:>5} {name} LIG L{1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
            f"{elem.rjust(2)}"
        )
        atom_idx += 1
    return pdb_lines


def _extract_ranked_ligand_sdfs(
    chai_output_dir: str | Path,
    smiles: str,
    out_dir: str | Path,
    num_poses: int,
) -> int:
    """Read Chai-1 NPZ score files, sort by aggregate_score desc, write top-N rank{i}.sdf.

    Chai-1 writes separate files per model:
      ``pred.model_idx_{N}.{pdb,cif}``
      ``scores.model_idx_{N}.npz``

    The NPZ files contain ``aggregate_score`` (shape ``(1,)``, float32).
    Returns the number of SDFs successfully written.
    """
    import numpy as np
    from rdkit import Chem

    chai_output_dir = Path(chai_output_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse template SMILES: {smiles}")

    # Discover all score files and pair with prediction files
    models = []
    for npz_path in sorted(chai_output_dir.glob("scores.model_idx_*.npz")):
        idx_str = npz_path.stem.replace("scores.model_idx_", "")
        # Try .cif first (newer Chai versions), fall back to .pdb
        pred_path = chai_output_dir / f"pred.model_idx_{idx_str}.cif"
        if not pred_path.exists():
            pred_path = chai_output_dir / f"pred.model_idx_{idx_str}.pdb"
        if not pred_path.exists():
            continue
        data = np.load(str(npz_path), allow_pickle=True)
        score = float(data["aggregate_score"].flat[0])
        models.append((score, pred_path))

    # Sort descending by aggregate_score
    models.sort(key=lambda x: x[0], reverse=True)

    written = 0
    for score, pred_path in models[:num_poses]:
        try:
            mol = _extract_ligand_from_chai_output(pred_path, template_mol=template)
        except Exception:
            continue
        written += 1
        Chem.MolToMolFile(mol, str(out_dir / f"rank{written}.sdf"))
    return written


def _run_chai_subprocess(
    fasta_path: str | Path,
    out_dir: str | Path,
    config: dict,
) -> None:
    """Invoke Chai-1's run_inference via the dedicated conda env's Python.

    Raises ``subprocess.CalledProcessError`` on nonzero exit and
    ``subprocess.TimeoutExpired`` on hard timeout.
    """
    import subprocess

    python_exec = config["python_exec_path"]
    cuda_device = config.get("cuda_device_index", 0)
    num_trunk_recycles = config.get("num_trunk_recycles", 3)
    num_diffn_timesteps = config.get("num_diffn_timesteps", 200)
    seed = config.get("seed", 42)
    use_esm_embeddings = config.get("use_esm_embeddings", True)

    os.makedirs(str(out_dir), exist_ok=True)

    script = (
        "import torch\n"
        "from chai_lab.chai1 import run_inference\n"
        "from pathlib import Path\n"
        f"run_inference(\n"
        f"    fasta_file=Path('{fasta_path}'),\n"
        f"    output_dir=Path('{out_dir}'),\n"
        f"    num_trunk_recycles={num_trunk_recycles},\n"
        f"    num_diffn_timesteps={num_diffn_timesteps},\n"
        f"    seed={seed},\n"
        f"    device=torch.device('cuda:{cuda_device}'),\n"
        f"    use_esm_embeddings={use_esm_embeddings},\n"
        f")\n"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    subprocess.run(
        [str(python_exec), "-c", script],
        env=env,
        check=True,
        timeout=config.get("timeout_seconds", 3600),
    )


def run_single(
    protein: str,
    ligand: str,
    output_dir: str,
    config: Optional[dict] = None,
    prefix: Optional[str] = None,
    **kwargs,
) -> str:
    """Dock a single protein+ligand pair with Chai-1.

    Writes a FASTA, runs Chai-1 subprocess, then extracts the top-N
    ranked ligand poses as ``rank{1..N}.sdf`` directly into ``output_dir``.
    Returns ``output_dir`` on success.
    """
    from dockstrat.utils.sequence import extract_protein_sequence

    config = config or {}

    if "python_exec_path" not in config:
        raise ValueError(
            "chai run_single: missing required config key 'python_exec_path'"
        )

    prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Build FASTA from PDB + SDF
    sequences = extract_protein_sequence(protein)
    if not sequences:
        raise ValueError(f"No protein sequences found in {protein}")
    smiles = _smiles_from_sdf(ligand)

    fasta_path = os.path.join(output_dir, f"{prefix}.fasta")
    _write_fasta(fasta_path, prefix, sequences, smiles)

    # Phase 2: Run Chai-1 subprocess
    _run_chai_subprocess(fasta_path, output_dir, config)

    # Phase 3: Extract ranked ligand SDFs
    num_poses = int(config.get("num_poses_to_keep", 5))
    _extract_ranked_ligand_sdfs(
        chai_output_dir=output_dir,
        smiles=smiles,
        out_dir=Path(output_dir),
        num_poses=num_poses,
    )
    return output_dir


def run_dataset(config: dict) -> None:
    """Run Chai-1 over an entire dataset directory.

    Iterates the per-system folders under ``config['input_dir']``. Each
    folder is expected to contain ``{sys_id}_protein.pdb`` and
    ``{sys_id}_ligand.sdf``. Writes ``rank{N}.sdf`` files into
    ``config['output_dir']/{sys_id}/`` per system.
    """
    import time
    import traceback

    from dockstrat.utils.log import get_custom_logger

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    skip_existing = config.get("skip_existing", True)
    max_num_inputs = config.get("max_num_inputs", None)

    os.makedirs(output_dir, exist_ok=True)
    logging_cfg = config.get("logging") or {}
    log_config = dict(config)
    if logging_cfg.get("log_dir") and "log_dir" not in log_config:
        log_config["log_dir"] = logging_cfg["log_dir"]
    logger = get_custom_logger(
        "chai-1", log_config,
        f"chai-1_timing_{config.get('dataset', 'unknown')}_{config.get('repeat_index', 0)}.log",
    )

    num_processed = 0
    for sys_id in sorted(os.listdir(input_dir)):
        sys_in = os.path.join(input_dir, sys_id)
        if not os.path.isdir(sys_in):
            continue
        if max_num_inputs is not None and num_processed >= max_num_inputs:
            logger.info(f"Reached max_num_inputs={max_num_inputs}. Stopping.")
            break

        sys_out = os.path.join(output_dir, sys_id)
        if skip_existing and os.path.exists(os.path.join(sys_out, "rank1.sdf")):
            logger.info(f"Skipping {sys_id} — rank1.sdf already exists.")
            continue

        protein = os.path.join(sys_in, f"{sys_id}_protein.pdb")
        ligand = os.path.join(sys_in, f"{sys_id}_ligand.sdf")
        if not (os.path.exists(protein) and os.path.exists(ligand)):
            logger.warning(f"Missing protein or ligand for {sys_id}. Skipping.")
            continue

        os.makedirs(sys_out, exist_ok=True)
        start = time.time()
        try:
            run_single(
                protein=protein,
                ligand=ligand,
                output_dir=sys_out,
                config=config,
                prefix=sys_id,
            )
            logger.info(f"{sys_id},{time.time() - start:.2f}")
            num_processed += 1
        except Exception as e:
            logger.error(f"Failed {sys_id}: {e}")
            with open(os.path.join(sys_out, "error_log.txt"), "w") as fh:
                traceback.print_exc(file=fh)


# ---------------------------------------------------------------------------
# Config loading + CLI entry point
# ---------------------------------------------------------------------------

def parse_config_file_with_omegaconf(config_path: str) -> dict:
    from omegaconf import OmegaConf
    import rootutils

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    project_root = str(rootutils.find_root(search_from=__file__, indicator=".project-root"))
    os.environ.setdefault("PROJECT_ROOT", project_root)
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    import sys
    config = parse_config_file_with_omegaconf(sys.argv[1])
    run_dataset(config)
