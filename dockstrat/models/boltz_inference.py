"""Boltz-1 / Boltz-2 docking wrapper.

Runs the ``boltz predict`` CLI in a dedicated conda env via subprocess,
then extracts ligand SDFs from the predicted mmCIFs using the known input
SMILES for bond-order recovery.

Both boltz1 and boltz2 are handled by this module; the ``model`` field in
the YAML config (``"boltz1"`` or ``"boltz2"``) selects the variant.

Public surface (used by dockstrat.engine):
  - run_single(protein, ligand, output_dir, config=None, prefix=None, **kwargs)
  - run_dataset(config)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def _build_boltz_input_yaml(
    system_id: str,
    pdb_path: str,
    sdf_path: str,
    model: str = "boltz2",
) -> Tuple[Dict, str]:
    """Construct the Boltz input-YAML dict for a single protein + ligand pair.

    Returns ``(yaml_dict, ligand_chain_id)`` so callers know which chain to
    extract the ligand from in the predicted structure.

    Each protein chain is set to ``msa: "empty"`` for single-sequence / no-MSA
    mode.  For Boltz-2, a ``properties`` block requesting affinity prediction
    is added automatically.
    """
    from dockstrat.utils.sequence import extract_protein_sequence

    sequences = extract_protein_sequence(pdb_path)
    if not sequences:
        raise ValueError(f"No protein sequences found in {pdb_path}")

    smiles = _smiles_from_sdf(sdf_path)

    entries: List[Dict] = []
    for i, seq in enumerate(sequences):
        chain_id = chr(ord("A") + i)
        entries.append({
            "protein": {
                "id": chain_id,
                "sequence": seq,
                "msa": "empty",
            }
        })

    ligand_chain_id = chr(ord("A") + len(sequences))
    entries.append({"ligand": {"id": ligand_chain_id, "smiles": smiles}})

    yaml_dict: Dict = {"version": 1, "sequences": entries}

    if model == "boltz2":
        yaml_dict["properties"] = [{"affinity": {"binder": ligand_chain_id}}]

    return yaml_dict, ligand_chain_id


def _extract_ligand_from_cif(
    cif_path: str | Path,
    ligand_chain_id: str,
    template_mol: "Chem.Mol",
) -> "Chem.Mol":
    """Pull the ligand out of a Boltz mmCIF and recover bond orders.

    Strategy: parse the mmCIF with Biopython's MMCIF2Dict, collect all atoms
    on ``ligand_chain_id``, synthesize a PDB block, parse with RDKit (which
    infers connectivity from distances), then call
    ``AssignBondOrdersFromTemplate`` to recover the correct bond orders.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mmcif_dict = MMCIF2Dict(str(cif_path))

    chain_list = mmcif_dict.get(
        "_atom_site.auth_asym_id",
        mmcif_dict.get("_atom_site.label_asym_id", []),
    )
    resname_list = mmcif_dict.get("_atom_site.label_comp_id", [])
    atom_name_list = mmcif_dict.get("_atom_site.label_atom_id", [])
    elem_list = mmcif_dict.get("_atom_site.type_symbol", [])
    x_list = mmcif_dict.get("_atom_site.Cartn_x", [])
    y_list = mmcif_dict.get("_atom_site.Cartn_y", [])
    z_list = mmcif_dict.get("_atom_site.Cartn_z", [])

    n_atoms = len(x_list)
    if len(y_list) != n_atoms or len(z_list) != n_atoms or len(chain_list) != n_atoms:
        raise ValueError(
            f"Malformed mmCIF {cif_path}: _atom_site columns have inconsistent lengths "
            f"(x={len(x_list)}, y={len(y_list)}, z={len(z_list)}, chain={len(chain_list)})"
        )

    pdb_lines = []
    atom_idx = 1
    for i in range(n_atoms):
        if chain_list[i] != ligand_chain_id:
            continue
        resname = (resname_list[i] if i < len(resname_list) else "LIG")[:3].ljust(3)
        name = (atom_name_list[i] if i < len(atom_name_list) else "X")[:4].ljust(4)
        elem = (elem_list[i] if i < len(elem_list) else name.strip()[0]).strip()
        x = float(x_list[i])
        y = float(y_list[i])
        z = float(z_list[i])
        pdb_lines.append(
            f"HETATM{atom_idx:>5} {name} {resname} {ligand_chain_id}{1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
            f"{elem.rjust(2)}"
        )
        atom_idx += 1

    if not pdb_lines:
        raise ValueError(
            f"No ligand atoms (chain {ligand_chain_id}) found in {cif_path}"
        )

    pdb_block = "\n".join(pdb_lines) + "\nEND\n"
    raw_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if raw_mol is None:
        raise ValueError(f"Failed to parse synthesized PDB block for ligand from {cif_path}")

    try:
        Chem.SanitizeMol(raw_mol)
    except Exception:
        pass

    annotated = AllChem.AssignBondOrdersFromTemplate(template_mol, raw_mol)
    return annotated


def _extract_ranked_ligand_sdfs(
    boltz_pred_dir: str | Path,
    stem: str,
    smiles: str,
    ligand_chain_id: str,
    out_dir: str | Path,
    num_poses: int,
    affinity_json_path: str | Path | None = None,
) -> int:
    """Read Boltz confidence JSONs, rank by score, write top-N rank{i}.sdf.

    Returns the number of SDFs successfully written.
    """
    from rdkit import Chem

    boltz_pred_dir = Path(boltz_pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover confidence JSON files
    conf_files = sorted(boltz_pred_dir.glob(f"confidence_{stem}_model_*.json"))
    if not conf_files:
        raise FileNotFoundError(
            f"No confidence JSON files found at {boltz_pred_dir}/confidence_{stem}_model_*.json"
        )

    # Parse confidence scores
    scored: List[Tuple[float, int]] = []
    for cf in conf_files:
        match = re.search(r"_model_(\d+)\.json$", cf.name)
        if match is None:
            continue
        model_idx = int(match.group(1))
        with open(cf) as fh:
            data = json.load(fh)
        score = float(data.get("confidence_score", 0.0))
        scored.append((score, model_idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse template SMILES: {smiles}")

    # Load affinity data if available (Boltz-2)
    affinity_data = None
    if affinity_json_path and Path(affinity_json_path).exists():
        with open(affinity_json_path) as fh:
            affinity_data = json.load(fh)

    written = 0
    for score, model_idx in scored[:num_poses]:
        cif = boltz_pred_dir / f"{stem}_model_{model_idx}.cif"
        if not cif.exists():
            continue
        try:
            mol = _extract_ligand_from_cif(cif, ligand_chain_id, template_mol=template)
        except Exception:
            continue
        written += 1

        # Attach confidence score as SDF property
        mol.SetProp("boltz_confidence_score", f"{score:.4f}")

        # Attach affinity data if available
        if affinity_data is not None:
            for key in ("affinity_pred_value", "affinity_probability_binary"):
                if key in affinity_data:
                    mol.SetProp(f"boltz2_{key}", str(affinity_data[key]))

        # Use SDWriter (not MolToMolFile) to preserve SD properties
        writer = Chem.SDWriter(str(out_dir / f"rank{written}.sdf"))
        writer.write(mol)
        writer.close()

    return written


def _run_boltz_subprocess(
    yaml_path: str | Path,
    out_dir: str | Path,
    config: dict,
) -> None:
    """Invoke ``boltz predict`` via subprocess.

    Raises ``subprocess.CalledProcessError`` on nonzero exit and
    ``subprocess.TimeoutExpired`` on hard timeout.
    """
    import subprocess

    boltz_binary = config["boltz_binary"]
    model = config.get("model", "boltz2")

    cmd = [
        str(boltz_binary), "predict", str(yaml_path),
        f"--out_dir={out_dir}",
        f"--model={model}",
        "--output_format=mmcif",
        f"--diffusion_samples={config.get('diffusion_samples', 5)}",
        f"--recycling_steps={config.get('recycling_steps', 3)}",
        f"--sampling_steps={config.get('sampling_steps', 200)}",
        "--override",
    ]

    if "seed" in config and config["seed"] is not None:
        cmd.append(f"--seed={config['seed']}")

    if "step_scale" in config and config["step_scale"] is not None:
        cmd.append(f"--step_scale={config['step_scale']}")

    # Boltz-2 affinity-specific params
    if model == "boltz2":
        if "sampling_steps_affinity" in config:
            cmd.append(f"--sampling_steps_affinity={config['sampling_steps_affinity']}")
        if "diffusion_samples_affinity" in config:
            cmd.append(f"--diffusion_samples_affinity={config['diffusion_samples_affinity']}")
        if config.get("affinity_mw_correction"):
            cmd.append("--affinity_mw_correction")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("cuda_device_index", 0))

    subprocess.run(
        cmd,
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
    """Dock a single protein+ligand pair with Boltz.

    Writes the Boltz input YAML, runs the subprocess, then extracts the top-N
    ranked ligand poses as ``rank{1..N}.sdf`` directly into ``output_dir``.
    Returns ``output_dir`` on success.
    """
    import yaml

    config = config or {}

    if "boltz_binary" not in config:
        raise ValueError("boltz run_single: missing required config key 'boltz_binary'")

    model = config.get("model", "boltz2")
    prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Build the input YAML
    yaml_dict, ligand_chain_id = _build_boltz_input_yaml(prefix, protein, ligand, model=model)
    yaml_path = Path(output_dir) / f"{prefix}.yaml"
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(yaml_dict, fh, default_flow_style=False, sort_keys=False)

    # 2. Run Boltz (writes to output_dir/boltz_results_{prefix}/predictions/{prefix}/...)
    _run_boltz_subprocess(yaml_path, Path(output_dir), config)

    # 3. Extract the ranked ligand SDFs
    smiles = _smiles_from_sdf(ligand)
    boltz_pred_dir = Path(output_dir) / f"boltz_results_{prefix}" / "predictions" / prefix
    if not boltz_pred_dir.exists():
        raise FileNotFoundError(
            f"Boltz produced no output directory at {boltz_pred_dir}"
        )

    # Check for affinity JSON (Boltz-2)
    affinity_json = boltz_pred_dir / f"affinity_{prefix}.json"
    affinity_json_path = str(affinity_json) if affinity_json.exists() else None

    num_poses = int(config.get("num_poses_to_keep", config.get("diffusion_samples", 5)))
    _extract_ranked_ligand_sdfs(
        boltz_pred_dir,
        stem=prefix,
        smiles=smiles,
        ligand_chain_id=ligand_chain_id,
        out_dir=Path(output_dir),
        num_poses=num_poses,
        affinity_json_path=affinity_json_path,
    )
    return output_dir


def run_dataset(config: dict) -> None:
    """Run Boltz over an entire dataset directory.

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
    model = config.get("model", "boltz2")

    os.makedirs(output_dir, exist_ok=True)

    logging_cfg = config.get("logging") or {}
    log_config = dict(config)
    if logging_cfg.get("log_dir") and "log_dir" not in log_config:
        log_config["log_dir"] = logging_cfg["log_dir"]
    logger = get_custom_logger(
        model, log_config,
        f"{model}_timing_{config.get('dataset', 'unknown')}_{config.get('repeat_index', 0)}.log",
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
