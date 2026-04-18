"""Protenix docking wrapper.

Runs ``protenix pred`` via subprocess in a dedicated conda env, then extracts
ligand SDFs from the predicted mmCIFs using the known input SMILES for
bond-order recovery.

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


def _build_protenix_input_json(
    system_id: str,
    pdb_path: str,
    sdf_path: str,
) -> Tuple[List[Dict], str]:
    """Construct the Protenix input-JSON array for a single protein + ligand pair.

    Returns ``(json_list, ligand_chain_id)`` where:
    - ``json_list`` is a single-element list matching Protenix's input format
    - ``ligand_chain_id`` is the predicted chain in the output CIF (e.g. ``"B0"``)

    Protenix uses ``proteinChain`` (not ``protein``) and ``ligand.ligand`` for
    the SMILES field.  Omitting MSA fields triggers single-sequence mode.
    """
    from dockstrat.utils.sequence import extract_protein_sequence

    sequences = extract_protein_sequence(pdb_path)
    if not sequences:
        raise ValueError(f"No protein sequences found in {pdb_path}")

    smiles = _smiles_from_sdf(sdf_path)

    entries: List[Dict] = []
    for i, seq in enumerate(sequences):
        entries.append({
            "proteinChain": {
                "sequence": seq,
                "count": 1,
            }
        })

    entries.append({
        "ligand": {
            "ligand": smiles,
            "count": 1,
        }
    })

    # Protenix assigns chain IDs as {letter}0: A0, B0, C0, ...
    ligand_chain_id = f"{chr(ord('A') + len(sequences))}0"

    return [{"name": system_id, "sequences": entries}], ligand_chain_id


def _discover_ligand_chain_id(
    cif_path: str | Path,
    expected_chain_id: str,
    template_mol: "Chem.Mol",
) -> str:
    """Determine the actual ligand chain ID in a Protenix CIF.

    Strategy:
    1. If ``expected_chain_id`` exists and has HETATM records, return it.
    2. Otherwise, scan all chains for those with only HETATM records.
    3. For each candidate, attempt ``AssignBondOrdersFromTemplate`` — return
       the first chain where this succeeds.
    4. Raise ``ValueError`` if no ligand chain is found.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    mmcif_dict = MMCIF2Dict(str(cif_path))

    chain_list = mmcif_dict.get(
        "_atom_site.auth_asym_id",
        mmcif_dict.get("_atom_site.label_asym_id", []),
    )
    group_list = mmcif_dict.get("_atom_site.group_PDB", [])

    # Build chain → group_PDB mapping
    chain_groups: Dict[str, set] = {}
    for c, g in zip(chain_list, group_list):
        chain_groups.setdefault(c, set()).add(g)

    # Strategy 1: check expected chain
    if expected_chain_id in chain_groups and "HETATM" in chain_groups[expected_chain_id]:
        return expected_chain_id

    # Strategy 2: find HETATM-only chains and try bond-order recovery
    hetatm_chains = [
        c for c, gs in chain_groups.items()
        if gs == {"HETATM"}
    ]

    for candidate in hetatm_chains:
        try:
            _extract_ligand_from_cif(cif_path, candidate, template_mol)
            return candidate
        except Exception:
            continue

    raise ValueError(
        f"No ligand chain found in {cif_path}. "
        f"Expected '{expected_chain_id}', available chains: {list(chain_groups.keys())}"
    )


def _extract_ligand_from_cif(
    cif_path: str | Path,
    ligand_chain_id: str,
    template_mol: "Chem.Mol",
) -> "Chem.Mol":
    """Pull the ligand out of a Protenix mmCIF and recover bond orders.

    Protenix uses multi-character chain IDs (e.g. ``B0``).  The synthesized
    PDB block uses a single-character placeholder (``L``) since PDB format
    only supports one-char chain IDs.
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
        # Use single-char placeholder 'L' since PDB format requires 1-char chain ID
        pdb_lines.append(
            f"HETATM{atom_idx:>5} {name} {resname} L{1:>4}    "
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
    protenix_job_dir: str | Path,
    job_name: str,
    smiles: str,
    ligand_chain_id: str,
    out_dir: str | Path,
    num_poses: int,
) -> int:
    """Rank Protenix predictions by confidence score, write top-N rank{i}.sdf.

    Raw Protenix output structure::

        {job_dir}/seed_{seed}/predictions/
            {job_name}_seed_{seed}_summary_confidence_sample_{sample}.json
            {job_name}_seed_{seed}_sample_{sample}.cif
            {sample}/
                {chain}.{resname}.sdf    (pre-extracted ligand SDF)

    Reads ``ranking_score`` from each confidence JSON, sorts descending,
    then copies the pre-extracted ligand SDF (or falls back to CIF
    extraction) for the top-N poses.

    Returns the number of SDFs successfully written.
    """
    from rdkit import Chem

    protenix_job_dir = Path(protenix_job_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all confidence JSONs across seed directories.
    # Protenix v2.0.0 names files as {job}_summary_confidence_sample_{N}.json
    # Older versions use {job}_seed_{seed}_summary_confidence_sample_{N}.json
    scored: List[tuple] = []  # (ranking_score, pred_dir, seed_str, sample)
    for seed_dir in sorted(protenix_job_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed_str = seed_dir.name[len("seed_"):]
        pred_dir = seed_dir / "predictions"
        if not pred_dir.is_dir():
            continue
        for conf_json in sorted(pred_dir.glob("*_summary_confidence_sample_*.json")):
            match = re.search(r"_sample_(\d+)\.json$", conf_json.name)
            if match is None:
                continue
            sample = int(match.group(1))
            with open(conf_json) as fh:
                data = json.load(fh)
            score = float(data.get("ranking_score", 0.0))
            scored.append((score, pred_dir, seed_str, sample))

    if not scored:
        raise FileNotFoundError(
            f"No confidence JSONs found in {protenix_job_dir}/seed_*/predictions/"
        )

    scored.sort(key=lambda x: x[0], reverse=True)

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse template SMILES: {smiles}")

    written = 0
    actual_chain_id = None
    for score, pred_dir, seed_str, sample in scored[:num_poses]:
        # Strategy 1: use pre-extracted SDF from Protenix (preferred)
        sample_dir = pred_dir / str(sample)
        sdf_files = list(sample_dir.glob("*.sdf")) if sample_dir.is_dir() else []
        if sdf_files:
            try:
                mol = next(Chem.SDMolSupplier(str(sdf_files[0]), removeHs=False), None)
                if mol is not None:
                    written += 1
                    writer = Chem.SDWriter(str(out_dir / f"rank{written}.sdf"))
                    writer.write(mol)
                    writer.close()
                    continue
            except Exception:
                pass

        # Strategy 2: fall back to CIF extraction
        # v2.0.0: {job_name}_sample_{sample}.cif
        # older:  {job_name}_seed_{seed}_sample_{sample}.cif
        cif = pred_dir / f"{job_name}_sample_{sample}.cif"
        if not cif.exists():
            cif = pred_dir / f"{job_name}_seed_{seed_str}_sample_{sample}.cif"
        if not cif.exists():
            continue
        try:
            if actual_chain_id is None:
                actual_chain_id = _discover_ligand_chain_id(cif, ligand_chain_id, template)
            mol = _extract_ligand_from_cif(cif, actual_chain_id, template_mol=template)
        except Exception:
            continue

        written += 1
        writer = Chem.SDWriter(str(out_dir / f"rank{written}.sdf"))
        writer.write(mol)
        writer.close()

    return written


def _run_protenix_subprocess(
    json_path: str | Path,
    out_dir: str | Path,
    config: dict,
) -> None:
    """Invoke ``protenix pred`` via subprocess.

    Raises ``subprocess.CalledProcessError`` on nonzero exit and
    ``subprocess.TimeoutExpired`` on hard timeout.
    """
    import subprocess

    protenix_binary = config["protenix_binary"]
    model_name = config.get("model_name", "protenix-v2")

    cmd = [
        str(protenix_binary), "pred",
        "-i", str(json_path),
        "-o", str(out_dir),
        "-n", model_name,
        "--use_msa", "False",
    ]

    seeds = config.get("seeds")
    if seeds is not None:
        cmd.extend(["-s", str(seeds)])

    # Inference knobs
    if "num_cycles" in config:
        cmd.extend(["-c", str(config["num_cycles"])])
    if "num_steps" in config:
        cmd.extend(["-p", str(config["num_steps"])])
    if "num_samples" in config:
        cmd.extend(["-e", str(config["num_samples"])])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("cuda_device_index", 0))
    # Use torch LayerNorm to avoid CUDA kernel compilation requirement
    env.setdefault("LAYERNORM_TYPE", "torch")

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
    """Dock a single protein+ligand pair with Protenix.

    Writes the Protenix input JSON, runs the subprocess, then extracts the
    top-N ranked ligand poses as ``rank{1..N}.sdf`` directly into
    ``output_dir``.  Returns ``output_dir`` on success.
    """
    config = config or {}

    if "protenix_binary" not in config:
        raise ValueError("protenix run_single: missing required config key 'protenix_binary'")

    prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Build the input JSON
    payload, ligand_chain_id = _build_protenix_input_json(prefix, protein, ligand)
    json_path = Path(output_dir) / f"{prefix}_protenix_input.json"
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # 2. Run Protenix
    _run_protenix_subprocess(json_path, Path(output_dir), config)

    # 3. Locate the raw output directory
    # Protenix writes to {output_dir}/{job_name}/seed_{seed}/predictions/...
    protenix_job_dir = Path(output_dir) / prefix
    if not protenix_job_dir.exists():
        raise FileNotFoundError(
            f"Protenix produced no output directory at {protenix_job_dir}"
        )

    # 4. Extract the ranked ligand SDFs
    smiles = _smiles_from_sdf(ligand)
    num_poses = int(config.get("num_poses_to_keep", 5))
    _extract_ranked_ligand_sdfs(
        protenix_job_dir,
        job_name=prefix,
        smiles=smiles,
        ligand_chain_id=ligand_chain_id,
        out_dir=Path(output_dir),
        num_poses=num_poses,
    )
    return output_dir


def run_dataset(config: dict) -> None:
    """Run Protenix over an entire dataset directory.

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
        "protenix", log_config,
        f"protenix_timing_{config.get('dataset', 'unknown')}_{config.get('repeat_index', 0)}.log",
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
