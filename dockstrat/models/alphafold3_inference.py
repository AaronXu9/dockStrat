"""AlphaFold3 docking wrapper.

Runs AF3 in no-MSA / single-sequence mode via a subprocess in a dedicated
conda env, then extracts ligand SDFs from the predicted mmCIFs using the
known input SMILES for bond-order recovery.

Public surface (used by dockstrat.engine):
  - run_single(protein, ligand, output_dir, config=None, prefix=None, **kwargs)
  - run_dataset(config)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional


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


def _build_af3_input_json(system_id: str, pdb_path: str, sdf_path: str) -> Dict:
    """Construct the AF3 input-JSON dict for a single protein + ligand pair.

    Uses no-MSA mode: each protein chain's ``unpairedMsa`` field carries
    only the query sequence as a single-row A3M, ``pairedMsa`` is empty,
    and ``templates`` is the empty list.
    """
    from dockstrat.utils.sequence import extract_protein_sequence

    sequences = extract_protein_sequence(pdb_path)
    if not sequences:
        raise ValueError(f"No protein sequences found in {pdb_path}")

    smiles = _smiles_from_sdf(sdf_path)

    entries: List[Dict] = []
    for i, seq in enumerate(sequences):
        entries.append({
            "protein": {
                "id": chr(ord("A") + i),
                "sequence": seq,
                "unpairedMsa": f">query\n{seq}\n",
                "pairedMsa": "",
                "templates": [],
            }
        })
    entries.append({"ligand": {"id": "L", "smiles": smiles}})

    return {
        "name": system_id,
        "modelSeeds": [1234],
        "sequences": entries,
        "dialect": "alphafold3",
        "version": 2,
    }


def _extract_ligand_from_cif(
    cif_path: str | Path,
    template_mol: "Chem.Mol",
) -> "Chem.Mol":
    """Pull the ligand (chain L) out of an AF3 mmCIF and recover bond orders.

    Strategy: parse the mmCIF with Biopython's MMCIF2Dict (a simple tokenizer
    that works with any minimal mmCIF loop), collect all atoms on chain L (the
    AF3 ligand chain) into a synthetic PDB block, parse that block with RDKit
    (which infers connectivity from distances), then call
    AssignBondOrdersFromTemplate with the known SMILES template to recover the
    correct bond orders.

    AF3 may label ligand atoms as either ATOM or HETATM depending on the
    upstream version; all atoms on chain L are collected regardless of the
    group_PDB label.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mmcif_dict = MMCIF2Dict(str(cif_path))

    # Extract per-atom arrays; chain id may live in auth_asym_id or label_asym_id
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
        if chain_list[i] != "L":
            continue
        resname = (resname_list[i] if i < len(resname_list) else "LIG")[:3].ljust(3)
        name = (atom_name_list[i] if i < len(atom_name_list) else "X")[:4].ljust(4)
        elem = (elem_list[i] if i < len(elem_list) else name.strip()[0]).strip()
        x = float(x_list[i])
        y = float(y_list[i])
        z = float(z_list[i])
        pdb_lines.append(
            f"HETATM{atom_idx:>5} {name} {resname} L{1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
            f"{elem.rjust(2)}"
        )
        atom_idx += 1

    if not pdb_lines:
        raise ValueError(f"No ligand atoms (chain L) found in {cif_path}")

    pdb_block = "\n".join(pdb_lines) + "\nEND\n"
    raw_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if raw_mol is None:
        raise ValueError(f"Failed to parse synthesized PDB block for ligand from {cif_path}")

    try:
        Chem.SanitizeMol(raw_mol)
    except Exception:
        # Sanitization may fail before bond orders are recovered; that's OK.
        pass

    annotated = AllChem.AssignBondOrdersFromTemplate(template_mol, raw_mol)
    return annotated


def _extract_ranked_ligand_sdfs(
    af3_system_dir: str | Path,
    name: str,
    smiles: str,
    out_dir: str | Path,
    num_poses: int,
) -> int:
    """Read AF3's ranking CSV, sort by score desc, write top-N rank{i}.sdf.

    AF3 v3.0.1 prefixes all output filenames with the job name:
    ``{name}_ranking_scores.csv`` and
    ``seed-{s}_sample-{n}/{name}_seed-{s}_sample-{n}_model.cif``.

    Walks the per-system AF3 output directory, reads the ranking CSV, and
    writes the top ``num_poses`` ligand poses as ``rank{1..N}.sdf`` in
    ``out_dir``. Bond orders are recovered from the input ``smiles`` template.

    Returns the number of SDFs successfully written.

    ``out_dir`` is created with ``mkdir(parents=True, exist_ok=True)`` so it
    must be a directory path; passing a path that points at an existing file
    will raise ``FileExistsError``.

    Passing ``num_poses=0`` is a no-op: the function returns 0 without
    writing any files or raising.
    """
    import pandas as pd
    from rdkit import Chem

    af3_system_dir = Path(af3_system_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = af3_system_dir / f"{name}_ranking_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"AF3 ranking_scores.csv not found at {scores_path}")

    scores = pd.read_csv(scores_path)
    scores = scores.sort_values("ranking_score", ascending=False).reset_index(drop=True)

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse template SMILES: {smiles}")

    written = 0
    for _, row in scores.head(num_poses).iterrows():
        seed = int(row["seed"])
        sample = int(row["sample"])
        cif = af3_system_dir / f"seed-{seed}_sample-{sample}" / f"{name}_seed-{seed}_sample-{sample}_model.cif"
        if not cif.exists():
            continue
        try:
            mol = _extract_ligand_from_cif(cif, template_mol=template)
        except Exception:
            continue
        written += 1
        Chem.MolToMolFile(mol, str(out_dir / f"rank{written}.sdf"))
    return written


def _run_af3_subprocess(
    json_path: str | Path,
    out_dir: str | Path,
    config: dict,
) -> None:
    """Invoke AF3's run_alphafold.py via the dedicated conda env's Python.

    Raises ``subprocess.CalledProcessError`` on nonzero exit and
    ``subprocess.TimeoutExpired`` on hard timeout. Both are caught
    by callers (run_single / run_dataset).

    CLI flag names verified against AF3 v3.0.1 (``run_alphafold.py --help``):
    all eight flags (``--json_path``, ``--output_dir``, ``--model_dir``,
    ``--norun_data_pipeline``, ``--run_inference``, ``--num_diffusion_samples``,
    ``--num_seeds``, ``--num_recycles``) match exactly.
    """
    import subprocess

    env_python = Path(config["alphafold3_env"]) / "bin" / "python"
    run_script = Path(config["alphafold3_dir"]) / "run_alphafold.py"

    cmd = [
        str(env_python), str(run_script),
        f"--json_path={json_path}",
        f"--output_dir={out_dir}",
        f"--model_dir={config['model_dir']}",
        "--norun_data_pipeline",
        "--run_inference=true",
        f"--num_diffusion_samples={config.get('num_samples', 5)}",
        f"--num_recycles={config.get('num_recycles', 10)}",
    ]
    # --num_seeds generates N random seeds from the single seed in the JSON.
    # AF3 requires num_seeds > 1; when num_seeds=1 (the default), omit the
    # flag entirely and AF3 uses the modelSeeds array from the JSON as-is.
    num_seeds = int(config.get("num_seeds", 1))
    if num_seeds > 1:
        cmd.append(f"--num_seeds={num_seeds}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("cuda_device_index", 0))
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # pip-installed NVIDIA wheels put .so files under site-packages/nvidia/*/lib
    # but they are not on LD_LIBRARY_PATH by default. JAX needs them to load
    # the CUDA plugin (cusparse, cublas, cudnn, etc.).
    nvidia_base = Path(config["alphafold3_env"]) / "lib" / "python3.12" / "site-packages" / "nvidia"
    if nvidia_base.is_dir():
        nvidia_lib_dirs = [
            str(p / "lib") for p in nvidia_base.iterdir()
            if (p / "lib").is_dir()
        ]
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(nvidia_lib_dirs + ([existing_ld] if existing_ld else []))

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
    """Dock a single protein+ligand pair with AlphaFold3.

    Writes the AF3 input JSON, runs the subprocess, then extracts the top-N
    ranked ligand poses as ``rank{1..N}.sdf`` directly into ``output_dir``.
    Returns ``output_dir`` on success.
    """
    import json

    config = config or {}

    # Validate required config keys at the API boundary so missing keys
    # surface a clear error instead of a bare KeyError from inside a helper.
    for required_key in ("alphafold3_env", "alphafold3_dir", "model_dir"):
        if required_key not in config:
            raise ValueError(
                f"alphafold3 run_single: missing required config key '{required_key}'"
            )

    prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Build the input JSON
    payload = _build_af3_input_json(prefix, protein, ligand)
    json_path = Path(output_dir) / f"{prefix}_af3_input.json"
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # 2. Run AF3 (writes to output_dir/{prefix.lower()}/...)
    _run_af3_subprocess(json_path, Path(output_dir), config)

    # 3. Extract the ranked ligand SDFs
    # AF3 writes to {output_dir}/{name}/ using the exact name from the JSON
    # (no lowercasing). The name in the JSON is set to `prefix` by
    # _build_af3_input_json.
    smiles = _smiles_from_sdf(ligand)
    af3_system_dir = Path(output_dir) / prefix
    if not af3_system_dir.exists():
        raise FileNotFoundError(
            f"AF3 produced no output directory at {af3_system_dir}"
        )
    num_poses = int(config.get("num_poses_to_keep", config.get("num_samples", 5)))
    _extract_ranked_ligand_sdfs(
        af3_system_dir,
        name=prefix,
        smiles=smiles,
        out_dir=Path(output_dir),
        num_poses=num_poses,
    )
    return output_dir


def run_dataset(config: dict) -> None:
    """Run AlphaFold3 over an entire dataset directory.

    Iterates the per-system folders under ``config['input_dir']``. Each
    folder is expected to contain ``{sys_id}_protein.pdb`` and
    ``{sys_id}_ligand.sdf``. Writes ``rank{N}.sdf`` files into
    ``config['output_dir']/{sys_id}/`` per system. On any failure, logs and
    moves on.
    """
    import time
    import traceback

    from dockstrat.utils.log import get_custom_logger

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    skip_existing = config.get("skip_existing", True)
    max_num_inputs = config.get("max_num_inputs", None)

    os.makedirs(output_dir, exist_ok=True)
    # get_custom_logger reads a flat `log_dir` key but the YAML nests it under
    # `logging.log_dir`; flatten before passing so logs land in the configured
    # directory instead of the caller's cwd.
    logging_cfg = config.get("logging") or {}
    log_config = dict(config)
    if logging_cfg.get("log_dir") and "log_dir" not in log_config:
        log_config["log_dir"] = logging_cfg["log_dir"]
    logger = get_custom_logger(
        "alphafold3", log_config,
        f"alphafold3_timing_{config.get('dataset', 'unknown')}_{config.get('repeat_index', 0)}.log",
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
