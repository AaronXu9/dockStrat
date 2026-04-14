"""
SurfDock inference module for dockstrat.

Full 4-step pipeline per complex:
  Step 1: compute protein surface (MSMS/APBS via computeTargetMesh_test_samples.py)
  Step 2: build inference input CSV
  Step 3: compute ESM pocket embeddings (sequence extraction → ESM → pocket map → .pt)
  Step 4: run diffusion sampling (accelerate launch inference_accelerate.py)

For the runsNposes benchmark dataset, steps 1–3 are pre-computed and stored under
/home/aoxu/projects/SurfDock/data/runsNposes_benchmark/, so run_dataset() skips
directly to step 4.
"""

import glob
import logging
import os
import re
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python(config):
    """Return the full path to the SurfDock conda env Python."""
    env = config.get("surfdock_env", "SurfDock")
    return f"/home/aoxu/miniconda3/envs/{env}/bin/python"


def _accelerate(config):
    env = config.get("surfdock_env", "SurfDock")
    return f"/home/aoxu/miniconda3/envs/{env}/bin/accelerate"


def _run(cmd, cwd=None, check=True, env=None):
    logger.debug("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(str(c) for c in cmd)}\n"
            f"  stderr: {result.stderr[-1000:]}"
        )
    return result


# ---------------------------------------------------------------------------
# Step 1: surface computation
# ---------------------------------------------------------------------------

def _compute_surface(data_dir, surface_out_dir, config):
    """Run surface computation for all proteins in data_dir.

    Uses _surfdock_surface_helper.py which imports computeAPBS_plinder
    (the version with correct temp-directory handling) instead of the
    buggy computeAPBS used by computeTargetMesh_test_samples.py.

    Expects data_dir/{prefix}/{prefix}_protein_processed.pdb + {prefix}_ligand.sdf
    Produces surface_out_dir/{prefix}/{prefix}_protein_processed_8A.{pdb,ply}
    """
    helper = os.path.join(os.path.dirname(__file__), "_surfdock_surface_helper.py")
    os.makedirs(surface_out_dir, exist_ok=True)
    _run([_python(config), helper,
          "--data_dir", data_dir,
          "--out_dir", surface_out_dir,
          "--surfdock_dir", config["surfdock_dir"]])


# ---------------------------------------------------------------------------
# Step 2: input CSV
# ---------------------------------------------------------------------------

def _build_input_csv(data_dir, surface_out_dir, csv_path, config):
    """Run construct_csv_input.py to produce the inference CSV."""
    surfdock_dir = config["surfdock_dir"]
    script = os.path.join(surfdock_dir, "inference_utils", "construct_csv_input.py")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    _run([_python(config), script,
          "--data_dir", data_dir,
          "--surface_out_dir", surface_out_dir,
          "--output_csv_file", csv_path])


# ---------------------------------------------------------------------------
# Step 3: ESM pocket embeddings
# ---------------------------------------------------------------------------

def _compute_esm_embeddings(csv_path, esm_base_dir, config):
    """Run all three ESM embedding steps and return path to the merged .pt file."""
    surfdock_dir = config["surfdock_dir"]
    py = _python(config)

    fasta_path = os.path.join(esm_base_dir, "sequences.fasta")
    full_emb_dir = os.path.join(esm_base_dir, "esm_embedding_output")
    pocket_emb_dir = os.path.join(esm_base_dir, "esm_embedding_pocket_output")
    pt_out = os.path.join(esm_base_dir, "esm_embedding_pocket_output_for_train",
                          "esm2_3billion_pdbbind_embeddings.pt")

    os.makedirs(full_emb_dir, exist_ok=True)
    os.makedirs(pocket_emb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(pt_out), exist_ok=True)

    # 3a: extract FASTA sequences from the input CSV
    _run([py, os.path.join(surfdock_dir, "datasets", "esm_embedding_preparation.py"),
          "--out_file", fasta_path,
          "--protein_ligand_csv", csv_path])

    # 3b: run ESM2 to get per-token embeddings
    _run([py, os.path.join(surfdock_dir, "esm", "scripts", "extract.py"),
          "esm2_t33_650M_UR50D", fasta_path, full_emb_dir,
          "--repr_layers", "33", "--include", "per_tok",
          "--truncation_seq_length", "4096"])

    # 3c: map full-protein embeddings → pocket-cropped embeddings
    _run([py, os.path.join(surfdock_dir, "datasets", "get_pocket_embedding.py"),
          "--protein_pocket_csv", csv_path,
          "--embeddings_dir", full_emb_dir,
          "--pocket_emb_save_dir", pocket_emb_dir])

    # 3d: merge individual pocket .pt files into one dict .pt
    _run([py, os.path.join(surfdock_dir, "datasets", "esm_pocket_embeddings_to_pt.py"),
          "--esm_embeddings_path", pocket_emb_dir,
          "--output_path", pt_out])

    return pt_out


# ---------------------------------------------------------------------------
# Step 4: inference
# ---------------------------------------------------------------------------

def _run_inference(csv_path, esm_pt, out_dir, config):
    """Launch inference_accelerate.py and return the SurfDock_docking_result dir."""
    surfdock_dir = config["surfdock_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # SurfDock requires this env var (used in utils/torus.py and utils/so3.py at import)
    precomputed_arrays = config.get(
        "precomputed_arrays",
        os.path.join(os.path.dirname(surfdock_dir), "precomputed", "precomputed_arrays"),
    )
    env = os.environ.copy()
    env["precomputed_arrays"] = precomputed_arrays

    cmd = [
        _accelerate(config), "launch",
        "--num_processes", str(config.get("num_gpus", 1)),
        "--main_process_port", str(config.get("main_process_port", 29510)),
        os.path.join(surfdock_dir, "inference_accelerate.py"),
        "--data_csv", csv_path,
        "--model_dir", config["diffusion_model_dir"],
        "--ckpt", "best_ema_inference_epoch_model.pt",
        "--confidence_model_dir", config["confidence_model_dir"],
        "--confidence_ckpt", "best_model.pt",
        "--save_docking_result",
        "--mdn_dist_threshold_test", str(config.get("mdn_dist_threshold", 3)),
        "--esm_embeddings_path", esm_pt,
        "--run_name", "dockstrat_run",
        "--project", "surfdock_inference",
        "--out_dir", out_dir,
        "--batch_size", str(config.get("batch_size", 40)),
        "--batch_size_molecule", "1",
        "--samples_per_complex", str(config.get("samples_per_complex", 40)),
        "--save_docking_result_number", str(config.get("num_poses", 40)),
        "--head_index", "0",
        "--tail_index", "10000",
        "--inference_mode", "evaluate",
        "--wandb_dir", os.path.join(out_dir, "wandb"),
    ]
    _run(cmd, cwd=surfdock_dir, env=env)
    return os.path.join(out_dir, "SurfDock_docking_result")


# ---------------------------------------------------------------------------
# Post-processing: collect ranked poses
# ---------------------------------------------------------------------------

def _collect_poses(docking_result_dir, pocket_stem, system_out_dir, num_poses):
    """Copy SurfDock output SDFs ranked by confidence into system_out_dir.

    SurfDock names files:
      {name}_sample_idx_{N}_rank_{R}_rmsd_{rmsd}_confidence_{conf}.sdf
    where rank_1 = highest confidence.
    """
    # The subdir name is {pocket_stem}_{ligand_stem}
    candidates = glob.glob(os.path.join(docking_result_dir, f"{pocket_stem}_*"))
    if not candidates:
        # fall back: look for any dir containing pocket_stem
        candidates = [d for d in glob.glob(os.path.join(docking_result_dir, "*"))
                      if pocket_stem in os.path.basename(d)]
    if not candidates:
        logger.warning("No output dir found for pocket_stem=%s in %s",
                       pocket_stem, docking_result_dir)
        return []

    subdir = candidates[0]
    sdf_files = glob.glob(os.path.join(subdir, "*.sdf"))

    def _rank(f):
        m = re.search(r"_rank_(\d+)_", os.path.basename(f))
        return int(m.group(1)) if m else 9999

    sdf_files.sort(key=_rank)
    os.makedirs(system_out_dir, exist_ok=True)
    rank_files = []
    for i, src in enumerate(sdf_files[:num_poses], 1):
        dst = os.path.join(system_out_dir, f"rank{i}.sdf")
        shutil.copy(src, dst)
        rank_files.append(dst)
    return rank_files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dataset(config):
    """Run SurfDock on the runsNposes dataset using pre-computed surface/ESM data."""
    import pandas as pd

    inputs_csv = config["inputs_csv"]
    esm_pt = config["esm_embeddings_path"]
    docking_result_base = config["docking_result_dir"]
    output_dir = config["output_dir"]
    num_poses = config.get("num_poses", 10)
    skip_existing = config.get("skip_existing", True)

    # Check which systems still need processing
    df = pd.read_csv(inputs_csv)
    if skip_existing:
        pending = df[df["pocket_path"].apply(
            lambda p: not os.path.exists(
                os.path.join(output_dir,
                             re.sub(r"_8A$", "", os.path.splitext(os.path.basename(p))[0]),
                             "rank1.sdf")))]
    else:
        pending = df

    if pending.empty:
        logger.info("[INFO] All systems already processed. Skipping.")
        return

    # Write a filtered CSV for the pending systems
    tmp_csv = os.path.join(docking_result_base, "pending_inputs.csv")
    os.makedirs(docking_result_base, exist_ok=True)
    pending.to_csv(tmp_csv, index=False)

    docking_result_dir = _run_inference(tmp_csv, esm_pt, docking_result_base, config)

    # Post-process: collect ranked poses per system
    for _, row in pending.iterrows():
        pocket_stem = os.path.splitext(os.path.basename(row["pocket_path"]))[0]
        system_id = re.sub(r"_8A$", "", pocket_stem)
        system_out = os.path.join(output_dir, system_id)
        _collect_poses(docking_result_dir, pocket_stem, system_out, num_poses)

    logger.info("[INFO] SurfDock dataset run complete. Results in %s", output_dir)


def run_single(protein, ligand, output_dir, config=None, prefix=None, **kwargs) -> str:
    """Run the full SurfDock pipeline for one protein–ligand pair.

    Steps run automatically in a temporary directory:
      1. Surface computation
      2. Input CSV construction
      3. ESM pocket embeddings
      4. Diffusion inference

    Parameters
    ----------
    protein : path to protein PDB
    ligand  : path to ligand SDF
    output_dir : base output directory (results go to output_dir/surfdock/{prefix}/)
    config  : dict of config overrides (merged with surfdock_inference.yaml defaults)
    prefix  : used as the system identifier; defaults to protein filename stem
    """
    from dockstrat.engine import _load_config
    cfg = _load_config("surfdock", config or {})
    cfg.update(kwargs)

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(protein))[0]

    num_poses = kwargs.get("num_poses", cfg.get("num_poses", 10))
    system_out = os.path.join(output_dir, "surfdock", prefix)

    if cfg.get("skip_existing") and os.path.exists(os.path.join(system_out, "rank1.sdf")):
        logger.info("[INFO] Skipping %s (already done)", prefix)
        return system_out

    with tempfile.TemporaryDirectory(prefix=f"surfdock_{prefix}_") as tmp:
        # --- Step 1: Create test_samples directory structure ---
        tmp_data_dir = os.path.join(tmp, "data")
        system_data_dir = os.path.join(tmp_data_dir, prefix)
        os.makedirs(system_data_dir, exist_ok=True)
        shutil.copy(protein, os.path.join(system_data_dir, f"{prefix}_protein_processed.pdb"))
        shutil.copy(ligand, os.path.join(system_data_dir, f"{prefix}_ligand.sdf"))

        tmp_surface_dir = os.path.join(tmp, "surface")
        tmp_csv = os.path.join(tmp, "csv", "input.csv")
        tmp_esm_dir = os.path.join(tmp, "esm")
        tmp_out = os.path.join(tmp, "inference_out")

        # --- Step 1: Compute surface ---
        logger.info("[INFO] Computing protein surface for %s ...", prefix)
        _compute_surface(tmp_data_dir, tmp_surface_dir, cfg)

        # --- Step 2: Build input CSV ---
        _build_input_csv(tmp_data_dir, tmp_surface_dir, tmp_csv, cfg)

        # --- Step 3: ESM pocket embeddings ---
        logger.info("[INFO] Computing ESM embeddings for %s ...", prefix)
        esm_pt = _compute_esm_embeddings(tmp_csv, tmp_esm_dir, cfg)

        # --- Step 4: Run inference ---
        logger.info("[INFO] Running SurfDock inference for %s ...", prefix)
        docking_result_dir = _run_inference(tmp_csv, esm_pt, tmp_out, cfg)

        # --- Post-process: collect ranked poses ---
        # pocket stem = "{prefix}_protein_processed_8A"
        pocket_stem = f"{prefix}_protein_processed_8A"
        os.makedirs(system_out, exist_ok=True)
        _collect_poses(docking_result_dir, pocket_stem, system_out, num_poses)

    logger.info("[INFO] SurfDock docking complete. Results in %s", system_out)
    return system_out


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf

    os.environ.setdefault(
        "PROJECT_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    )
    cfg_path = os.path.join(
        os.path.dirname(__file__), "../../dockstrat_config/model/surfdock_inference.yaml"
    )
    config = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    run_dataset(config)
