# -------------------------------------------------------------------------------------------------------------------------------------
# Originally from PoseBench (https://github.com/BioinfoMachineLearning/PoseBench), adapted for CogLigandBench
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import subprocess  # nosec
import uuid
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from dockstrat import register_custom_omegaconf_resolvers
from dockstrat.utils.utils import find_ligand_files, find_protein_files

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="dynamicbind_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained DynamicBind model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    pocket_suffix = "_bs_cropped" if cfg.pocket_only_baseline else ""
    pocket_only_suffix = "_pocket_only" if cfg.pocket_only_baseline else ""

    os.environ["MKL_THREADING_LAYER"] = "GNU"  # address MKL threading issue
    protein_filepaths = find_protein_files(Path(cfg.input_data_dir + pocket_suffix))
    ligand_filepaths = [
        ligand_filepath
        for ligand_filepath in find_ligand_files(Path(cfg.input_ligand_csv_dir), extension="csv")
        if any(
            ligand_filepath.stem.split("_")[0] in protein_filepath.stem
            for protein_filepath in protein_filepaths
        )
    ]
    if len(protein_filepaths) > len(ligand_filepaths):
        protein_filepaths = [
            protein_filepath
            for protein_filepath in protein_filepaths
            if (
                cfg.dataset == "dockgen"
                and any(
                    "_".join(ligand_filepath.stem.split("_")[:4]) in protein_filepath.stem
                    for ligand_filepath in ligand_filepaths
                )
            )
            or (
                cfg.dataset != "dockgen"
                and any(
                    ligand_filepath.stem.split("_")[0] in protein_filepath.stem
                    for ligand_filepath in ligand_filepaths
                )
            )
        ]
    for ligand_filepath in ligand_filepaths:
        if len(protein_filepaths) < len(ligand_filepaths):
            protein_filepaths.append(
                next(
                    protein_filepath
                    for protein_filepath in protein_filepaths
                    if (
                        "_".join(cfg.dataset == "dockgen" and ligand_filepath.stem.split("_")[:4])
                        in protein_filepath.stem
                    )
                    or (
                        cfg.dataset != "dockgen"
                        and ligand_filepath.stem.split("_")[0] in protein_filepath.stem
                    )
                )
            )
    assert len(protein_filepaths) == len(
        ligand_filepaths
    ), f"Number of protein ({len(protein_filepaths)}) and ligand ({len(ligand_filepaths)}) files must be equal."
    protein_filepaths = sorted(protein_filepaths)
    ligand_filepaths = sorted(ligand_filepaths)
    if cfg.max_num_inputs and protein_filepaths and ligand_filepaths:
        protein_filepaths = protein_filepaths[: cfg.max_num_inputs]
        ligand_filepaths = ligand_filepaths[: cfg.max_num_inputs]
        assert (
            len(protein_filepaths) > 0 and len(ligand_filepaths) > 0
        ), "No input files found after subsetting with `max_num_inputs`."
    for protein_filepath, ligand_filepath in zip(protein_filepaths, ligand_filepaths):
        assert (
            protein_filepath.stem.split("_")[0] == ligand_filepath.stem.split("_")[0]
        ), "Protein and ligand files must have the same ID."
        if cfg.dataset == "dockgen":
            assert "_".join(protein_filepath.stem.split("_")[:4]) == "_".join(
                ligand_filepath.stem.split("_")[:4]
            ), "Protein and ligand files must have the same ID."
        ligand_output_filepaths = list(
            glob.glob(
                os.path.join(
                    cfg.dynamicbind_exec_dir,
                    "inference",
                    "outputs",
                    "results",
                    f"{cfg.dataset}{pocket_only_suffix}_{ligand_filepath.stem}_{cfg.repeat_index}",
                    "index0_idx_0",
                    "rank1_ligand*.sdf",
                )
            )
        )
        if cfg.skip_existing and ligand_output_filepaths:
            logger.info(
                f"Skipping inference for completed protein `{protein_filepath}` and ligand `{ligand_filepath}`."
            )
            continue
        unique_cache_id = uuid.uuid4()
        unique_cache_path = (
            str(cfg.cache_path)
            + f"_{cfg.dataset}{pocket_only_suffix}_{ligand_filepath.stem}_{cfg.repeat_index}_{unique_cache_id}"
        )
        try:
            subprocess.run(
                [
                    cfg.python_exec_path,
                    os.path.join(cfg.dynamicbind_exec_dir, "run_single_protein_inference.py"),
                    protein_filepath,
                    ligand_filepath,
                    "--samples_per_complex",
                    str(cfg.samples_per_complex),
                    "--savings_per_complex",
                    str(cfg.savings_per_complex),
                    "--inference_steps",
                    str(cfg.inference_steps),
                    "--batch_size",
                    str(cfg.batch_size),
                    "--cache_path",
                    unique_cache_path,
                    "--header",
                    str(cfg.header)
                    + f"{pocket_only_suffix}_{ligand_filepath.stem}"
                    + f"_{cfg.repeat_index}",
                    "--device",
                    str(cfg.cuda_device_index),
                    "--python",
                    str(cfg.python_exec_path),
                    "--relax_python",
                    str(cfg.python_exec_path),
                    "--results",
                    str(os.path.join(cfg.dynamicbind_exec_dir, "inference", "outputs", "results")),
                    "--no_relax",  # NOTE: must be set to `True` for CogLigandBench since method-native relaxation is not supported
                    "--paper",  # NOTE: must be set to `True` for CogLigandBench since only the paper weights are available
                ],
                check=True,
            )  # nosec
        except Exception as e:
            logger.error(
                f"Error occurred while running DynamicBind inference for protein `{protein_filepath}` and ligand `{ligand_filepath}`: {e}. Skipping..."
            )
            continue
        logger.info(
            f"DynamicBind inference for protein `{protein_filepath}` and `{ligand_filepath}` complete."
        )


# ---------------------------------------------------------------------------
# Public API (OmegaConf-based, no Hydra required)
# ---------------------------------------------------------------------------

def _run_dynamicbind_subprocess(
    protein_filepath,
    ligand_filepath,
    header: str,
    config: dict,
    cache_path: str,
):
    """Run DynamicBind subprocess for a single protein-ligand pair."""
    subprocess.run(
        [
            config["python_exec_path"],
            os.path.join(config["dynamicbind_exec_dir"], "run_single_protein_inference.py"),
            str(protein_filepath),
            str(ligand_filepath),
            "--samples_per_complex", str(config.get("samples_per_complex", 40)),
            "--savings_per_complex", str(config.get("savings_per_complex", 1)),
            "--inference_steps", str(config.get("inference_steps", 20)),
            "--batch_size", str(config.get("batch_size", 5)),
            "--cache_path", cache_path,
            "--header", header,
            "--device", str(config.get("cuda_device_index", 0)),
            "--python", config["python_exec_path"],
            "--relax_python", config["python_exec_path"],
            "--results", str(os.path.join(config["dynamicbind_exec_dir"], "inference", "outputs", "results")),
            "--no_relax",
            "--paper",
        ],
        check=True,
    )  # nosec


def run_dataset(config: dict):
    """Run DynamicBind over a benchmark dataset via a standard inputs CSV.

    The CSV must have columns: complex_name, protein_path, ligand_path (SDF).
    SMILES is extracted from each SDF and passed to DynamicBind as a temp CSV.
    """
    import uuid
    import tempfile
    from omegaconf import OmegaConf
    from rdkit import Chem
    import pandas as pd
    import time
    from dockstrat.utils.log import get_custom_logger

    os.environ["MKL_THREADING_LAYER"] = "GNU"
    logger = get_custom_logger("dynamicbind", config, f"dynamicbind_timing_{config['repeat_index']}.log")
    inputs_df = pd.read_csv(config["inputs_csv"])

    for _, row in inputs_df.iterrows():
        system_id = row["complex_name"]
        protein_pdb = row["protein_path"]
        ligand_sdf = row["ligand_path"]

        if not (os.path.exists(protein_pdb) and os.path.exists(ligand_sdf)):
            logger.warning(f"Missing files for {system_id}. Skipping.")
            continue

        out_dir = os.path.join(config["output_dir"], system_id)
        if config.get("skip_existing", False) and os.path.exists(out_dir):
            existing = glob.glob(os.path.join(out_dir, "**", "rank1_ligand*.sdf"), recursive=True)
            if existing:
                logger.info(f"Skipping {system_id} — already done.")
                continue

        # Extract SMILES and write temp ligand CSV for DynamicBind
        mol = next(Chem.SDMolSupplier(ligand_sdf, removeHs=True), None)
        if mol is None:
            logger.error(f"Cannot read ligand for {system_id}. Skipping.")
            continue
        smiles = Chem.MolToSmiles(mol)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tf:
            tf.write(f"ligand\n{smiles}\n")
            ligand_csv = tf.name

        cache_path = config.get("cache_path", "/tmp/dynamicbind_cache")
        unique_cache = f"{cache_path}_{system_id}_{uuid.uuid4()}"
        header = f"{config.get('benchmark', 'dataset')}_{system_id}_{config['repeat_index']}"

        start = time.time()
        try:
            _run_dynamicbind_subprocess(protein_pdb, ligand_csv, header, config, unique_cache)

            # Copy results to canonical output dir
            results_glob = glob.glob(
                os.path.join(config["dynamicbind_exec_dir"], "inference", "outputs", "results",
                             header, "index0_idx_0", "rank*.sdf")
            )
            os.makedirs(out_dir, exist_ok=True)
            for f in results_glob:
                import shutil
                shutil.copy(f, out_dir)

            logger.info(f"{system_id},{time.time() - start:.2f}")
        except Exception as e:
            logger.error(f"DynamicBind failed for {system_id}: {e}")
        finally:
            os.unlink(ligand_csv)


def run_single(
    protein: str,
    ligand: str,
    output_dir: str,
    config: dict = None,
    prefix: str = None,
    **kwargs,
) -> str:
    """Run DynamicBind on a single protein (PDB) + ligand (SDF) pair.

    SMILES is extracted from the SDF. Outputs (rank*.sdf) are copied to output_dir.
    """
    import uuid
    import tempfile
    import shutil
    from rdkit import Chem

    config = config or {}
    prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
    os.makedirs(output_dir, exist_ok=True)
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    mol = next(Chem.SDMolSupplier(ligand, removeHs=True), None)
    if mol is None:
        raise ValueError(f"Cannot read molecule from {ligand}")
    smiles = Chem.MolToSmiles(mol)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tf:
        tf.write(f"ligand\n{smiles}\n")
        ligand_csv = tf.name

    cache_path = config.get("cache_path", "/tmp/dynamicbind_cache")
    unique_cache = f"{cache_path}_{prefix}_{uuid.uuid4()}"
    header = f"single_{prefix}_{config.get('repeat_index', 0)}"

    try:
        _run_dynamicbind_subprocess(protein, ligand_csv, header, config, unique_cache)

        results_glob = glob.glob(
            os.path.join(config.get("dynamicbind_exec_dir", "forks/DynamicBind"),
                         "inference", "outputs", "results", header, "index0_idx_0", "rank*.sdf")
        )
        for f in results_glob:
            shutil.copy(f, output_dir)

        print(f"[INFO] DynamicBind docking complete. Results in {output_dir}")
    except Exception as e:
        print(f"[ERROR] DynamicBind docking failed: {e}")
        raise
    finally:
        os.unlink(ligand_csv)

    return output_dir


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import sys
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
    run_dataset(cfg)
