"""
Unified docking engine for dockstrat.

Usage from a notebook or script:

    from dockstrat import dock_engine

    # Dataset mode — run over a full benchmark
    dock_engine('vina',  dataset='runsNposes')
    dock_engine('gnina', dataset='runsNposes', repeat_index=1)
    dock_engine('chai',  dataset='runsNposes', cuda_device_index=1)

    # Single-molecule mode — dock one protein-ligand pair
    dock_engine('vina',        protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('gnina',       protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('chai',        protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('dynamicbind', protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('unidock2',    protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('surfdock',    protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('alphafold3',  protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('boltz1',      protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('boltz2',      protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('protenix',    protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')

    # Pass any config override as a kwarg:
    dock_engine('vina', dataset='runsNposes', top_n=5, exhaustiveness=16)

Available methods: vina, gnina, chai, dynamicbind, unidock2, surfdock, alphafold3, boltz1, boltz2, protenix
"""

import importlib
import os
from typing import Optional

import rootutils
from omegaconf import OmegaConf

_PROJECT_ROOT = str(rootutils.find_root(search_from=__file__, indicator=".project-root"))

SUPPORTED_METHODS = ("vina", "gnina", "chai", "dynamicbind", "unidock2", "surfdock", "alphafold3", "boltz1", "boltz2", "protenix")

_CONFIG_PATHS = {
    "vina":         os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "vina_inference.yaml"),
    "gnina":        os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "gnina_inference.yaml"),
    "chai":         os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "chai_inference.yaml"),
    "dynamicbind":  os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "dynamicbind_inference.yaml"),
    "unidock2":     os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "unidock2_inference.yaml"),
    "surfdock":     os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "surfdock_inference.yaml"),
    "alphafold3":   os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "alphafold3_inference.yaml"),
    "boltz1":       os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "boltz1_inference.yaml"),
    "boltz2":       os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "boltz2_inference.yaml"),
    "protenix":     os.path.join(_PROJECT_ROOT, "dockstrat_config", "model", "protenix_inference.yaml"),
}

_METHOD_MODULES = {
    "vina":        "dockstrat.models.vina_inference",
    "gnina":       "dockstrat.models.gnina_inference",
    "chai":        "dockstrat.models.chai_inference",
    "dynamicbind": "dockstrat.models.dynamicbind_inference",
    "unidock2":    "dockstrat.models.unidock2_inference",
    "surfdock":    "dockstrat.models.surfdock_inference",
    "alphafold3":  "dockstrat.models.alphafold3_inference",
    "boltz1":      "dockstrat.models.boltz_inference",
    "boltz2":      "dockstrat.models.boltz_inference",
    "protenix":    "dockstrat.models.protenix_inference",
}


def _load_config(method: str, overrides: dict) -> dict:
    """Load method YAML config and apply overrides, then resolve ${oc.env:...} variables."""
    config_path = _CONFIG_PATHS[method]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found for method '{method}': {config_path}")

    os.environ.setdefault("PROJECT_ROOT", _PROJECT_ROOT)
    cfg = OmegaConf.load(config_path)

    # Apply overrides before resolution so ${...} interpolations see the new values
    non_none = {k: v for k, v in overrides.items() if v is not None}
    if non_none:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(non_none))

    return OmegaConf.to_container(cfg, resolve=True)


def _get_module(method: str):
    """Lazily import and return the method inference module."""
    return importlib.import_module(_METHOD_MODULES[method])


def dock_engine(
    method: str,
    dataset: Optional[str] = None,
    protein: Optional[str] = None,
    ligand: Optional[str] = None,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """Run docking with the specified method.

    Args:
        method:     Docking method. One of: vina, gnina, chai, dynamicbind, unidock2, surfdock, alphafold3.
        dataset:    Dataset name for batch mode (e.g. 'runsNposes'). Exclusive with protein/ligand.
        protein:    Path to receptor PDB for single-molecule mode.
        ligand:     Path to ligand SDF for single-molecule mode.
        output_dir: Base output directory for single-molecule mode.
                    Results land in output_dir/{method}/{prefix}/.
        prefix:     Name used for the output subdirectory and file prefix in single mode.
                    Defaults to the protein filename stem.
        **kwargs:   Config field overrides (e.g. repeat_index=1, top_n=5, cuda_device_index=1).

    Returns:
        Path to the output directory (single mode) or the dataset output_dir (batch mode).
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}")

    single_mode = protein is not None and ligand is not None
    dataset_mode = dataset is not None

    if not single_mode and not dataset_mode:
        raise ValueError(
            "Provide either 'dataset' for batch mode, "
            "or both 'protein' and 'ligand' for single-molecule mode."
        )
    if single_mode and dataset_mode:
        raise ValueError("'dataset' and 'protein'/'ligand' are mutually exclusive.")

    if dataset_mode:
        # Override both 'benchmark' (vina/gnina) and 'dataset' (chai/dynamicbind/unidock2)
        overrides = {"benchmark": dataset, "dataset": dataset, **kwargs}
        config = _load_config(method, overrides)
        mod = _get_module(method)
        mod.run_dataset(config)
        return config.get("output_dir")

    else:
        protein = os.path.abspath(protein)
        ligand = os.path.abspath(ligand)

        if output_dir is None:
            output_dir = os.path.join(_PROJECT_ROOT, "outputs", "single_molecule")

        _prefix = prefix or os.path.splitext(os.path.basename(protein))[0]
        system_out = os.path.join(os.path.abspath(output_dir), method, _prefix)
        os.makedirs(system_out, exist_ok=True)

        config = _load_config(method, kwargs)
        mod = _get_module(method)
        return mod.run_single(
            protein=protein,
            ligand=ligand,
            output_dir=system_out,
            config=config,
            prefix=_prefix,
        )
