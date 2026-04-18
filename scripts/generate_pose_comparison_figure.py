#!/usr/bin/env python3
"""Generate the pose-comparison figure for the README.

Picks one HIGH-similarity and one LOW-similarity runsNposes system, then
renders 4 PyMOL panels:
  - AF3 vs crystal (high sim) — both should converge
  - AF3 vs crystal (low sim)  — AF3 expected to drift
  - GNINA vs crystal (high sim) — both should converge
  - GNINA vs crystal (low sim)  — GNINA expected to remain near-native

Outputs PNGs into docs/figures/ plus a selection_metadata.json that records
the chosen system IDs, SuCOS, and rank-1 RMSDs (for the README caption).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_CSV = PROJECT_ROOT / "data" / "runsNposes_archive" / "zenodo_downloads" / "annotations.csv"
RUNS_NPOSES_DIR = PROJECT_ROOT / "data" / "runsNposes"
AF3_POSES_DIR = PROJECT_ROOT / "zenodo_staging" / "cofolding_poses" / "af3"
AF3_PREDICTIONS_CSV = PROJECT_ROOT / "zenodo_staging" / "cofolding_results_tables" / "predictions" / "af3.csv"
GNINA_TARBALL = PROJECT_ROOT / "zenodo_staging" / "docking_poses_gnina.tar.gz"
GNINA_EXTRACT_DIR = Path("/tmp/gnina_extracted")
WORK_DIR = Path("/tmp/pose_compare")
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
PYMOL_BIN = "/home/aoxu/miniconda3/envs/PyMOL-PoseBench/bin/pymol"
PML_TEMPLATE = PROJECT_ROOT / "scripts" / "_pose_comparison.pml"

SIMILARITY_COL = "sucos_shape_pocket_qcov"

# ---------------------------------------------------------------------------
# AF3 ligand extraction (HETATM-based; the cofolding_poses/af3 CIFs label
# ligands as HETATM records on a non-"L" chain, so we can't reuse the
# chain-L-specific helper from dockstrat.models.alphafold3_inference).
# ---------------------------------------------------------------------------

def _extract_ligand_from_cif_hetatm(cif_path: Path, template_mol):
    """Pull HETATM records out of a CIF and recover bond orders from a SMILES template."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from rdkit.Chem import AllChem

    d = MMCIF2Dict(str(cif_path))
    groups = d.get("_atom_site.group_PDB", [])
    chains = d.get("_atom_site.auth_asym_id", d.get("_atom_site.label_asym_id", []))
    resnames = d.get("_atom_site.label_comp_id", [])
    atom_names = d.get("_atom_site.label_atom_id", [])
    elements = d.get("_atom_site.type_symbol", [])
    xs = d.get("_atom_site.Cartn_x", [])
    ys = d.get("_atom_site.Cartn_y", [])
    zs = d.get("_atom_site.Cartn_z", [])

    pdb_lines = []
    atom_idx = 1
    for i, g in enumerate(groups):
        if g != "HETATM":
            continue
        name = (atom_names[i] if i < len(atom_names) else "X")[:4].ljust(4)
        elem = (elements[i] if i < len(elements) else name.strip()[0]).strip()
        x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
        pdb_lines.append(
            f"HETATM{atom_idx:>5} {name} LIG L{1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem.rjust(2)}"
        )
        atom_idx += 1

    if not pdb_lines:
        raise ValueError(f"No HETATM ligand atoms found in {cif_path}")

    pdb_block = "\n".join(pdb_lines) + "\nEND\n"
    raw = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if raw is None:
        raise ValueError(f"RDKit could not parse synthesized PDB block from {cif_path}")
    try:
        Chem.SanitizeMol(raw)
    except Exception:
        pass
    return AllChem.AssignBondOrdersFromTemplate(template_mol, raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[pose-fig] {msg}", flush=True)


def ensure_gnina_extracted() -> Path:
    """Extract the GNINA tarball once; return the gnina/ root."""
    gnina_root = GNINA_EXTRACT_DIR / "gnina"
    if gnina_root.is_dir():
        log(f"GNINA already extracted at {gnina_root}")
        return gnina_root
    log(f"Extracting {GNINA_TARBALL.name} to {GNINA_EXTRACT_DIR} (one-time)")
    GNINA_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(GNINA_TARBALL, "r:gz") as tf:
        tf.extractall(GNINA_EXTRACT_DIR)
    return gnina_root


def find_gnina_rank1(gnina_sys_dir: Path) -> Path | None:
    """Locate gnina/{sys}/rank1_score*.sdf."""
    matches = list(gnina_sys_dir.glob("rank1_score*.sdf"))
    return matches[0] if matches else None


def af3_top_cif(af3_sys_dir: Path) -> tuple[Path, float] | None:
    """Return (top_cif_path, ranking_score) from the system's ranking_scores.csv."""
    scores_csv = af3_sys_dir / "ranking_scores.csv"
    if not scores_csv.exists():
        return None
    df = pd.read_csv(scores_csv).sort_values("ranking_score", ascending=False)
    for _, row in df.iterrows():
        seed, sample = int(row["seed"]), int(row["sample"])
        cif = af3_sys_dir / f"seed-{seed}_sample-{sample}.cif"
        if cif.exists():
            return cif, float(row["ranking_score"])
    return None


def find_af3_ligand_chain(cif_path: Path, target_atom_count: int) -> str | None:
    """Find the AF3 HETATM chain whose atom count matches the crystal ligand.

    AF3 may emit multiple HETATM chains (multiple ligand instances or
    cofactors); we want the one corresponding to the system's primary
    ligand. Match by heavy-atom count.
    """
    from collections import Counter
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    d = MMCIF2Dict(str(cif_path))
    groups = d.get("_atom_site.group_PDB", [])
    chains = d.get(
        "_atom_site.auth_asym_id",
        d.get("_atom_site.label_asym_id", []),
    )
    counts = Counter(c for g, c in zip(groups, chains) if g == "HETATM")
    if not counts:
        return None
    # Exact match on atom count
    for chain, n in counts.items():
        if n == target_atom_count:
            return chain
    # Fallback: closest by absolute difference
    chain, _ = min(counts.items(), key=lambda kv: abs(kv[1] - target_atom_count))
    return chain


def crystal_paths(sys_id: str) -> tuple[Path, Path] | None:
    """Return (protein_pdb, crystal_ligand_sdf) for a system or None."""
    pdb = RUNS_NPOSES_DIR / sys_id / f"{sys_id}_protein.pdb"
    sdf = RUNS_NPOSES_DIR / sys_id / f"{sys_id}_ligand.sdf"
    return (pdb, sdf) if (pdb.exists() and sdf.exists()) else None


def safe_rmsd(crystal_sdf: Path, pred_sdf: Path) -> float | None:
    """Symmetry-corrected RMSD, returning None on RDKit failure."""
    try:
        ref = Chem.MolFromMolFile(str(crystal_sdf), removeHs=True)
        pose = Chem.MolFromMolFile(str(pred_sdf), removeHs=True)
        if ref is None or pose is None:
            return None
        return float(CalcRMS(ref, pose))
    except Exception:
        return None


def extract_af3_ligand(af3_cif: Path, crystal_sdf: Path, out_sdf: Path) -> Path | None:
    """Extract the ligand from an AF3 CIF using the crystal SMILES as template."""
    try:
        crystal = Chem.MolFromMolFile(str(crystal_sdf), removeHs=True)
        if crystal is None:
            return None
        template_smiles = Chem.MolToSmiles(crystal)
        template = Chem.MolFromSmiles(template_smiles)
        mol = _extract_ligand_from_cif_hetatm(af3_cif, template_mol=template)
        if mol is None:
            return None
        Chem.MolToMolFile(mol, str(out_sdf))
        return out_sdf
    except Exception as exc:
        log(f"  AF3 extraction failed for {af3_cif.parent.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_systems():
    """Find one HIGH-sim and one LOW-sim system meeting the contrast criteria."""
    log(f"Loading annotations from {ANNOTATIONS_CSV.name}")
    ann = pd.read_csv(ANNOTATIONS_CSV)

    # The annotations file has multiple rows per system (one per training neighbor).
    # Use the maximum sucos_shape_pocket_qcov per system as the reported similarity
    # (i.e., closest training match).
    sims = ann.groupby("system_id")[SIMILARITY_COL].max().reset_index()
    sims = sims.rename(columns={SIMILARITY_COL: "sucos"})
    log(f"Annotations: {len(sims)} systems, sucos range [{sims['sucos'].min():.2f}, {sims['sucos'].max():.2f}]")

    log("Pre-loading AF3 predictions table")
    af3_pred = pd.read_csv(AF3_PREDICTIONS_CSV)
    # Use rank-1 (best ranking_score) per system; take min RMSD across ligand chains
    af3_top = (
        af3_pred.sort_values("ranking_score", ascending=False)
        .drop_duplicates(["target", "seed", "sample"])
        .groupby("target")
        .first()  # top by ranking_score after sort
        .reset_index()
    )
    af3_top_rmsd = af3_top.set_index("target")["rmsd"].to_dict()

    log("Extracting GNINA tarball...")
    gnina_root = ensure_gnina_extracted()
    available_gnina = {p.name for p in gnina_root.iterdir() if p.is_dir()}
    log(f"  GNINA available for {len(available_gnina)} systems")

    available_af3 = {p.name for p in AF3_POSES_DIR.iterdir() if p.is_dir()}
    log(f"  AF3 available for {len(available_af3)} systems")

    # Filter to systems with all 4 artifacts
    sims["has_runsnposes"] = sims["system_id"].apply(lambda s: crystal_paths(s) is not None)
    sims["has_af3"] = sims["system_id"].isin(available_af3)
    sims["has_gnina"] = sims["system_id"].isin(available_gnina)
    sims["af3_rmsd"] = sims["system_id"].map(af3_top_rmsd)
    candidates = sims[sims["has_runsnposes"] & sims["has_af3"] & sims["has_gnina"]].copy()
    log(f"  Systems with all 4 artifacts: {len(candidates)}")

    # Compute GNINA rank-1 RMSD inline
    log("Computing GNINA rank-1 RMSDs (this may take a minute)...")
    gnina_rmsds = []
    for sys_id in candidates["system_id"]:
        sdf = find_gnina_rank1(gnina_root / sys_id)
        crystal = crystal_paths(sys_id)
        if sdf is None or crystal is None:
            gnina_rmsds.append(None)
            continue
        gnina_rmsds.append(safe_rmsd(crystal[1], sdf))
    candidates["gnina_rmsd"] = gnina_rmsds
    candidates = candidates.dropna(subset=["af3_rmsd", "gnina_rmsd"])
    log(f"  Candidates with both RMSDs computed: {len(candidates)}")

    # NOTE: sucos_shape_pocket_qcov is a percentage (0-100), not a fraction (0-1).
    # HIGH-sim: SuCOS in [70, 100], both methods succeed (RMSD < 2A).
    high_pool = candidates[
        (candidates["sucos"] >= 70.0)
        & (candidates["af3_rmsd"] < 2.0)
        & (candidates["gnina_rmsd"] < 2.0)
    ].copy()
    log(f"  HIGH-sim pool (sucos>=70, both<2A): {len(high_pool)}")
    if high_pool.empty:
        raise RuntimeError("No HIGH-sim candidate found")

    # Pick the median-sucos one for a representative example (not the most extreme)
    high_pool = high_pool.sort_values("sucos")
    high = high_pool.iloc[len(high_pool) // 2]
    log(f"  -> HIGH pick: {high['system_id']} (sucos={high['sucos']:.2f}, "
        f"af3_rmsd={high['af3_rmsd']:.2f}, gnina_rmsd={high['gnina_rmsd']:.2f})")

    # LOW-sim: SuCOS <= 20, AF3 fails (>4A), GNINA succeeds (<2A).
    # Relax stepwise if no perfect match.
    for sucos_max, af3_min, gnina_max in [
        (20.0, 4.0, 2.0),
        (20.0, 3.5, 2.5),
        (30.0, 3.0, 3.0),
    ]:
        low_pool = candidates[
            (candidates["sucos"] <= sucos_max)
            & (candidates["af3_rmsd"] > af3_min)
            & (candidates["gnina_rmsd"] < gnina_max)
        ].copy()
        log(f"  LOW-sim pool (sucos<={sucos_max}, af3>{af3_min}, gnina<{gnina_max}): {len(low_pool)}")
        if not low_pool.empty:
            break
    else:
        raise RuntimeError("No LOW-sim contrast candidate found")

    # Pick the one with biggest contrast: AF3 RMSD - GNINA RMSD
    low_pool["contrast"] = low_pool["af3_rmsd"] - low_pool["gnina_rmsd"]
    low = low_pool.sort_values("contrast", ascending=False).iloc[0]
    log(f"  -> LOW pick: {low['system_id']} (sucos={low['sucos']:.2f}, "
        f"af3_rmsd={low['af3_rmsd']:.2f}, gnina_rmsd={low['gnina_rmsd']:.2f})")

    return {
        "high": {
            "system_id": high["system_id"],
            "sucos": float(high["sucos"]),
            "af3_rmsd": float(high["af3_rmsd"]),
            "gnina_rmsd": float(high["gnina_rmsd"]),
        },
        "low": {
            "system_id": low["system_id"],
            "sucos": float(low["sucos"]),
            "af3_rmsd": float(low["af3_rmsd"]),
            "gnina_rmsd": float(low["gnina_rmsd"]),
        },
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_panel(receptor: Path, crystal: Path, predicted: Path, mode: str, output: Path) -> None:
    cmd = [
        PYMOL_BIN, "-cq", str(PML_TEMPLATE), "--",
        str(receptor), str(crystal), str(predicted), mode, str(output),
    ]
    log(f"  Rendering {output.name} (mode={mode})")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        log(f"  PyMOL stderr:\n{proc.stderr}")
        raise RuntimeError(f"PyMOL render failed for {output}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    gnina_root = ensure_gnina_extracted()

    selection = select_systems()

    panels = []
    for regime in ("high", "low"):
        sys_id = selection[regime]["system_id"]
        protein_pdb, crystal_sdf = crystal_paths(sys_id)

        # AF3: pass the top-ranked CIF directly (PyMOL will align + filter HETATMs
        # to the chain whose atom count matches the crystal ligand).
        af3_dir = AF3_POSES_DIR / sys_id
        top = af3_top_cif(af3_dir)
        if top is None:
            raise RuntimeError(f"No AF3 CIF for {sys_id}")
        af3_cif = top[0]

        # Find the matching ligand chain by atom count
        crystal_mol = Chem.MolFromMolFile(str(crystal_sdf), removeHs=True)
        n_heavy = crystal_mol.GetNumHeavyAtoms()
        af3_chain = find_af3_ligand_chain(af3_cif, n_heavy)
        if af3_chain is None:
            raise RuntimeError(f"No matching AF3 HETATM chain for {sys_id}")
        log(f"  [{regime}] AF3 top CIF: {af3_cif.name} (rscore={top[1]:.3f}, "
            f"chain={af3_chain}, CSV rmsd={selection[regime]['af3_rmsd']:.2f}A)")

        # GNINA: copy rank-1 SDF (already in the crystal frame)
        gnina_src = find_gnina_rank1(gnina_root / sys_id)
        if gnina_src is None:
            raise RuntimeError(f"No GNINA rank1 SDF for {sys_id}")
        gnina_sdf = WORK_DIR / f"{sys_id}_gnina_rank1.sdf"
        shutil.copy(gnina_src, gnina_sdf)

        panels.extend([
            ("af3",   regime, protein_pdb, crystal_sdf, af3_cif,   f"cif:{af3_chain}"),
            ("gnina", regime, protein_pdb, crystal_sdf, gnina_sdf, "sdf"),
        ])

    log("Rendering 4 panels with PyMOL")
    for method, regime, receptor, crystal, predicted, mode in panels:
        out = FIGURES_DIR / f"{method}_{regime}_sim.png"
        render_panel(receptor, crystal, predicted, mode, out)

    metadata_path = FIGURES_DIR / "selection_metadata.json"
    metadata_path.write_text(json.dumps(selection, indent=2))
    log(f"Wrote {metadata_path.name}")
    log(f"All 4 PNGs in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
