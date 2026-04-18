# CogLigandBench Documentation Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all PoseBench-specific content from tracked Markdown files and replace with CogLigandBench-specific documentation.

**Architecture:** Six files are rewritten or created (README.md, docs/api.md, docs/running_methods.md, docs/rmsd_analysis.md, notebooks/04_utils/README_icm_rmsd.md) and one file is deleted (notebooks/ORGANIZATION_PLAN.md). No code changes; documentation only.

**Tech Stack:** Markdown, git

---

## File map

| Action | File |
|--------|------|
| Rewrite | `README.md` |
| Create | `docs/api.md` |
| Rewrite | `docs/running_methods.md` |
| Rewrite | `docs/rmsd_analysis.md` |
| Delete | `notebooks/ORGANIZATION_PLAN.md` |
| Fix path | `notebooks/04_utils/README_icm_rmsd.md` |

`docs/quickstart.md` is already clean — no changes needed.

---

## Task 1: Rewrite README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the new README.md**

Replace the entire file with:

```markdown
# CogLigandBench

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A benchmarking framework for protein-ligand docking methods evaluated against experimental crystal structures.

---

## Why Crystal Structures

Most docking benchmarks pair a ligand with an AlphaFold-predicted or homology-modelled receptor. CogLigandBench uses **co-crystal structures from the PDB** instead: the receptor conformation is the one that was experimentally observed in the presence of the ligand. This isolates docking error from structure-prediction error and mirrors the real structure-based drug design (SBDD) workflow, where a solved crystal structure of the target is the starting point.

**Benchmark datasets:**

| Dataset | Systems | Source |
|---------|---------|--------|
| runsNposes | ~1,280 | [runs-n-poses](https://github.com/plinder-org/runs-n-poses) |
| Astex Diverse | 85 | PDB (curated fragment set) |
| PoseBusters | 428 | [posebusters](https://github.com/maabuu/posebusters) |
| DockGen | 189 | [DockGen](https://github.com/HannesStark/DockGen) |

---

## Supported Methods & Preliminary Results

Results below are **top-1 pose, runsNposes benchmark** (crystal structures, no structure prediction).

| Method | Type | % ≤ 2 Å RMSD | Median RMSD |
|--------|------|-------------|-------------|
| SurfDock | Deep learning | 59.5% | 1.56 Å |
| GNINA | CNN scoring | 33.9% | 2.56 Å |
| ICM-RTCNN | Physics + ML | 27.5% | 3.42 Å |
| ICM | Physics-based | 25.7% | 3.69 Å |
| Vina | Physics-based | 7.5% | 7.43 Å |
| UniDock2 | Physics-based | 2.0%† | 24.35 Å† |

†UniDock2 result is tentative — run artifact under investigation.

**Conda environments by method:**

| Method | Conda env | Notes |
|--------|-----------|-------|
| `vina` | system | Needs `obabel`, `vina` on `$PATH` |
| `gnina` | system | Binary at `forks/GNINA/gnina` |
| `chai` | `forks/chai-lab/chai-lab/` | Pass `python_exec_path=` to `dock_engine` |
| `dynamicbind` | `forks/DynamicBind/DynamicBind/` | Path set in YAML config |
| `unidock2` | `unidock2` | `conda create -n unidock2 -c conda-forge unidock` |
| `surfdock` | `SurfDock` | Requires MSMS/APBS/pdb2pqr tools |
| `icm` | system | Requires commercial ICM license |

---

## Installation & Quickstart

### 1. Install the package

```bash
cd /path/to/CogLigandBench
conda create -n cogligandbench python=3.10 -y
conda activate cogligandbench
pip install -e .
```

### 2. Configure environment variables

```bash
cp .env.example .env
# edit .env with paths for the methods you want to use
set -a && source .env && set +a
```

### 3. Run docking

```python
from cogligandbench import dock_engine

# Single molecule
result = dock_engine(
    'vina',
    protein='receptor.pdb',
    ligand='ligand.sdf',
    output_dir='./results',
)

# Full runsNposes benchmark
dock_engine('gnina', dataset='runsNposes')
dock_engine('gnina', dataset='runsNposes', repeat_index=1)
```

Poses are written as ranked SDF files under `results/`. See [docs/quickstart.md](docs/quickstart.md) for per-method setup and [docs/api.md](docs/api.md) for the full `dock_engine` API.

### 4. Compute RMSD

```python
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS
from pathlib import Path

ref  = Chem.MolFromMolFile('crystal_ligand.sdf', removeHs=True)
pose = Chem.MolFromMolFile(str(next(Path(result).glob('*rank1*.sdf'))), removeHs=True)
rmsd = CalcRMS(ref, pose)
print(f'RMSD: {rmsd:.2f} Å')
```

---

## Project Structure

```
cogligandbench/         # Python package — dock_engine, per-method wrappers
  engine.py             # dock_engine unified entry point
  models/               # per-method inference modules (run_dataset, run_single)
  data/                 # input preparation and output extraction scripts
  analysis/             # RMSD, complex alignment, scoring
  utils/                # logging and data utilities

cogligand_config/       # Hydra/OmegaConf YAML configs
  model/                # per-method inference configs

data/                   # benchmark datasets (crystal structures, not in git)
  runsNposes/
  astex_diverse_set/
  posebusters_benchmark_set/
  dockgen_set/

forks/                  # third-party codebases (submodules / local installs)
  GNINA/                # GNINA binary
  Vina/                 # AutoDock Vina + ADFR suite
  DynamicBind/          # DynamicBind diffusion docking
  chai-lab/             # Chai-1
  UniDock2/             # UniDock2
  SurfDock/             # SurfDock (code + weights; tools installed separately)
  ICM/                  # ICM docking scripts

docs/                   # documentation
tests/                  # pytest suite (fast smoke tests + slow end-to-end)
```

---

## Citation & Acknowledgements

If you use CogLigandBench, please cite:

```bibtex
@software{cogligandbench2024,
  author  = {Xu, Aaron and others},
  title   = {CogLigandBench: Crystal-Structure Protein-Ligand Docking Benchmark},
  year    = {2024},
  url     = {https://github.com/AaronXu9/CogliandBench},
}
```

CogLigandBench was built on top of [PoseBench](https://github.com/BioinfoMachineLearning/PoseBench) (Morehead et al., 2024), which targets predicted/AlphaFold structures.
```

- [ ] **Step 2: Verify no PoseBench in title/main body**

```bash
grep -n "PoseBench" README.md
```

Expected: one line only — the acknowledgement line near the bottom (`built on top of [PoseBench]`).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for CogLigandBench (remove PoseBench content)"
```

---

## Task 2: Write docs/api.md

**Files:**
- Create: `docs/api.md`

- [ ] **Step 1: Write docs/api.md**

```markdown
# API Reference — CogLigandBench

## `dock_engine`

```python
from cogligandbench import dock_engine

result = dock_engine(
    method,
    *,
    # dataset mode
    dataset=None,
    repeat_index=0,
    # single-molecule mode
    protein=None,
    ligand=None,
    output_dir=None,
    prefix=None,
    # per-method kwargs (merged into YAML config)
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | required | One of `'vina'`, `'gnina'`, `'chai'`, `'dynamicbind'`, `'unidock2'`, `'surfdock'` |
| `dataset` | `str \| None` | `None` | Dataset name for batch mode (e.g. `'runsNposes'`). Mutually exclusive with `protein`/`ligand`. |
| `repeat_index` | `int` | `0` | Run index; used in output directory naming for batch mode. |
| `protein` | `str \| Path \| None` | `None` | Path to receptor PDB for single-molecule mode. |
| `ligand` | `str \| Path \| None` | `None` | Path to ligand SDF for single-molecule mode. |
| `output_dir` | `str \| Path \| None` | `None` | Output root for single-molecule mode. Results land in `output_dir/{method}/{prefix}/`. |
| `prefix` | `str \| None` | stem of protein filename | Identifier used for output filenames in single-molecule mode. |
| `**kwargs` | any | — | Method-specific overrides merged into the YAML config (see per-method tables below). |

### Return value

- **Single-molecule mode:** `str` — path to `output_dir/{method}/{prefix}/` containing ranked pose files.
- **Dataset mode:** `None` — results written directly to the path in the YAML config.

---

## Single-molecule vs dataset mode

```python
# Single-molecule mode — provide protein, ligand, output_dir
result_dir = dock_engine(
    'vina',
    protein='receptor.pdb',
    ligand='ligand.sdf',
    output_dir='./results',
    prefix='my_system',
)
# → poses written to ./results/vina/my_system/

# Dataset mode — provide dataset name
dock_engine('vina', dataset='runsNposes')
dock_engine('vina', dataset='runsNposes', repeat_index=1)
# → poses written to path defined in cogligand_config/model/vina_inference.yaml
```

---

## Per-method kwargs

All methods accept `**kwargs` that override keys in their YAML config.

### Vina

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `top_n` | `int` | `10` | Number of poses to write |
| `exhaustiveness` | `int` | `32` | Vina search exhaustiveness |
| `num_modes` | `int` | `20` | Max number of binding modes |
| `box_size` | `list[float]` | auto | `[x, y, z]` in Å; auto-computed from ligand if omitted |

### GNINA

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `top_n` | `int` | `10` | Number of poses to write |
| `exhaustiveness` | `int` | `32` | Search exhaustiveness |
| `num_modes` | `int` | `20` | Max binding modes |
| `cnn_scoring` | `str` | `'rescore'` | CNNscore mode: `'rescore'`, `'refinement'`, `'metrorescore'` |

### Chai-1

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `python_exec_path` | `str` | `'python3'` | **Required in practice.** Path to the chai conda env Python. |
| `num_trunk_recycles` | `int` | `3` | Number of trunk recycling iterations |
| `num_diffn_timesteps` | `int` | `200` | Diffusion timesteps |
| `seed` | `int` | `42` | Random seed |

### DynamicBind

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `samples_per_complex` | `int` | `40` | Total samples to generate |
| `savings_per_complex` | `int` | `40` | Top-N samples to save |
| `python_exec_path` | `str` | from YAML | DynamicBind conda env Python (set in config) |

### UniDock2

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `num_poses` | `int` | `10` | Number of output poses |
| `box_size` | `list[float]` | `[15, 15, 15]` | Docking box in Å |
| `search_mode` | `str` | `'detail'` | `'fast'`, `'balance'`, or `'detail'` |
| `energy_range` | `float` | `15.0` | Energy window in kcal/mol |

### SurfDock

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `num_poses` | `int` | `40` | Number of poses to keep |
| `samples_per_complex` | `int` | `40` | Diffusion samples to generate |
| `batch_size` | `int` | `5` | Inference batch size |
| `surfdock_dir` | `str` | `$SURFDOCK_DIR` | Path to local SurfDock installation |

---

## Output file naming

| Method | Pattern | Location |
|--------|---------|----------|
| `vina` | `{prefix}_pose{N}_score{S:.2f}.sdf` | `output_dir/vina/{prefix}/` |
| `gnina` | `{prefix}_pose{N}_score{S:.2f}.sdf` | `output_dir/gnina/{prefix}/` |
| `chai` | `pred.model_idx_{0-4}.pdb` | `output_dir/chai/{prefix}/` |
| `dynamicbind` | `rank{N}_ligand_lddt{L}_affinity{A}.sdf` | `output_dir/dynamicbind/{prefix}/` |
| `unidock2` | `rank{N}.sdf` | `output_dir/unidock2/{prefix}/` |
| `surfdock` | `rank{N}.sdf` | `output_dir/surfdock/{prefix}/` |

**Ranking:** Lower rank = better pose, except GNINA where `score` is CNNscore (higher = better).

---

## Config override pattern

Each `**kwarg` passed to `dock_engine` is merged into the method's YAML config before inference. Config files live in `cogligand_config/model/<method>_inference.yaml`.

```python
# This call:
dock_engine('vina', protein='r.pdb', ligand='l.sdf', output_dir='./out', top_n=5)

# Is equivalent to loading cogligand_config/model/vina_inference.yaml
# then overriding: cfg.top_n = 5
```

To inspect the full list of config keys for a method:

```python
from omegaconf import OmegaConf
cfg = OmegaConf.load('cogligand_config/model/vina_inference.yaml')
print(OmegaConf.to_yaml(cfg))
```

---

## RMSD calculation

```python
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS
from pathlib import Path

def top1_rmsd(crystal_sdf: str, result_dir: str, method: str = 'vina') -> float:
    """Compute symmetry-corrected RMSD for the top-ranked pose."""
    ref = Chem.MolFromMolFile(crystal_sdf, removeHs=True)

    # Find rank-1 pose file
    if method == 'chai':
        pattern = 'pred.model_idx_0.pdb'
        mol = Chem.MolFromPDBFile(str(next(Path(result_dir).glob(pattern))), removeHs=True)
    else:
        pattern = '*rank1*.sdf' if method in ('unidock2', 'surfdock', 'dynamicbind') else '*pose1*.sdf'
        mol = Chem.MolFromMolFile(str(next(Path(result_dir).glob(pattern))), removeHs=True)

    return CalcRMS(ref, mol)  # symmetry-corrected
```

**Note for Chai-1:** The `pred.model_idx_*.pdb` files contain the full protein-ligand complex. Extract the ligand residue before computing RMSD, or use `cogligandbench/data/chai_output_extraction.py` for batch extraction.
```

- [ ] **Step 2: Commit**

```bash
git add docs/api.md
git commit -m "docs: add docs/api.md with full dock_engine API reference"
```

---

## Task 3: Rewrite docs/running_methods.md

**Files:**
- Modify: `docs/running_methods.md`

Changes needed:
1. Line 12: `export PROJECT_ROOT=/path/to/PoseBench` → `/path/to/CogLigandBench`
2. Line 14: Remove `rfaa, diffdock, fabind, neuralplexer` from the comment
3. Lines 128-133 (DynamicBind): Fix config ref from `configs/model/dynamicbind_inference.yaml (uses upstream posebench config)` → `cogligand_config/model/dynamicbind_inference.yaml`
4. Lines 138-195: Delete DiffDock, FABind, NeuralPLexer, and RFAA sections entirely
5. Update the Notes section to remove references to deleted methods

- [ ] **Step 1: Fix the PROJECT_ROOT path and comment on line 12/14**

In `docs/running_methods.md`, replace:
```
export PROJECT_ROOT=/path/to/PoseBench

# Hydra-based methods (chai, dynamicbind, rfaa, diffdock, fabind, neuralplexer):
```
with:
```
export PROJECT_ROOT=/path/to/CogLigandBench

# Hydra-based methods (chai, dynamicbind):
```

- [ ] **Step 2: Fix DynamicBind config reference**

Replace:
```
**Config:** `configs/model/dynamicbind_inference.yaml` (uses upstream posebench config)
```
with:
```
**Config:** `cogligand_config/model/dynamicbind_inference.yaml`
```

- [ ] **Step 3: Delete DiffDock, FABind, NeuralPLexer, RFAA sections**

Delete the entire `## DiffDock`, `## FABind`, `## NeuralPLexer`, and `## RoseTTAFold-All-Atom (RFAA)` sections and their content.

- [ ] **Step 4: Update the Notes section**

Replace the final Notes block with:
```markdown
## Notes

- **Input CSV format** (Vina/GNINA): columns `complex_name`, `protein_path`, `ligand_path`
- **Input directory format** (Chai-1): `<input_dir>/<complex_name>/<complex_name>.fasta`
- **Input directory format** (ICM): `<data_dir>/<complex_name>/<complex_name>.pdb` + `*.sdf`
- **Output poses** are written as ranked SDF files: `<complex_name>_pose<rank>_score<score>.sdf`
- **Timing logs** record `complex_name,elapsed_seconds` per entry
```

- [ ] **Step 5: Verify no remaining PoseBench config path references**

```bash
grep -n "configs/model/" docs/running_methods.md
grep -n "PoseBench" docs/running_methods.md
```

Expected: zero matches for both.

- [ ] **Step 6: Commit**

```bash
git add docs/running_methods.md
git commit -m "docs: remove deleted methods from running_methods.md, fix PROJECT_ROOT path"
```

---

## Task 4: Rewrite docs/rmsd_analysis.md

**Files:**
- Modify: `docs/rmsd_analysis.md`

The current file only covers Vina/UniDock2, references `/home/aoxu/projects/PoseBench`, and documents a standalone `scripts/rmsd_analysis.py` workflow that is no longer the recommended approach. Replace entirely with a method-agnostic reference that covers all 6 methods.

- [ ] **Step 1: Write the new docs/rmsd_analysis.md**

Replace the entire file with:

```markdown
# RMSD Analysis — CogLigandBench

This document describes how to compute RMSD between docked poses and crystal ligands for all supported methods. All analysis uses RDKit's symmetry-corrected RMSD (`CalcRMS`), which accounts for equivalent atom orderings.

---

## Output locations and file patterns

After running `dock_engine`, poses land in `output_dir/{method}/{prefix}/`. The table below lists the rank-1 file pattern for each method.

| Method | Rank-1 file | Notes |
|--------|------------|-------|
| `vina` | `*_pose1_score*.sdf` | Scored by Vina energy (most negative = best) |
| `gnina` | `*_pose1_score*.sdf` | Scored by CNNscore (highest = best) |
| `chai` | `pred.model_idx_0.pdb` | Full complex PDB; ligand must be extracted |
| `dynamicbind` | `rank1_ligand_lddt*.sdf` | Scored by DynamicBind lDDT |
| `unidock2` | `rank1.sdf` | Scored by Vina energy (most negative = best) |
| `surfdock` | `rank1.sdf` | Scored by confidence (highest = best) |

---

## Single-system RMSD

```python
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS
from pathlib import Path

def compute_rmsd(crystal_sdf: str, pose_sdf: str) -> float:
    ref  = Chem.MolFromMolFile(crystal_sdf, removeHs=True)
    pose = Chem.MolFromMolFile(pose_sdf, removeHs=True)
    if ref is None or pose is None:
        raise ValueError(f"Could not load molecule from {crystal_sdf} or {pose_sdf}")
    return CalcRMS(ref, pose)  # symmetry-corrected

# Example
rmsd = compute_rmsd(
    'data/runsNposes/8gkf__1__1.A__1.J/8gkf__1__1.A__1.J_ligand.sdf',
    'results/vina/8gkf_test/8gkf_test_pose1_score-7.23.sdf',
)
print(f'RMSD: {rmsd:.2f} Å')
```

---

## Batch RMSD over runsNposes

```python
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS

DATA_DIR  = Path('data/runsNposes')        # crystal structures
RESULTS   = Path('results/vina')           # dock_engine output root
METHOD    = 'vina'
RANK1_GLOB = '*_pose1_score*.sdf'          # adjust per method (see table above)

records = []
for system_dir in sorted(DATA_DIR.iterdir()):
    system = system_dir.name
    crystal = system_dir / f'{system}_ligand.sdf'
    pose_dir = RESULTS / system
    poses = sorted(pose_dir.glob(RANK1_GLOB)) if pose_dir.exists() else []

    if not crystal.exists() or not poses:
        records.append({'system': system, 'rmsd': None})
        continue

    ref  = Chem.MolFromMolFile(str(crystal), removeHs=True)
    pose = Chem.MolFromMolFile(str(poses[0]), removeHs=True)
    try:
        rmsd = CalcRMS(ref, pose)
    except Exception as e:
        rmsd = None
    records.append({'system': system, 'rmsd': rmsd})

df = pd.DataFrame(records)
valid = df['rmsd'].dropna()
print(f"Systems:         {len(df)}")
print(f"With results:    {valid.count()}")
print(f"Median RMSD:     {valid.median():.2f} Å")
print(f"% ≤ 2 Å (top-1): {(valid <= 2.0).mean() * 100:.1f}%")
print(f"% ≤ 5 Å (top-1): {(valid <= 5.0).mean() * 100:.1f}%")
```

---

## Chai-1: extract ligand before RMSD

Chai-1 outputs full protein-ligand complex PDBs. Extract the ligand first:

```python
from cogligandbench.data.chai_output_extraction import extract_ligand_from_complex

# Extract ligand from complex PDB
ligand_sdf = extract_ligand_from_complex(
    complex_pdb='results/chai/8gkf_test/pred.model_idx_0.pdb',
    output_sdf='results/chai/8gkf_test/ligand_model_0.sdf',
)

# Then compute RMSD normally
rmsd = compute_rmsd('data/runsNposes/8gkf__1__1.A__1.J/8gkf__1__1.A__1.J_ligand.sdf', ligand_sdf)
```

For batch extraction over a full dataset run, use `cogligandbench/data/chai_output_extraction.py`.

---

## Preliminary benchmark results (runsNposes, top-1)

| Method | Systems | % ≤ 2 Å | % ≤ 5 Å | Median RMSD |
|--------|---------|---------|---------|-------------|
| SurfDock | ~1,280 | 59.5% | 79.2% | 1.56 Å |
| GNINA | ~1,280 | 33.9% | 58.1% | 2.56 Å |
| ICM-RTCNN | ~1,280 | 27.5% | 54.3% | 3.42 Å |
| ICM | ~1,280 | 25.7% | 52.8% | 3.69 Å |
| Vina | ~1,280 | 7.5% | 28.4% | 7.43 Å |
| UniDock2 | ~1,280 | 2.0%† | — | 24.35 Å† |

†UniDock2 tentative — run artifact under investigation.
```

- [ ] **Step 2: Verify no hardcoded paths remain**

```bash
grep -n "PoseBench\|/home/aoxu" docs/rmsd_analysis.md
```

Expected: zero matches.

- [ ] **Step 3: Commit**

```bash
git add docs/rmsd_analysis.md
git commit -m "docs: rewrite rmsd_analysis.md to cover all 6 methods, remove PoseBench paths"
```

---

## Task 5: Delete ORGANIZATION_PLAN.md and fix README_icm_rmsd.md

**Files:**
- Delete: `notebooks/ORGANIZATION_PLAN.md`
- Modify: `notebooks/04_utils/README_icm_rmsd.md` (lines 50-52)

- [ ] **Step 1: Delete ORGANIZATION_PLAN.md**

```bash
git rm notebooks/ORGANIZATION_PLAN.md
```

- [ ] **Step 2: Fix paths in README_icm_rmsd.md**

Replace (lines 50-52):
```
- `base_outdir`: Base output directory (default: "/home/aoxu/projects/PoseBench/forks")
- `data_dir`: Data directory with reference files (default: "/home/aoxu/projects/PoseBench/data/runsNposes")
- `BASE_DIRS`: Dictionary mapping method names to their output directories
```
with:
```
- `base_outdir`: Base output directory (default: `${PROJECT_ROOT}/forks`)
- `data_dir`: Data directory with reference files (default: `${PROJECT_ROOT}/data/runsNposes`)
- `BASE_DIRS`: Dictionary mapping method names to their output directories
```

Also replace line 29 in the same file:
```
- The `calculate_rmsd.icm` script (automatically discovered)
```
— no change needed there.

Also find and replace the two PoseBench references in the file:
- `"/home/aoxu/projects/PoseBench/forks"` → `"${PROJECT_ROOT}/forks"`
- `"/home/aoxu/projects/PoseBench/data/runsNposes"` → `"${PROJECT_ROOT}/data/runsNposes"`

- [ ] **Step 3: Verify clean**

```bash
grep -n "PoseBench\|/home/aoxu" notebooks/04_utils/README_icm_rmsd.md
```

Expected: zero matches.

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_utils/README_icm_rmsd.md
git commit -m "docs: remove ORGANIZATION_PLAN.md, fix hardcoded PoseBench paths in README_icm_rmsd.md"
```

---

## Task 6: Final verification and push

- [ ] **Step 1: Check no tracked .md files contain `/home/aoxu/` or PoseBench title content**

```bash
git grep -l "PoseBench" -- "*.md" | grep -v "docs/superpowers/"
git grep -n "/home/aoxu/projects/PoseBench" -- "*.md"
```

Expected: no matches in `docs/running_methods.md`, `docs/rmsd_analysis.md`, `README.md`, or `notebooks/04_utils/README_icm_rmsd.md`. The spec/plan files under `docs/superpowers/` may reference PoseBench in historical context — those are acceptable.

- [ ] **Step 2: Push to remote**

```bash
git push origin main
```

---

## Success criteria

- [ ] `README.md`: title is "# CogLigandBench"; no PoseBench in main body (one acknowledgement line is fine)
- [ ] `docs/api.md`: exists; covers `dock_engine` signature, all 6 methods' kwargs, output patterns
- [ ] `docs/running_methods.md`: no `configs/model/` refs; PROJECT_ROOT is `/path/to/CogLigandBench`; no DiffDock/FABind/NeuralPLexer/RFAA sections
- [ ] `docs/rmsd_analysis.md`: no `/home/aoxu/` paths; covers all 6 methods
- [ ] `notebooks/ORGANIZATION_PLAN.md`: deleted
- [ ] `notebooks/04_utils/README_icm_rmsd.md`: no `/home/aoxu/` paths
