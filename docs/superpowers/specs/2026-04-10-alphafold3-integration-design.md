# AlphaFold3 Integration Design

**Date:** 2026-04-10
**Goal:** Add AlphaFold3 as a first-class docking method in CogLigandBench, reachable via `dock_engine('alphafold3', ...)` in both single-molecule and dataset modes.

**Approach:** Run AF3 in **no-MSA / single-sequence mode** through a subprocess that invokes `run_alphafold.py` in a dedicated conda env living at `/mnt/katritch_lab2/aoxu/envs/alphafold3`. No MSA/template databases are required. Ligand SDFs are extracted inline from the predicted mmCIFs using the known input SMILES for bond-order recovery.

---

## 1. Scope

- **Target repo:** `/mnt/katritch_lab2/aoxu/CogLigandBench`. PoseBench is legacy and is not referenced or ported.
- **Modes supported:** full parity with existing methods — `run_single` and `run_dataset`.
- **MSA strategy:** single-sequence / no-MSA mode. AF3 runs with `--norun_data_pipeline` and the input JSON carries an A3M containing only the query sequence. No `~630 GB` database download, no `db_dir`.
- **Output:** `rank{N}.sdf` per predicted pose, ranked by AF3's own `ranking_scores.csv`. Matches the Vina/GNINA/UniDock2/SurfDock output shape so downstream RMSD scripts work without modification.
- **Default datasets:** runsNposes is the primary target; other datasets (posebusters_benchmark, astex_diverse, etc.) work by pointing the config's `input_dir` at a different folder with the same per-system layout.
- **Hardware:** sane single-GPU defaults with `cuda_device_index` config field, OOM / timeout → log + skip. No explicit multi-GPU coordination.

---

## 2. Directory Layout

New files and directories:

```
CogLigandBench/
  envs/                                  # NEW symlink directory
    alphafold3 -> /mnt/katritch_lab2/aoxu/envs/alphafold3
  forks/
    alphafold3/
      alphafold3/                        # clone of github.com/google-deepmind/alphafold3
      models/
        af3.bin                          # decompressed from af3.bin.zst (one-time)
      prediction_outputs/                # already exists
  cogligand_config/model/
    alphafold3_inference.yaml            # NEW
  cogligandbench/
    models/
      alphafold3_inference.py            # NEW
    utils/
      sequence.py                        # NEW — shared PDB→sequence helper (extracted from chai_inference.py)
  scripts/
    install_alphafold3_env.sh            # NEW one-shot install + weight-decompression helper
  docs/superpowers/specs/
    2026-04-10-alphafold3-integration-design.md    # this file
  tests/
    test_alphafold3_inference.py         # NEW unit tests for helpers
```

**Design rationale:**

- **`alphafold3/alphafold3/` nested path** mirrors `forks/chai-lab/chai-lab/`. The outer directory is the CogLigandBench fork entry; the inner directory is the upstream checkout.
- **`forks/alphafold3/models/af3.bin` outside the upstream clone** keeps weights visible to CI/docs but isolated from upstream rebases.
- **`envs/alphafold3` is a symlink**, not a real directory. The actual env lives on `/mnt/katritch_lab2/aoxu/envs/alphafold3` so it does not consume home-directory space and can be shared with sibling projects.
- **`scripts/install_alphafold3_env.sh`** makes the one-time install reproducible and documents the recipe in an executable form.

---

## 3. Install Flow (`scripts/install_alphafold3_env.sh`)

Idempotent shell script capturing the one-time setup. Steps, in order:

1. `mkdir -p /mnt/katritch_lab2/aoxu/envs`
2. `conda create -p /mnt/katritch_lab2/aoxu/envs/alphafold3 python=3.11 hmmer -c bioconda -c conda-forge -y`
   - `hmmer` is installed even though we skip the data pipeline because AF3's imports resolve the `jackhmmer`/`nhmmer` binaries at startup.
3. `mkdir -p CogLigandBench/envs && ln -sfn /mnt/katritch_lab2/aoxu/envs/alphafold3 CogLigandBench/envs/alphafold3`
4. `git clone https://github.com/google-deepmind/alphafold3 forks/alphafold3/alphafold3`
5. `conda run -p /mnt/katritch_lab2/aoxu/envs/alphafold3 pip install -r forks/alphafold3/alphafold3/dev-requirements.txt`
6. `conda run -p /mnt/katritch_lab2/aoxu/envs/alphafold3 pip install -e forks/alphafold3/alphafold3`
7. Decompress weights: `zstd -d af3.bin.zst -o forks/alphafold3/models/af3.bin` (or the Python equivalent via `zstandard`).
8. Sanity check: `conda run -p /mnt/katritch_lab2/aoxu/envs/alphafold3 python -c "import alphafold3"`.

Key install decisions:

- **`conda create -p <prefix>`** (not `-n`) because the env is at a non-default location.
- **Direct `{env}/bin/python` invocation** from the wrapper instead of `conda run`: faster startup and better stderr preservation for debugging.
- **Install script is committed.** No hidden state in somebody's shell history.

---

## 4. Config (`cogligand_config/model/alphafold3_inference.yaml`)

```yaml
# Dataset mode (runsNposes and friends)
dataset: runsNposes
benchmark: ${dataset}
repeat_index: 0
skip_existing: true
max_num_inputs: null

# Input: a dataset dir with per-system folders, each containing {id}_protein.pdb + {id}_ligand.sdf
input_dir: ${oc.env:PROJECT_ROOT}/data/${dataset}

# Output: rank{N}.sdf per system, plus cached JSONs/mmCIFs
output_dir: ${oc.env:PROJECT_ROOT}/forks/alphafold3/prediction_outputs/${dataset}_${repeat_index}

# Environment / install
alphafold3_env: ${oc.env:PROJECT_ROOT}/envs/alphafold3        # symlink path; resolves to /mnt/...
alphafold3_dir: ${oc.env:PROJECT_ROOT}/forks/alphafold3/alphafold3
model_dir:      ${oc.env:PROJECT_ROOT}/forks/alphafold3/models

# Inference knobs
cuda_device_index: 0
num_samples: 5          # AF3 default: 5 diffusion samples per seed
num_seeds: 1            # one seed = 5 samples; bump to 2 for 10 total, etc.
num_recycles: 10        # AF3 default
timeout_seconds: 3600   # per-system hard cap; on timeout, log + skip

# Logging
logging:
  level: INFO
  file: alphafold3_docking.log
  console: true
  log_dir: ${oc.env:PROJECT_ROOT}/logs
```

**Key choices:**

- `alphafold3_env` points at the in-repo symlink, not `/mnt/...`, so the config is portable and consistent with other paths in the repo.
- `model_dir` is separate from `alphafold3_dir` because AF3's CLI takes `--model_dir` as its own flag and weights shouldn't live inside the upstream clone.
- No `inputs_csv` — we iterate `input_dir` directly like Chai, matching the runsNposes layout.
- No `db_dir` / MSA fields — we pass `--norun_data_pipeline` and bake empty MSAs into the JSON.
- Defaults `num_samples=5, num_seeds=1, num_recycles=10` match AF3 paper defaults; `timeout_seconds=3600` is a safety net, not an expectation.

---

## 5. Wrapper Module (`cogligandbench/models/alphafold3_inference.py`)

Public surface: `run_single(protein, ligand, output_dir, config=None, prefix=None, **kwargs)` and `run_dataset(config)`. Matches the contract expected by `engine._get_module(...).run_single(...)` at `cogligandbench/engine.py:139`.

### 5.1 Shared helper — protein sequence extraction

`_extract_protein_sequence` is currently defined inside `cogligandbench/models/chai_inference.py`. The design extracts it into `cogligandbench/utils/sequence.py` so both `chai_inference.py` and `alphafold3_inference.py` use the same implementation. The AA3→AA1 mapping (`AA_3TO1`) moves with it. Chai's module is updated to import from the new location — no behavior change.

### 5.2 Input-JSON construction

```python
def _build_af3_input_json(system_id: str, pdb_path: str, sdf_path: str) -> dict:
    """Construct the AF3 input-JSON dict for a single protein + ligand pair."""
    sequences = _extract_protein_sequence(pdb_path)   # list[str] of chain sequences
    smiles = _smiles_from_sdf(sdf_path)               # RDKit, canonical
    entries = []
    for i, seq in enumerate(sequences):
        entries.append({
            "protein": {
                "id": chr(ord("A") + i),
                "sequence": seq,
                "unpairedMsa": f">query\n{seq}\n",     # single-sequence A3M = no MSA
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
```

- **Single-sequence A3M** (`">query\n{seq}\n"`) is the no-MSA mechanism: syntactically valid A3M, zero evolutionary signal. Required because AF3 still parses the MSA field even under `--norun_data_pipeline`.
- **Ligand chain id is always `L`**, regardless of how many protein chains there are (A, B, C, …).

### 5.3 Subprocess invocation

```python
def _run_af3_subprocess(json_path: Path, out_dir: Path, config: dict):
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
        f"--num_seeds={config.get('num_seeds', 1)}",
        f"--num_recycles={config.get('num_recycles', 10)}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("cuda_device_index", 0))
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    subprocess.run(cmd, env=env, check=True, timeout=config.get("timeout_seconds", 3600))
```

- **Direct `{env}/bin/python`** instead of `conda run -p <prefix>` — faster and preserves stderr cleanly.
- `CUDA_VISIBLE_DEVICES` is set per call so multiple dataset-mode runs on different GPUs do not interfere.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` prevents JAX from grabbing the entire GPU upfront, matching the shared-card reality of the lab.
- **CLI flag names shown above (`--num_diffusion_samples`, `--num_seeds`, `--num_recycles`) are intent, not ground truth.** The exact flag spellings can drift between AF3 upstream releases. The implementation plan's first task after the env is installed is to run `{env}/bin/python run_alphafold.py --help` and reconcile the flags above with the actual surface the cloned version exposes. The four core flags (`--json_path`, `--output_dir`, `--model_dir`, `--norun_data_pipeline`) are stable across versions and will not change.

### 5.4 Output extraction (mmCIF → ranked SDFs)

AF3's output layout for a single JSON input is:

```
out_dir/{system_id_lowercase}/
  seed-{s}_sample-{n}/model.cif         # one per (seed, sample)
  ranking_scores.csv                     # columns: seed, sample, ranking_score
  {system_id}_summary_confidences.json
```

```python
def _extract_ranked_ligand_sdfs(af3_system_dir: Path, smiles: str, out_dir: Path, num_poses: int):
    scores = pd.read_csv(af3_system_dir / "ranking_scores.csv")
    scores = scores.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    template = Chem.MolFromSmiles(smiles)
    for rank, row in scores.head(num_poses).iterrows():
        cif = af3_system_dir / f"seed-{int(row.seed)}_sample-{int(row['sample'])}" / "model.cif"
        ligand_mol = _extract_ligand_from_cif(cif, template_mol=template)
        Chem.MolToMolFile(ligand_mol, str(out_dir / f"rank{rank + 1}.sdf"))


def _extract_ligand_from_cif(cif_path: Path, template_mol: Chem.Mol) -> Chem.Mol:
    """Pull the ligand (chain L) out of an AF3 mmCIF and recover bond orders from the SMILES template."""
    # 1. Parse the CIF with Biopython's MMCIFParser
    # 2. Select HETATM records belonging to chain "L" (the ligand)
    # 3. Build an editable RDKit Mol: add one atom per HETATM record (element from CIF) and a single Conformer with 3D coords
    # 4. raw_mol = mol.GetMol(); Chem.SanitizeMol(raw_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_ADJUSTHS)
    # 5. annotated = AllChem.AssignBondOrdersFromTemplate(template_mol, raw_mol)
    # 6. Return annotated
```

- **Bond-order recovery from the known SMILES** via `rdkit.Chem.AllChem.AssignBondOrdersFromTemplate`. We already fed the SMILES into AF3, so we reuse it here instead of guessing chemistry from 3D distances. This is both correct-by-construction and robust to unusual ligands.
- **Ranking follows AF3's own `ranking_scores.csv`** (highest score = `rank1`).
- If `ranking_scores.csv` is missing, raise `FileNotFoundError`. Task 7's `run_single` catches this in its per-system error handler and logs the failure.

### 5.5 Public API

```python
def run_single(protein, ligand, output_dir, config=None, prefix=None, **kwargs) -> str:
    """Dock a single protein+ligand pair. Writes rank{1..N}.sdf to output_dir."""
    # 1. Build JSON input to a temp path
    # 2. Run AF3 subprocess
    # 3. Extract ranked ligand SDFs from the AF3 output tree into output_dir
    # 4. Return output_dir

def run_dataset(config: dict):
    """Iterate config['input_dir'] and dock each subdirectory's protein+ligand pair.

    Per-system loop mirrors chai_inference.run_dataset:
      - skip_existing check (is rank1.sdf already present?)
      - build JSON → subprocess → extract → log timing or error
    """
```

### 5.6 Error handling

- **Subprocess nonzero exit** → log stderr, write `error_log.txt` into the per-system output directory, continue.
- **Timeout** → log and continue; the partial AF3 output tree is left in place for post-mortem.
- **OOM** (surfaces as JAX `XlaRuntimeError`) → caught by the generic exception branch, same behavior as above.
- **Ligand extraction failure** (e.g., AF3 produced a model but the ligand atoms are missing or bond-order recovery fails) → log, skip that pose, continue with other ranks.

---

## 6. Engine Registration

Three files change.

**`cogligandbench/engine.py`:**

- Add `"alphafold3"` to `SUPPORTED_METHODS`
- Add `"alphafold3"` entry to `_CONFIG_PATHS` pointing at `cogligand_config/model/alphafold3_inference.yaml`
- Add `"alphafold3"` entry to `_METHOD_MODULES` pointing at `cogligandbench.models.alphafold3_inference`
- Update the top-of-file docstring examples and the `Available methods` line

**`cogligandbench/models/alphafold3_inference.py`:** new module (Section 5).

**`CLAUDE.md`:**

- Add an `alphafold3` row to the method-summary table
- Add a new `### AlphaFold3` subsection under "Per-Method Details" following the Chai / DynamicBind template: inputs, outputs, post-processing, requirements, config fields

**No changes** to `cogligandbench/__init__.py`, `cogligandbench/data/`, or `cogligandbench/analysis/` — the unified `dock_engine` entry point and inline SDF output mean downstream RMSD scripts work without modification.

---

## 7. Testing

The repo's existing two-tier test pattern in `tests/test_engine_smoke.py` gives most of the coverage for free.

### 7.1 Free Level-1 coverage via `SUPPORTED_METHODS` parametrization

Tests in `tests/test_engine_smoke.py` that use `@pytest.mark.parametrize("method", SUPPORTED_METHODS)` automatically cover `alphafold3` once it is added to the tuple:

- `test_all_config_files_exist` — AF3 YAML exists at the registered path
- `test_config_loads` — AF3 YAML parses and resolves `${oc.env:PROJECT_ROOT}` interpolations
- `test_config_override` — `skip_existing` override works on AF3 config
- `test_module_imports` — `cogligandbench.models.alphafold3_inference` imports cleanly and exports `run_single` + `run_dataset`

No new test code for these — adding `"alphafold3"` to `SUPPORTED_METHODS` is enough.

### 7.2 New Level-2 end-to-end test

`tests/test_engine_smoke.py` gains a `test_alphafold3_single` method inside the `TestEndToEnd` class, following `test_chai_single`'s shape. It skips if `envs/alphafold3/bin/python` or `forks/alphafold3/models/af3.bin` is missing, otherwise runs AF3 on the `8gkf` fixture with `num_samples=1, num_seeds=1, num_recycles=3` for speed and asserts `rank1.sdf` exists.

### 7.3 New unit tests (`tests/test_alphafold3_inference.py`)

Subprocess-free, GPU-free unit coverage of the wrapper helpers:

1. **`test_build_af3_input_json_structure`** — 8gkf fixture PDB+SDF → `_build_af3_input_json` returns a dict with `dialect == "alphafold3"`, at least one `protein` entry with non-empty `sequence`, and a `ligand` entry whose SMILES round-trips via RDKit canonicalization.
2. **`test_build_af3_input_json_single_sequence_msa`** — every protein entry has `unpairedMsa` starting with `">query\n"` followed by the same string as `sequence`. Verifies the no-MSA invariant.
3. **`test_extract_ligand_from_cif_bond_orders`** — a tiny mmCIF fixture generated in `setup_method` from the 8gkf ligand SDF (just HETATM lines for chain `L`). Extract it with the known SMILES template. Assert the canonical SMILES of the result matches the template. Verifies `AssignBondOrdersFromTemplate` usage.
4. **`test_run_single_routes_correctly`** — monkeypatch `_run_af3_subprocess` to write a stub mmCIF + `ranking_scores.csv` into the expected output tree; call `run_single`; assert `rank1.sdf` exists. Verifies the `run_single` control flow without GPUs.
5. **`test_run_dataset_skip_existing`** — monkeypatch `_run_af3_subprocess` with a call counter; point `input_dir` at a tmp tree with two fake systems, one already containing `rank1.sdf`; call `run_dataset` with `skip_existing=True`; assert the subprocess was invoked exactly once.

No checked-in binary fixtures; mmCIF in test 3 is generated at runtime.

---

## 8. Out of Scope

The following are explicitly deferred and must be handled separately if needed later:

- **MSA / template databases.** Running AF3 with real MSAs requires downloading ~630 GB of public databases. If this becomes useful later, add `db_dir` to the config, drop `--norun_data_pipeline`, and populate `unpairedMsa` from the AF3 data pipeline.
- **Docker-based execution.** The user explicitly requested a conda env. The AF3 Dockerfile is not referenced or supported.
- **Multi-GPU scheduling.** `cuda_device_index` is configurable per run but there is no built-in round-robin.
- **PoseBench compatibility.** PoseBench is legacy; no code is ported from it, no imports resolve to it. This is enforced by the existing `test_no_posebench_imports` check in `tests/test_engine_smoke.py`.
- **Extra benchmark datasets beyond runsNposes.** The config structure supports other datasets (posebusters_benchmark, astex_diverse, dockgen, casp15) via `input_dir` override, but the design does not include preparing those datasets.

---

## 9. Open Questions

None at this time. The design was brainstormed end-to-end and all decisions were confirmed.
