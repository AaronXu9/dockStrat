# AlphaFold3 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AlphaFold3 as a first-class docking method in CogLigandBench, reachable via `dock_engine('alphafold3', ...)` in both single-molecule and dataset modes.

**Architecture:** Run AF3 in no-MSA / single-sequence mode through a subprocess that invokes `run_alphafold.py` in a dedicated conda env at `/mnt/katritch_lab2/aoxu/envs/alphafold3` (symlinked from `CogLigandBench/envs/alphafold3`). The wrapper module builds AF3's input JSON with empty MSAs, runs the subprocess, and extracts ranked ligand SDFs from the predicted mmCIFs using the known input SMILES for bond-order recovery.

**Tech Stack:** Python 3.11, RDKit, Biopython (for mmCIF parsing), Conda, AlphaFold3 (github.com/google-deepmind/alphafold3), JAX/CUDA, OmegaConf, pytest.

**Spec:** `docs/superpowers/specs/2026-04-10-alphafold3-integration-design.md`

---

## File Structure

| File | Responsibility | New / Modify |
|------|----------------|--------------|
| `cogligandbench/utils/sequence.py` | Shared PDB→protein-sequence helper (extracted from `chai_inference.py`) | NEW |
| `cogligandbench/models/chai_inference.py` | Import `AA_3TO1` and `_extract_protein_sequence` from `utils/sequence.py` instead of defining them locally | MODIFY |
| `cogligand_config/model/alphafold3_inference.yaml` | Hydra/OmegaConf config for AF3 docking | NEW |
| `cogligandbench/models/alphafold3_inference.py` | AF3 wrapper: input prep, subprocess invocation, mmCIF→SDF extraction, public `run_single` / `run_dataset` | NEW |
| `cogligandbench/engine.py` | Register `alphafold3` in `SUPPORTED_METHODS`, `_CONFIG_PATHS`, `_METHOD_MODULES`; update top-of-file docstring | MODIFY |
| `tests/test_alphafold3_inference.py` | Subprocess-free unit tests for the wrapper helpers (JSON construction, mmCIF extraction, run_single/run_dataset routing with mocks) | NEW |
| `tests/test_engine_smoke.py` | Add `test_alphafold3_single` to the Level-2 `TestEndToEnd` class | MODIFY |
| `scripts/install_alphafold3_env.sh` | Idempotent one-shot env install: conda env, AF3 clone, pip install, weight decompression, sanity check | NEW |
| `CLAUDE.md` | Add `alphafold3` row to method-summary table and a `### AlphaFold3` subsection under "Per-Method Details" | MODIFY |

---

## Task Order Rationale

Tasks 1–8 are pure code work — they use mocks for the subprocess and do **not** require AF3 to be installed. They can be implemented and verified end-to-end on any machine (no GPU, no env, no weights). Tasks 9–11 cover the heavy install + the live integration test + docs and run on the workstation that has the GPU and the weights.

This ordering means the agentic worker is never blocked on a long-running install while writing code, and the install is reproducible because the install script (Task 9) is committed.

---

## Task 1: Extract `_extract_protein_sequence` into a shared utility

The current `_extract_protein_sequence` and `AA_3TO1` live inside `cogligandbench/models/chai_inference.py`. The AlphaFold3 wrapper needs the same helpers, so we extract them into `cogligandbench/utils/sequence.py` first. This is a pure refactor — Chai's behavior must not change.

**Files:**
- Create: `cogligandbench/utils/sequence.py`
- Create: `tests/test_sequence_utils.py`
- Modify: `cogligandbench/models/chai_inference.py:19-46`

- [ ] **Step 1: Write the failing test for the shared utility**

Create `tests/test_sequence_utils.py`:

```python
"""Unit tests for the shared protein-sequence utility."""

import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PROTEIN = os.path.join(
    PROJECT_ROOT, "data", "runsNposes", "8gkf__1__1.A__1.J",
    "8gkf__1__1.A__1.J_protein.pdb",
)


class TestExtractProteinSequence:
    def test_returns_list_of_nonempty_sequences(self):
        from cogligandbench.utils.sequence import extract_protein_sequence

        if not os.path.exists(FIXTURE_PROTEIN):
            pytest.skip(f"Fixture not found: {FIXTURE_PROTEIN}")

        seqs = extract_protein_sequence(FIXTURE_PROTEIN)
        assert isinstance(seqs, list)
        assert len(seqs) >= 1
        for s in seqs:
            assert isinstance(s, str)
            assert len(s) > 0
            # All characters must be in the canonical 1-letter AA alphabet (or X for unknown)
            assert set(s).issubset(set("ACDEFGHIKLMNPQRSTVWYUX"))

    def test_aa_3to1_covers_standard_residues(self):
        from cogligandbench.utils.sequence import AA_3TO1

        for three, one in [
            ("ALA", "A"), ("ARG", "R"), ("GLY", "G"), ("LYS", "K"),
            ("VAL", "V"), ("HIS", "H"), ("HSD", "H"), ("MSE", "M"),
        ]:
            assert AA_3TO1[three] == one
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_sequence_utils.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'cogligandbench.utils.sequence'`

- [ ] **Step 3: Create the new shared utility**

Create `cogligandbench/utils/sequence.py`:

```python
"""Shared helpers for extracting protein sequences from PDB files.

Used by methods (chai, alphafold3, ...) that need the per-chain amino-acid
sequence as input. Extracted from chai_inference.py to avoid duplication.
"""

from typing import List

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HSD": "H", "HSE": "H", "HSP": "H", "HIE": "H", "HID": "H",
    "MSE": "M", "SEC": "U",
}


def extract_protein_sequence(pdb_path: str) -> List[str]:
    """Return one amino-acid sequence string per chain in the PDB file.

    Sequences are read from the CA atoms (one residue per CA), with insertions
    de-duplicated. Unknown residue codes are encoded as ``X``.
    """
    from biopandas.pdb import PandasPdb

    ppdb = PandasPdb().read_pdb(pdb_path)
    atoms = ppdb.df["ATOM"]
    ca = atoms[atoms["atom_name"] == "CA"].drop_duplicates(
        subset=["chain_id", "residue_number", "insertion"]
    )
    sequences: List[str] = []
    for _chain_id, group in ca.groupby("chain_id", sort=False):
        seq = "".join(AA_3TO1.get(r, "X") for r in group["residue_name"])
        if seq:
            sequences.append(seq)
    return sequences
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_sequence_utils.py -v`

Expected: PASS, both tests green.

- [ ] **Step 5: Update `chai_inference.py` to import from the new location**

In `cogligandbench/models/chai_inference.py`, replace lines 19–46 (the local `AA_3TO1` dict and the local `_extract_protein_sequence` function definition) with a single import:

```python
from cogligandbench.utils.sequence import AA_3TO1, extract_protein_sequence as _extract_protein_sequence
```

Place this import alongside the other top-level imports in `chai_inference.py` (after the existing `from cogligandbench.utils.log import get_custom_logger` line). Delete the now-unused `from biopandas.pdb import PandasPdb` import inside the function (it's now inside `utils/sequence.py`).

- [ ] **Step 6: Verify chai's existing tests still pass**

Run: `pytest tests/test_engine_smoke.py::TestMethodModules::test_module_imports -v`

Expected: PASS for all methods including `chai`.

Run: `python -c "from cogligandbench.models.chai_inference import _extract_protein_sequence, AA_3TO1; print(AA_3TO1['ALA'])"`

Expected: prints `A` (no ImportError).

- [ ] **Step 7: Commit**

```bash
git add cogligandbench/utils/sequence.py tests/test_sequence_utils.py cogligandbench/models/chai_inference.py
git commit -m "$(cat <<'EOF'
refactor: extract protein-sequence helper into cogligandbench.utils.sequence

Move AA_3TO1 and _extract_protein_sequence out of chai_inference.py and into
a shared utility so the upcoming alphafold3 wrapper can reuse them.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add the AlphaFold3 YAML config + register in `engine.py` + module skeleton

This task creates three things at once because they MUST land together to make the parametrized Level-1 tests in `tests/test_engine_smoke.py` pass: a config file, a stub module that exports `run_single` / `run_dataset`, and the engine registration.

**Files:**
- Create: `cogligand_config/model/alphafold3_inference.yaml`
- Create: `cogligandbench/models/alphafold3_inference.py`
- Modify: `cogligandbench/engine.py:25-54`

- [ ] **Step 1: Verify the Level-1 tests currently pass (baseline)**

Run: `pytest tests/test_engine_smoke.py::TestConfigLoading tests/test_engine_smoke.py::TestMethodModules -v`

Expected: PASS for all 6 currently registered methods (vina, gnina, chai, dynamicbind, unidock2, surfdock).

- [ ] **Step 2: Add `alphafold3` to `SUPPORTED_METHODS` and verify the test now FAILS**

Modify `cogligandbench/engine.py`. Find lines 36–54 and replace them with:

```python
SUPPORTED_METHODS = ("vina", "gnina", "chai", "dynamicbind", "unidock2", "surfdock", "alphafold3")

_CONFIG_PATHS = {
    "vina":         os.path.join(_PROJECT_ROOT, "cogligand_config", "model", "vina_inference.yaml"),
    "gnina":        os.path.join(_PROJECT_ROOT, "cogligand_config", "model", "gnina_inference.yaml"),
    "chai":         os.path.join(_PROJECT_ROOT, "cogligand_config", "model", "chai_inference.yaml"),
    "dynamicbind":  os.path.join(_PROJECT_ROOT, "cogligand_config", "model", "dynamicbind_inference.yaml"),
    "unidock2":     os.path.join(_PROJECT_ROOT, "cogligand_config", "model", "unidock2_inference.yaml"),
    "surfdock":     os.path.join(_PROJECT_ROOT, "cogligand_config", "model", "surfdock_inference.yaml"),
    "alphafold3":   os.path.join(_PROJECT_ROOT, "cogligand_config", "model", "alphafold3_inference.yaml"),
}

_METHOD_MODULES = {
    "vina":        "cogligandbench.models.vina_inference",
    "gnina":       "cogligandbench.models.gnina_inference",
    "chai":        "cogligandbench.models.chai_inference",
    "dynamicbind": "cogligandbench.models.dynamicbind_inference",
    "unidock2":    "cogligandbench.models.unidock2_inference",
    "surfdock":    "cogligandbench.models.surfdock_inference",
    "alphafold3":  "cogligandbench.models.alphafold3_inference",
}
```

Run: `pytest tests/test_engine_smoke.py::TestConfigLoading tests/test_engine_smoke.py::TestMethodModules -v`

Expected: 3 NEW failures for `alphafold3` parametrizations:
- `test_all_config_files_exist[...]`: FAIL (no config file)
- `test_config_loads[alphafold3]`: FAIL (FileNotFoundError)
- `test_module_imports[alphafold3]`: FAIL (No module named 'cogligandbench.models.alphafold3_inference')

- [ ] **Step 3: Update the engine top-of-file docstring**

In `cogligandbench/engine.py`, lines 1–25 contain a docstring with usage examples. Add `'alphafold3'` to the example list. Replace lines 13–24 with:

```python
    # Single-molecule mode — dock one protein-ligand pair
    dock_engine('vina',        protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('gnina',       protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('chai',        protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('dynamicbind', protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('unidock2',    protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('surfdock',    protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')
    dock_engine('alphafold3',  protein='receptor.pdb', ligand='ligand.sdf', output_dir='./out')

    # Pass any config override as a kwarg:
    dock_engine('vina', dataset='runsNposes', top_n=5, exhaustiveness=16)

Available methods: vina, gnina, chai, dynamicbind, unidock2, surfdock, alphafold3
"""
```

- [ ] **Step 4: Create the AlphaFold3 YAML config**

Create `cogligand_config/model/alphafold3_inference.yaml`:

```yaml
# AlphaFold3 inference configuration (no-MSA / single-sequence mode).
# See docs/superpowers/specs/2026-04-10-alphafold3-integration-design.md

# Dataset mode (runsNposes and friends)
dataset: runsNposes
benchmark: ${dataset}
repeat_index: 0
skip_existing: true
max_num_inputs: null

# Input: a dataset directory with per-system folders, each containing
# {id}_protein.pdb and {id}_ligand.sdf
input_dir: ${oc.env:PROJECT_ROOT}/data/${dataset}

# Output: rank{N}.sdf per system, plus cached AF3 JSON inputs and mmCIF outputs
output_dir: ${oc.env:PROJECT_ROOT}/forks/alphafold3/prediction_outputs/${dataset}_${repeat_index}

# Environment / install
alphafold3_env: ${oc.env:PROJECT_ROOT}/envs/alphafold3        # symlink → /mnt/katritch_lab2/aoxu/envs/alphafold3
alphafold3_dir: ${oc.env:PROJECT_ROOT}/forks/alphafold3/alphafold3
model_dir:      ${oc.env:PROJECT_ROOT}/forks/alphafold3/models

# Inference knobs
cuda_device_index: 0
num_samples: 5            # AF3 default: 5 diffusion samples per seed
num_seeds: 1              # one seed × num_samples = total poses
num_recycles: 10          # AF3 default
num_poses_to_keep: 5      # how many top-ranked SDFs to write
timeout_seconds: 3600     # per-system hard cap; on timeout, log + skip

# Logging
logging:
  level: INFO
  file: alphafold3_docking.log
  console: true
  log_dir: ${oc.env:PROJECT_ROOT}/logs
```

- [ ] **Step 5: Create the module skeleton with stub functions**

Create `cogligandbench/models/alphafold3_inference.py`:

```python
"""AlphaFold3 docking wrapper.

Runs AF3 in no-MSA / single-sequence mode via a subprocess in a dedicated
conda env, then extracts ligand SDFs from the predicted mmCIFs using the
known input SMILES for bond-order recovery.

Public surface (used by cogligandbench.engine):
  - run_single(protein, ligand, output_dir, config=None, prefix=None, **kwargs)
  - run_dataset(config)
"""

from __future__ import annotations

import os
from typing import Optional


def run_single(
    protein: str,
    ligand: str,
    output_dir: str,
    config: Optional[dict] = None,
    prefix: Optional[str] = None,
    **kwargs,
) -> str:
    """Dock a single protein+ligand pair with AlphaFold3.

    Implementation lands in a later task; this stub exists so the engine
    registration tests pass.
    """
    raise NotImplementedError("alphafold3 run_single not implemented yet")


def run_dataset(config: dict) -> None:
    """Run AlphaFold3 over an entire dataset directory.

    Implementation lands in a later task; this stub exists so the engine
    registration tests pass.
    """
    raise NotImplementedError("alphafold3 run_dataset not implemented yet")
```

- [ ] **Step 6: Run the Level-1 tests to verify everything is green**

Run: `pytest tests/test_engine_smoke.py::TestConfigLoading tests/test_engine_smoke.py::TestMethodModules -v`

Expected: PASS for all 7 methods, including the 3 new `alphafold3` parametrizations.

- [ ] **Step 7: Commit**

```bash
git add cogligandbench/engine.py cogligand_config/model/alphafold3_inference.yaml cogligandbench/models/alphafold3_inference.py
git commit -m "$(cat <<'EOF'
feat(alphafold3): register engine entry, config, and module skeleton

Adds alphafold3 to dock_engine's SUPPORTED_METHODS, ships the YAML config
for no-MSA inference, and creates the inference module with stub run_single
and run_dataset (NotImplementedError). Level-1 parametrized smoke tests
now cover alphafold3.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement `_smiles_from_sdf` and `_build_af3_input_json`

These two helpers are pure functions and are the foundation of the wrapper. We test-drive both.

**Files:**
- Modify: `cogligandbench/models/alphafold3_inference.py`
- Create: `tests/test_alphafold3_inference.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphafold3_inference.py`:

```python
"""Unit tests for the AlphaFold3 wrapper helpers (subprocess-free)."""

import json
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(PROJECT_ROOT, "data", "runsNposes", "8gkf__1__1.A__1.J")
FIXTURE_PROTEIN = os.path.join(FIXTURE_DIR, "8gkf__1__1.A__1.J_protein.pdb")
FIXTURE_LIGAND = os.path.join(FIXTURE_DIR, "8gkf__1__1.A__1.J_ligand.sdf")


def _require_fixture():
    if not (os.path.exists(FIXTURE_PROTEIN) and os.path.exists(FIXTURE_LIGAND)):
        pytest.skip(f"Fixture not found at {FIXTURE_DIR}")


class TestSmilesFromSdf:
    def test_returns_canonical_smiles(self):
        _require_fixture()
        from rdkit import Chem
        from cogligandbench.models.alphafold3_inference import _smiles_from_sdf

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        assert isinstance(smiles, str)
        assert len(smiles) > 0
        # Canonicalization should be a fixed point
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == smiles


class TestBuildAf3InputJson:
    def test_structure_has_required_top_level_keys(self):
        _require_fixture()
        from cogligandbench.models.alphafold3_inference import _build_af3_input_json

        d = _build_af3_input_json("8gkf_test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        assert d["dialect"] == "alphafold3"
        assert d["version"] == 2
        assert d["name"] == "8gkf_test"
        assert isinstance(d["modelSeeds"], list)
        assert len(d["modelSeeds"]) >= 1
        assert "sequences" in d and isinstance(d["sequences"], list)

    def test_sequences_have_one_protein_per_chain_and_one_ligand(self):
        _require_fixture()
        from cogligandbench.models.alphafold3_inference import _build_af3_input_json

        d = _build_af3_input_json("8gkf_test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        proteins = [s for s in d["sequences"] if "protein" in s]
        ligands = [s for s in d["sequences"] if "ligand" in s]
        assert len(proteins) >= 1
        assert len(ligands) == 1
        for p in proteins:
            assert p["protein"]["sequence"]
            assert p["protein"]["id"]                      # chain id, e.g. "A"
        assert ligands[0]["ligand"]["id"] == "L"
        assert ligands[0]["ligand"]["smiles"]

    def test_protein_entries_use_single_sequence_a3m(self):
        """The no-MSA invariant: every protein has unpairedMsa = '>query\\n{seq}\\n'."""
        _require_fixture()
        from cogligandbench.models.alphafold3_inference import _build_af3_input_json

        d = _build_af3_input_json("8gkf_test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        for entry in d["sequences"]:
            if "protein" not in entry:
                continue
            seq = entry["protein"]["sequence"]
            assert entry["protein"]["unpairedMsa"] == f">query\n{seq}\n"
            assert entry["protein"]["pairedMsa"] == ""
            assert entry["protein"]["templates"] == []

    def test_json_round_trips(self):
        """The output must be JSON-serializable as-is."""
        _require_fixture()
        from cogligandbench.models.alphafold3_inference import _build_af3_input_json

        d = _build_af3_input_json("8gkf_test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded == d
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_alphafold3_inference.py -v`

Expected: 5 FAILures, all with `ImportError: cannot import name '_smiles_from_sdf' (or '_build_af3_input_json') from 'cogligandbench.models.alphafold3_inference'`.

- [ ] **Step 3: Implement the helpers**

In `cogligandbench/models/alphafold3_inference.py`, add the following imports at the top (after `from typing import Optional`):

```python
from pathlib import Path
from typing import Dict, List
```

Then add the two helpers above the existing `run_single` stub:

```python
def _smiles_from_sdf(sdf_path: str) -> str:
    """Read the first molecule from an SDF and return its canonical SMILES."""
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
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
    from cogligandbench.utils.sequence import extract_protein_sequence

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_alphafold3_inference.py -v`

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add cogligandbench/models/alphafold3_inference.py tests/test_alphafold3_inference.py
git commit -m "$(cat <<'EOF'
feat(alphafold3): implement _smiles_from_sdf and _build_af3_input_json

Pure helpers that turn a (PDB, SDF) pair into the AF3 input JSON dict
with single-sequence A3Ms (no-MSA mode), one protein entry per chain,
and one ligand entry on chain L.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement `_extract_ligand_from_cif`

This converts an AF3-predicted mmCIF into an RDKit `Mol` with correct bond orders, using the known SMILES template. The test generates a tiny synthetic mmCIF on the fly from the 8gkf ligand SDF — no checked-in fixture is needed.

**Files:**
- Modify: `cogligandbench/models/alphafold3_inference.py`
- Modify: `tests/test_alphafold3_inference.py`

- [ ] **Step 1: Write the failing test (with a fixture-builder helper)**

Append the following to `tests/test_alphafold3_inference.py`:

```python
def _write_tiny_cif_from_sdf(sdf_path, cif_path, chain_id="L", resname="LIG"):
    """Build a minimal mmCIF containing chain ``chain_id`` HETATM records
    with the heavy-atom coordinates from the first molecule in ``sdf_path``.
    """
    from rdkit import Chem

    mol = next(Chem.SDMolSupplier(str(sdf_path), removeHs=True), None)
    assert mol is not None, f"Cannot read SDF: {sdf_path}"
    conf = mol.GetConformer()

    lines = [
        "data_test",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    ]
    for i, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        elem = atom.GetSymbol()
        # Atom name must be unique within the residue; use element + index
        atom_name = f"{elem}{i}"
        lines.append(
            f"HETATM {i} {elem} {atom_name} {resname} {chain_id} 1 "
            f"{pos.x:.3f} {pos.y:.3f} {pos.z:.3f} 1.00 0.00"
        )
    cif_path.write_text("\n".join(lines) + "\n")


class TestExtractLigandFromCif:
    def test_recovers_canonical_smiles_from_synthetic_cif(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from cogligandbench.models.alphafold3_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        cif_path = tmp_path / "tiny.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path)

        template_smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        template = Chem.MolFromSmiles(template_smiles)

        mol = _extract_ligand_from_cif(cif_path, template_mol=template)
        assert mol is not None
        # After bond-order recovery, the canonical SMILES should match
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == template_smiles

    def test_raises_when_chain_l_missing(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from cogligandbench.models.alphafold3_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        # Write a CIF where the ligand is on chain "Z" instead of "L"
        cif_path = tmp_path / "wrong_chain.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="Z")

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        with pytest.raises(ValueError, match="No ligand atoms"):
            _extract_ligand_from_cif(cif_path, template_mol=template)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_alphafold3_inference.py::TestExtractLigandFromCif -v`

Expected: 2 FAIL with `ImportError: cannot import name '_extract_ligand_from_cif'`.

- [ ] **Step 3: Implement `_extract_ligand_from_cif`**

In `cogligandbench/models/alphafold3_inference.py`, add the implementation above `run_single`:

```python
def _extract_ligand_from_cif(cif_path, template_mol):
    """Pull the ligand (chain L) out of an AF3 mmCIF and recover bond orders.

    Strategy: parse the mmCIF with Biopython, collect HETATM records on chain
    "L" into a synthetic PDB block, parse that block with RDKit (which infers
    connectivity from distances), then call AssignBondOrdersFromTemplate with
    the known SMILES template to recover the correct bond orders.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser
    from rdkit import Chem
    from rdkit.Chem import AllChem

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("af3_pred", str(cif_path))

    pdb_lines = []
    atom_idx = 1
    for model in structure:
        for chain in model:
            if chain.id != "L":
                continue
            for residue in chain:
                resname = (residue.get_resname() or "LIG")[:3].ljust(3)
                for atom in residue:
                    elem = (atom.element or atom.get_name()[0]).strip()
                    name = atom.get_name()[:4].ljust(4)
                    x, y, z = atom.coord
                    pdb_lines.append(
                        f"HETATM{atom_idx:>5} {name} {resname} L{1:>4}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
                        f"{elem.rjust(2)}"
                    )
                    atom_idx += 1
            break  # only one chain L

    if not pdb_lines:
        raise ValueError(f"No ligand atoms (chain L HETATM) found in {cif_path}")

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_alphafold3_inference.py::TestExtractLigandFromCif -v`

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add cogligandbench/models/alphafold3_inference.py tests/test_alphafold3_inference.py
git commit -m "$(cat <<'EOF'
feat(alphafold3): implement mmCIF→ligand SDF extraction with bond-order recovery

Parses chain L HETATM records out of AF3's predicted mmCIF, builds a PDB
block, parses it with RDKit, and recovers bond orders via the known SMILES
template using AssignBondOrdersFromTemplate.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement `_extract_ranked_ligand_sdfs`

This walks an AF3 output directory, reads `ranking_scores.csv`, and writes `rank{N}.sdf` files in descending score order.

**Files:**
- Modify: `cogligandbench/models/alphafold3_inference.py`
- Modify: `tests/test_alphafold3_inference.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_alphafold3_inference.py`:

```python
class TestExtractRankedLigandSdfs:
    def test_writes_rank_files_in_score_order(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from cogligandbench.models.alphafold3_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        # Build a fake AF3 system output dir with two samples
        af3_dir = tmp_path / "8gkf_test"
        af3_dir.mkdir()
        sample0 = af3_dir / "seed-1234_sample-0"
        sample1 = af3_dir / "seed-1234_sample-1"
        sample0.mkdir()
        sample1.mkdir()
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, sample0 / "model.cif")
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, sample1 / "model.cif")

        # ranking_scores: sample 1 ranked higher than sample 0
        (af3_dir / "ranking_scores.csv").write_text(
            "seed,sample,ranking_score\n"
            "1234,0,0.40\n"
            "1234,1,0.85\n"
        )

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        out = tmp_path / "out"
        out.mkdir()

        _extract_ranked_ligand_sdfs(af3_dir, smiles=smiles, out_dir=out, num_poses=2)

        rank1 = out / "rank1.sdf"
        rank2 = out / "rank2.sdf"
        assert rank1.exists()
        assert rank2.exists()
        # Both files should be readable as RDKit molecules
        m1 = next(Chem.SDMolSupplier(str(rank1), removeHs=True), None)
        m2 = next(Chem.SDMolSupplier(str(rank2), removeHs=True), None)
        assert m1 is not None
        assert m2 is not None

    def test_respects_num_poses_cap(self, tmp_path):
        _require_fixture()
        from cogligandbench.models.alphafold3_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        af3_dir = tmp_path / "sys"
        af3_dir.mkdir()
        for s in range(3):
            d = af3_dir / f"seed-1234_sample-{s}"
            d.mkdir()
            _write_tiny_cif_from_sdf(FIXTURE_LIGAND, d / "model.cif")
        (af3_dir / "ranking_scores.csv").write_text(
            "seed,sample,ranking_score\n"
            "1234,0,0.10\n"
            "1234,1,0.50\n"
            "1234,2,0.30\n"
        )

        out = tmp_path / "out"
        out.mkdir()
        _extract_ranked_ligand_sdfs(
            af3_dir, smiles=_smiles_from_sdf(FIXTURE_LIGAND), out_dir=out, num_poses=2,
        )
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]  # only top-2 written
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_alphafold3_inference.py::TestExtractRankedLigandSdfs -v`

Expected: 2 FAIL with `ImportError: cannot import name '_extract_ranked_ligand_sdfs'`.

- [ ] **Step 3: Implement `_extract_ranked_ligand_sdfs`**

In `cogligandbench/models/alphafold3_inference.py`, add above `run_single`:

```python
def _extract_ranked_ligand_sdfs(af3_system_dir, smiles, out_dir, num_poses):
    """Read ``ranking_scores.csv``, sort by score desc, write top-N rank{i}.sdf."""
    import pandas as pd
    from rdkit import Chem

    af3_system_dir = Path(af3_system_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = af3_system_dir / "ranking_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"AF3 ranking_scores.csv not found at {scores_path}")

    scores = pd.read_csv(scores_path)
    scores = scores.sort_values("ranking_score", ascending=False).reset_index(drop=True)

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse template SMILES: {smiles}")

    written = 0
    for rank, row in scores.head(num_poses).iterrows():
        seed = int(row["seed"])
        sample = int(row["sample"])
        cif = af3_system_dir / f"seed-{seed}_sample-{sample}" / "model.cif"
        if not cif.exists():
            continue
        try:
            mol = _extract_ligand_from_cif(cif, template_mol=template)
        except Exception:
            continue
        Chem.MolToMolFile(mol, str(out_dir / f"rank{rank + 1}.sdf"))
        written += 1
    return written
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_alphafold3_inference.py::TestExtractRankedLigandSdfs -v`

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add cogligandbench/models/alphafold3_inference.py tests/test_alphafold3_inference.py
git commit -m "$(cat <<'EOF'
feat(alphafold3): implement ranked ligand SDF extraction

Walks an AF3 system output directory, reads ranking_scores.csv, sorts by
score descending, and writes the top-N ligand poses as rank{i}.sdf using
the known SMILES template for bond-order recovery.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement `_run_af3_subprocess`

This is a thin shell over `subprocess.run` and is not unit-tested directly — it is exercised by the `run_single` test in Task 7 (via monkeypatch) and by the Level-2 end-to-end test in Task 11.

**Files:**
- Modify: `cogligandbench/models/alphafold3_inference.py`

- [ ] **Step 1: Add the implementation**

In `cogligandbench/models/alphafold3_inference.py`, add above `run_single`:

```python
def _run_af3_subprocess(json_path, out_dir, config):
    """Invoke AF3's run_alphafold.py via the dedicated conda env's Python.

    Raises ``subprocess.CalledProcessError`` on nonzero exit and
    ``subprocess.TimeoutExpired`` on hard timeout. Both are caught
    by callers (run_single / run_dataset).

    NOTE on CLI flag names: ``--num_diffusion_samples``, ``--num_seeds``,
    and ``--num_recycles`` are intent — the exact spellings can drift across
    AF3 upstream releases. Task 10 reconciles them against the actually-installed
    version. The four core flags (--json_path, --output_dir, --model_dir,
    --norun_data_pipeline) are stable.
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
        f"--num_seeds={config.get('num_seeds', 1)}",
        f"--num_recycles={config.get('num_recycles', 10)}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("cuda_device_index", 0))
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    subprocess.run(
        cmd,
        env=env,
        check=True,
        timeout=config.get("timeout_seconds", 3600),
    )
```

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `pytest tests/test_engine_smoke.py::TestMethodModules::test_module_imports -v`

Expected: PASS for all 7 methods.

- [ ] **Step 3: Run the existing alphafold3 unit tests to confirm no regression**

Run: `pytest tests/test_alphafold3_inference.py -v`

Expected: all 9 tests PASS (5 from Task 3 + 2 from Task 4 + 2 from Task 5).

- [ ] **Step 4: Commit**

```bash
git add cogligandbench/models/alphafold3_inference.py
git commit -m "$(cat <<'EOF'
feat(alphafold3): add _run_af3_subprocess wrapper

Calls AF3's run_alphafold.py via the dedicated conda env's Python with
the no-MSA flags set. CLI flag spellings are reconciled against the
installed AF3 in a later task; this commit captures the spec's intent.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Implement `run_single` (TDD with monkeypatched subprocess)

`run_single` orchestrates the per-call flow: write JSON, run subprocess, extract SDFs. The test monkeypatches `_run_af3_subprocess` so no GPU or AF3 install is needed.

**Files:**
- Modify: `cogligandbench/models/alphafold3_inference.py`
- Modify: `tests/test_alphafold3_inference.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_alphafold3_inference.py`:

```python
class TestRunSingle:
    def test_run_single_writes_rank_sdfs_with_mocked_subprocess(self, tmp_path, monkeypatch):
        _require_fixture()
        import cogligandbench.models.alphafold3_inference as af3_mod

        config = {
            "alphafold3_env": "/nonexistent/env",
            "alphafold3_dir": "/nonexistent/dir",
            "model_dir": "/nonexistent/models",
            "num_samples": 2,
            "num_seeds": 1,
            "num_poses_to_keep": 2,
        }

        # Stub subprocess: write a tiny AF3-like output tree directly
        def fake_subprocess(json_path, out_dir, cfg):
            import json
            with open(json_path) as fh:
                payload = json.load(fh)
            sys_id = payload["name"]
            af3_dir = Path(out_dir) / sys_id.lower()
            af3_dir.mkdir(parents=True, exist_ok=True)
            for s in range(2):
                d = af3_dir / f"seed-1234_sample-{s}"
                d.mkdir()
                _write_tiny_cif_from_sdf(FIXTURE_LIGAND, d / "model.cif")
            (af3_dir / "ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n1234,0,0.30\n1234,1,0.70\n"
            )

        monkeypatch.setattr(af3_mod, "_run_af3_subprocess", fake_subprocess)

        out = tmp_path / "out"
        out.mkdir()

        result = af3_mod.run_single(
            protein=FIXTURE_PROTEIN,
            ligand=FIXTURE_LIGAND,
            output_dir=str(out),
            config=config,
            prefix="8gkf_test",
        )
        assert result == str(out)
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_alphafold3_inference.py::TestRunSingle -v`

Expected: FAIL with `NotImplementedError: alphafold3 run_single not implemented yet`.

- [ ] **Step 3: Implement `run_single`**

In `cogligandbench/models/alphafold3_inference.py`, replace the existing `run_single` stub with:

```python
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
    smiles = _smiles_from_sdf(ligand)
    af3_system_dir = Path(output_dir) / prefix.lower()
    if not af3_system_dir.exists():
        raise FileNotFoundError(
            f"AF3 produced no output directory at {af3_system_dir}"
        )
    num_poses = int(config.get("num_poses_to_keep", config.get("num_samples", 5)))
    _extract_ranked_ligand_sdfs(
        af3_system_dir,
        smiles=smiles,
        out_dir=Path(output_dir),
        num_poses=num_poses,
    )
    return output_dir
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_alphafold3_inference.py::TestRunSingle -v`

Expected: PASS.

- [ ] **Step 5: Run the full alphafold3 test file to verify no regressions**

Run: `pytest tests/test_alphafold3_inference.py -v`

Expected: all 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add cogligandbench/models/alphafold3_inference.py tests/test_alphafold3_inference.py
git commit -m "$(cat <<'EOF'
feat(alphafold3): implement run_single

Build JSON → run subprocess → extract ranked SDFs. Tested with a
monkeypatched subprocess that writes a synthetic AF3 output tree, so
no GPU or AF3 install is needed for the unit test.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Implement `run_dataset` (TDD with mocked subprocess + skip-existing)

**Files:**
- Modify: `cogligandbench/models/alphafold3_inference.py`
- Modify: `tests/test_alphafold3_inference.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_alphafold3_inference.py`:

```python
class TestRunDataset:
    def _make_fake_dataset(self, root, system_ids):
        """Build data/runsNposes-style folders under ``root``."""
        import shutil
        for sid in system_ids:
            sys_dir = root / sid
            sys_dir.mkdir(parents=True)
            shutil.copy(FIXTURE_PROTEIN, sys_dir / f"{sid}_protein.pdb")
            shutil.copy(FIXTURE_LIGAND, sys_dir / f"{sid}_ligand.sdf")

    def test_run_dataset_iterates_systems_and_writes_rank_sdfs(self, tmp_path, monkeypatch):
        _require_fixture()
        import cogligandbench.models.alphafold3_inference as af3_mod

        input_dir = tmp_path / "data"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        self._make_fake_dataset(input_dir, ["sys_a", "sys_b"])

        call_count = {"n": 0}

        def fake_subprocess(json_path, out_dir, cfg):
            import json as _json
            call_count["n"] += 1
            with open(json_path) as fh:
                payload = _json.load(fh)
            sys_id = payload["name"]
            af3_dir = Path(out_dir) / sys_id.lower()
            af3_dir.mkdir(parents=True, exist_ok=True)
            (af3_dir / "seed-1234_sample-0").mkdir()
            _write_tiny_cif_from_sdf(
                FIXTURE_LIGAND, af3_dir / "seed-1234_sample-0" / "model.cif"
            )
            (af3_dir / "ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n1234,0,0.50\n"
            )

        monkeypatch.setattr(af3_mod, "_run_af3_subprocess", fake_subprocess)

        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "alphafold3_env": "/nonexistent",
            "alphafold3_dir": "/nonexistent",
            "model_dir": "/nonexistent",
            "num_samples": 1,
            "num_seeds": 1,
            "num_poses_to_keep": 1,
            "skip_existing": True,
            "max_num_inputs": None,
            "repeat_index": 0,
            "dataset": "fake",
            "logging": {
                "level": "INFO",
                "file": "alphafold3_test.log",
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        }
        af3_mod.run_dataset(config)

        assert call_count["n"] == 2
        assert (output_dir / "sys_a" / "rank1.sdf").exists()
        assert (output_dir / "sys_b" / "rank1.sdf").exists()

    def test_run_dataset_skips_already_completed_systems(self, tmp_path, monkeypatch):
        _require_fixture()
        import cogligandbench.models.alphafold3_inference as af3_mod

        input_dir = tmp_path / "data"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        self._make_fake_dataset(input_dir, ["already_done", "needs_work"])

        # Pre-populate one system's rank1.sdf
        done_dir = output_dir / "already_done"
        done_dir.mkdir()
        (done_dir / "rank1.sdf").write_text("dummy\n")

        call_count = {"n": 0}

        def fake_subprocess(json_path, out_dir, cfg):
            import json as _json
            call_count["n"] += 1
            with open(json_path) as fh:
                sys_id = _json.load(fh)["name"]
            af3_dir = Path(out_dir) / sys_id.lower()
            af3_dir.mkdir(parents=True, exist_ok=True)
            (af3_dir / "seed-1234_sample-0").mkdir()
            _write_tiny_cif_from_sdf(
                FIXTURE_LIGAND, af3_dir / "seed-1234_sample-0" / "model.cif"
            )
            (af3_dir / "ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n1234,0,0.50\n"
            )

        monkeypatch.setattr(af3_mod, "_run_af3_subprocess", fake_subprocess)

        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "alphafold3_env": "/nonexistent",
            "alphafold3_dir": "/nonexistent",
            "model_dir": "/nonexistent",
            "num_samples": 1,
            "num_seeds": 1,
            "num_poses_to_keep": 1,
            "skip_existing": True,
            "max_num_inputs": None,
            "repeat_index": 0,
            "dataset": "fake",
            "logging": {
                "level": "INFO",
                "file": "alphafold3_test.log",
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        }
        af3_mod.run_dataset(config)

        # Only "needs_work" should have been processed
        assert call_count["n"] == 1
        assert (output_dir / "needs_work" / "rank1.sdf").exists()
        # The pre-populated one should be untouched
        assert (output_dir / "already_done" / "rank1.sdf").read_text() == "dummy\n"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_alphafold3_inference.py::TestRunDataset -v`

Expected: 2 FAIL with `NotImplementedError: alphafold3 run_dataset not implemented yet`.

- [ ] **Step 3: Implement `run_dataset`**

In `cogligandbench/models/alphafold3_inference.py`, replace the existing `run_dataset` stub with:

```python
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

    from cogligandbench.utils.log import get_custom_logger

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    skip_existing = config.get("skip_existing", True)
    max_num_inputs = config.get("max_num_inputs", None)

    os.makedirs(output_dir, exist_ok=True)
    logger = get_custom_logger(
        "alphafold3", config,
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_alphafold3_inference.py::TestRunDataset -v`

Expected: 2 PASS.

- [ ] **Step 5: Run the full smoke test suite**

Run: `pytest tests/test_engine_smoke.py tests/test_alphafold3_inference.py tests/test_sequence_utils.py -v`

Expected: all green except for `@pytest.mark.slow` tests, which are skipped by default.

- [ ] **Step 6: Commit**

```bash
git add cogligandbench/models/alphafold3_inference.py tests/test_alphafold3_inference.py
git commit -m "$(cat <<'EOF'
feat(alphafold3): implement run_dataset with skip_existing

Iterates per-system folders, calls run_single for each, logs timings,
and survives per-system failures. skip_existing checks for rank1.sdf
in the output directory.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Write `scripts/install_alphafold3_env.sh`

This is the committed install recipe. It is idempotent (re-running is safe) and produces a working AF3 conda env at `/mnt/katritch_lab2/aoxu/envs/alphafold3` with the symlink, the cloned upstream code, and the decompressed weights. The script does not run automatically — Task 10 documents how the user runs it.

**Files:**
- Create: `scripts/install_alphafold3_env.sh`

- [ ] **Step 1: Create the install script**

Create `scripts/install_alphafold3_env.sh`:

```bash
#!/usr/bin/env bash
# Install AlphaFold3 for CogLigandBench.
#
# Creates a dedicated conda env at /mnt/katritch_lab2/aoxu/envs/alphafold3,
# symlinks it from CogLigandBench/envs/alphafold3, clones the AlphaFold3
# upstream repo into forks/alphafold3/alphafold3, pip-installs it into the
# env, and decompresses af3.bin.zst → forks/alphafold3/models/af3.bin.
#
# Idempotent: safe to re-run. Each step skips when the target already exists.
#
# Usage: bash scripts/install_alphafold3_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PARENT="/mnt/katritch_lab2/aoxu/envs"
ENV_PREFIX="${ENV_PARENT}/alphafold3"
ENV_LINK="${PROJECT_ROOT}/envs/alphafold3"
AF3_SRC="${PROJECT_ROOT}/forks/alphafold3/alphafold3"
AF3_MODELS="${PROJECT_ROOT}/forks/alphafold3/models"
WEIGHTS_ZST="${PROJECT_ROOT}/af3.bin.zst"
WEIGHTS_BIN="${AF3_MODELS}/af3.bin"

echo "[1/7] Ensuring env parent directory exists at ${ENV_PARENT}"
mkdir -p "${ENV_PARENT}"

echo "[2/7] Creating conda env at ${ENV_PREFIX}"
if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    conda create -y -p "${ENV_PREFIX}" \
        -c bioconda -c conda-forge \
        python=3.11 hmmer
else
    echo "      env already exists; skipping create"
fi

echo "[3/7] Symlinking ${ENV_LINK} → ${ENV_PREFIX}"
mkdir -p "${PROJECT_ROOT}/envs"
ln -sfn "${ENV_PREFIX}" "${ENV_LINK}"

echo "[4/7] Cloning AlphaFold3 source to ${AF3_SRC}"
if [ ! -d "${AF3_SRC}/.git" ]; then
    git clone https://github.com/google-deepmind/alphafold3 "${AF3_SRC}"
else
    echo "      AF3 source already cloned; skipping"
fi

echo "[5/7] pip-installing AlphaFold3 into ${ENV_PREFIX}"
"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip
if [ -f "${AF3_SRC}/dev-requirements.txt" ]; then
    "${ENV_PREFIX}/bin/python" -m pip install -r "${AF3_SRC}/dev-requirements.txt"
fi
"${ENV_PREFIX}/bin/python" -m pip install -e "${AF3_SRC}"

echo "[6/7] Decompressing weights → ${WEIGHTS_BIN}"
mkdir -p "${AF3_MODELS}"
if [ ! -f "${WEIGHTS_BIN}" ]; then
    if [ ! -f "${WEIGHTS_ZST}" ]; then
        echo "ERROR: ${WEIGHTS_ZST} not found. Place af3.bin.zst at the project root and re-run."
        exit 1
    fi
    if command -v zstd >/dev/null 2>&1; then
        zstd -d "${WEIGHTS_ZST}" -o "${WEIGHTS_BIN}"
    else
        "${ENV_PREFIX}/bin/python" - <<PY
import zstandard, pathlib
src = pathlib.Path("${WEIGHTS_ZST}")
dst = pathlib.Path("${WEIGHTS_BIN}")
with src.open("rb") as fh_in, dst.open("wb") as fh_out:
    zstandard.ZstdDecompressor().copy_stream(fh_in, fh_out)
PY
    fi
else
    echo "      weights already present; skipping"
fi

echo "[7/7] Sanity check: importing alphafold3 from the env"
"${ENV_PREFIX}/bin/python" -c "import alphafold3; print('alphafold3 import OK')"

echo
echo "Done. AF3 is installed at:"
echo "  env:     ${ENV_PREFIX}  (symlinked from ${ENV_LINK})"
echo "  source:  ${AF3_SRC}"
echo "  weights: ${WEIGHTS_BIN}"
echo
echo "Next: run 'bash ${PROJECT_ROOT}/envs/alphafold3/bin/python ${AF3_SRC}/run_alphafold.py --help' to see the CLI surface (Task 10)."
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x scripts/install_alphafold3_env.sh`

- [ ] **Step 3: Lint the script with shellcheck if available**

Run: `command -v shellcheck >/dev/null && shellcheck scripts/install_alphafold3_env.sh || echo "shellcheck not installed; skipping lint"`

Expected: either no output (lint clean) or the "skipping lint" message. Fix any errors that shellcheck reports.

- [ ] **Step 4: Commit**

```bash
git add scripts/install_alphafold3_env.sh
git commit -m "$(cat <<'EOF'
feat(alphafold3): add idempotent install script

Creates a dedicated conda env at /mnt/katritch_lab2/aoxu/envs/alphafold3,
symlinks it from CogLigandBench/envs/alphafold3, clones the upstream AF3
repo, pip-installs it, and decompresses af3.bin.zst into forks/alphafold3/
models/af3.bin. Each step is skipped if its target already exists.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Run the install + reconcile AF3 CLI flag names

This is a manual / interactive task that runs on the workstation that has the GPU. It produces a single commit if and only if AF3's actual CLI flag names differ from the spec.

**Files (potentially):**
- Modify: `cogligandbench/models/alphafold3_inference.py` (only if flag names need to change)

- [ ] **Step 1: Run the install script**

Run: `bash scripts/install_alphafold3_env.sh`

Expected: script completes with `alphafold3 import OK` printed at the end. The new env is at `/mnt/katritch_lab2/aoxu/envs/alphafold3`, symlinked from `envs/alphafold3`. Weights are at `forks/alphafold3/models/af3.bin`. AF3 source is cloned at `forks/alphafold3/alphafold3`.

If the script fails, fix the underlying issue and re-run. Do not commit changes to the install script unless they are needed for it to succeed.

- [ ] **Step 2: Inspect AF3's CLI surface**

Run: `envs/alphafold3/bin/python forks/alphafold3/alphafold3/run_alphafold.py --help 2>&1 | head -100`

Expected: a flag listing. Look specifically at the flags for:
- `--num_diffusion_samples` (or any other spelling for "samples per seed")
- `--num_seeds` (or any other spelling for "seed count")
- `--num_recycles` (or any other spelling for "recycles")
- `--norun_data_pipeline` (this should be present and stable)
- `--run_inference` (this should be present and stable)

Write down the actual spellings.

- [ ] **Step 3: Reconcile `_run_af3_subprocess` if needed**

In `cogligandbench/models/alphafold3_inference.py`, locate the `_run_af3_subprocess` function. If any of the flag names from Step 2 differ from what's in the code, replace the corresponding `f"--..."` lines with the actual flag spellings. If they all match, no edit is needed.

If you change any flag names, also update the **NOTE on CLI flag names** docstring in `_run_af3_subprocess` so it accurately describes which flags are currently in use.

- [ ] **Step 4: Run the unit tests to confirm the change did not break the mocked path**

Run: `pytest tests/test_alphafold3_inference.py -v`

Expected: all 12 tests still PASS.

- [ ] **Step 5: Commit (only if you changed `_run_af3_subprocess` in Step 3)**

```bash
git add cogligandbench/models/alphafold3_inference.py
git commit -m "$(cat <<'EOF'
fix(alphafold3): reconcile run_alphafold.py CLI flag names with installed version

The spec captured flag names by intent; this updates them to match the
actual AF3 release in envs/alphafold3 as verified by --help.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

If no changes were needed in Step 3, skip this commit and move to Task 11.

---

## Task 11: Add the Level-2 end-to-end test and update CLAUDE.md

Final task. Adds a slow end-to-end smoke test that runs real AF3 (skipped if env or weights are missing) and updates the project's CLAUDE.md to document the new method.

**Files:**
- Modify: `tests/test_engine_smoke.py:154-168` (add new test method to `TestEndToEnd`)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the Level-2 test to `tests/test_engine_smoke.py`**

In `tests/test_engine_smoke.py`, locate the `TestEndToEnd` class (around line 102). After `test_surfdock_single` (around line 200), add a new method:

```python
    def test_alphafold3_single(self):
        af3_env = os.path.join(PROJECT_ROOT, "envs", "alphafold3")
        af3_weights = os.path.join(PROJECT_ROOT, "forks", "alphafold3", "models", "af3.bin")
        af3_run = os.path.join(PROJECT_ROOT, "forks", "alphafold3", "alphafold3", "run_alphafold.py")
        if not os.path.exists(os.path.join(af3_env, "bin", "python")):
            pytest.skip(f"AlphaFold3 env not found: {af3_env}")
        if not os.path.exists(af3_weights):
            pytest.skip(f"AlphaFold3 weights not found: {af3_weights}")
        if not os.path.exists(af3_run):
            pytest.skip(f"AlphaFold3 source not found: {af3_run}")

        with tempfile.TemporaryDirectory() as tmp:
            result = dock_engine(
                "alphafold3",
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=tmp,
                prefix="8gkf_test",
                num_samples=1,
                num_seeds=1,
                num_recycles=3,         # fast path for the test
                num_poses_to_keep=1,
                timeout_seconds=1800,
            )
            assert result is not None
            sdfs = glob.glob(os.path.join(result, "rank*.sdf"))
            assert len(sdfs) >= 1, f"No rank*.sdf files found in {result}"
```

- [ ] **Step 2: Run the Level-2 test**

Run: `pytest tests/test_engine_smoke.py::TestEndToEnd::test_alphafold3_single -v -m slow -s`

Expected on a workstation with the env + weights installed: PASS (after several minutes of GPU compute). On any other machine: SKIPPED.

If it fails, diagnose by reading stderr from the AF3 subprocess. Common causes: (i) flag name mismatch from Task 10 not fully reconciled; (ii) JSON schema mismatch (AF3 may have updated `dialect`/`version` requirements); (iii) GPU OOM. Fix the underlying issue and re-run before moving on.

- [ ] **Step 3: Update CLAUDE.md — method-summary table**

In `CLAUDE.md`, locate the method-summary table under "## Method Reference" (around line 90). Add a new row at the end of the table, between the `surfdock` row and the closing line:

```markdown
| `alphafold3` | `rank{N}.sdf` | AF3 ranking_score (highest best) | `envs/alphafold3` (symlink) | AF3 source clone, decompressed `af3.bin` weights |
```

- [ ] **Step 4: Update CLAUDE.md — per-method details**

In `CLAUDE.md`, locate the "## Per-Method Details" section. After the SurfDock subsection (it ends with the dataset-mode note about precomputed Steps 1–3), append a new subsection:

````markdown
---

### AlphaFold3

**Input preprocessing:**
- Protein chain sequences extracted from the PDB via `cogligandbench.utils.sequence.extract_protein_sequence` (CA atoms, biopandas, AA3→AA1 mapping with `X` for unknowns).
- Ligand SMILES extracted from the SDF via RDKit (`Chem.SDMolSupplier` → `MolToSmiles`, canonical).
- An AF3 input JSON is built per system: one `protein` entry per chain (chain ids `A`, `B`, ..., each with a single-sequence A3M as `unpairedMsa`, empty `pairedMsa`, empty `templates`) and one `ligand` entry on chain `L` carrying the canonical SMILES. `dialect: "alphafold3"`, `version: 2`, `modelSeeds: [1234]`.
- This JSON is written to `{output_dir}/{prefix}_af3_input.json` before each run for inspection.

**Outputs:** `{output_dir}/rank{N}.sdf` — top-N ligand poses extracted from the predicted mmCIFs and bond-order-recovered against the input SMILES via `rdkit.Chem.AllChem.AssignBondOrdersFromTemplate`. Ranked by AF3's own `ranking_scores.csv` (highest score = `rank1.sdf`). Intermediate AF3 outputs (full mmCIFs, confidence JSONs) remain in `{output_dir}/{prefix_lowercase}/` for post-mortem.

**Post-processing:** None — extraction is inline. SDFs are ready for downstream RMSD analysis.

**Requirements:**
- Conda env at `{PROJECT_ROOT}/envs/alphafold3` (symlink to `/mnt/katritch_lab2/aoxu/envs/alphafold3`). Created by `bash scripts/install_alphafold3_env.sh`.
- AF3 source clone at `{PROJECT_ROOT}/forks/alphafold3/alphafold3` (created by the same install script).
- Decompressed weights at `{PROJECT_ROOT}/forks/alphafold3/models/af3.bin` (decompressed from `af3.bin.zst` by the install script).
- GPU required. `num_recycles=10`, `num_samples=5` on the default config; reduce for faster smoke tests.
- Runs in **single-sequence / no-MSA mode** — no MSA databases, no `db_dir`. Inference uses `--norun_data_pipeline`.
- Config: `cogligand_config/model/alphafold3_inference.yaml`
  - `cuda_device_index`: GPU to bind via `CUDA_VISIBLE_DEVICES`
  - `num_samples`, `num_seeds`, `num_recycles`: AF3 inference knobs
  - `num_poses_to_keep`: how many top-ranked SDFs to write per system
  - `timeout_seconds`: per-system hard cap (default 3600)
````

- [ ] **Step 5: Run the full test suite (no slow tests) one last time**

Run: `pytest tests/ -v`

Expected: all green. Slow tests are skipped without `-m slow`. The `test_no_posebench_imports` check in `tests/test_engine_smoke.py` should also pass — verify the new module does not import anything from `posebench`.

- [ ] **Step 6: Final commit**

```bash
git add tests/test_engine_smoke.py CLAUDE.md
git commit -m "$(cat <<'EOF'
feat(alphafold3): add Level-2 end-to-end test and document method in CLAUDE.md

Adds test_alphafold3_single under TestEndToEnd (skipped without env+weights),
plus a method-summary row and a per-method details subsection in CLAUDE.md
documenting inputs, outputs, requirements, and config knobs.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

**Spec coverage:**
- §1 Scope (full parity, no-MSA, runsNposes) → Tasks 2, 7, 8 cover both modes; config in Task 2 sets up runsNposes; no-MSA invariant tested in Task 3.
- §2 Directory layout → Tasks 1, 2, 9 create every file in the layout, including `cogligandbench/utils/sequence.py`.
- §3 Install flow → Task 9 (script) + Task 10 (run + reconcile).
- §4 Config → Task 2.
- §5.1 shared sequence helper → Task 1.
- §5.2 input JSON → Task 3.
- §5.3 subprocess → Task 6 (and reconciled in Task 10).
- §5.4 mmCIF→SDF + ranking → Tasks 4 + 5.
- §5.5 public API → Tasks 7 + 8.
- §5.6 error handling → Task 8 (try/except per system) + Task 7 (subprocess errors propagate).
- §6 engine registration → Task 2.
- §7 testing → Tasks 1, 3, 4, 5, 7, 8 (unit tests) + Task 11 (Level-2).
- §8 out-of-scope → respected throughout (no MSAs, no Docker, no multi-GPU, no PoseBench).

**Placeholder scan:** No "TBD"/"TODO"/"implement later" in any task. Every code-changing step shows the exact code or the exact line being changed. The single soft spot is Task 10's `_run_af3_subprocess` reconciliation, which is intentional — the spec acknowledges flag names can drift across upstream releases — and the task documents the diagnostic procedure rather than guessing.

**Type / signature consistency:**
- `extract_protein_sequence(pdb_path: str) -> List[str]` — Task 1, used in Task 3.
- `_smiles_from_sdf(sdf_path: str) -> str` — Task 3, used in Tasks 5, 7.
- `_build_af3_input_json(system_id: str, pdb_path: str, sdf_path: str) -> Dict` — Task 3, used in Task 7.
- `_extract_ligand_from_cif(cif_path, template_mol)` — Task 4, used in Task 5.
- `_extract_ranked_ligand_sdfs(af3_system_dir, smiles, out_dir, num_poses)` — Task 5, used in Task 7.
- `_run_af3_subprocess(json_path, out_dir, config)` — Task 6, used in Task 7.
- `run_single(protein, ligand, output_dir, config=None, prefix=None, **kwargs) -> str` — Tasks 2 (stub) → 7 (impl), used in Task 8 and (via the engine) in Task 11.
- `run_dataset(config: dict) -> None` — Tasks 2 (stub) → 8 (impl).
- `num_poses_to_keep` config key — defined in Task 2's YAML, read by Task 7's `run_single`, exercised by Tasks 7, 8, 11. Consistent.

All checks pass.
