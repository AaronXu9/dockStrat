# CogLigandBench Documentation Cleanup — Design Spec
**Date:** 2026-03-28

## Goal
Remove remaining PoseBench branding/content from the repository and replace with CogLigandBench-specific documentation. Audience is split: broad scientific audience (README) and lab members (docs/api.md).

---

## Files to Create or Rewrite

### 1. `README.md` — Complete rewrite (~300-400 lines)
**Audience:** Broad scientific audience unfamiliar with the project.

**Sections:**
1. **Header** — Title `# CogLigandBench`, MIT/Python badges, one-sentence tagline: *"A benchmarking framework for protein-ligand docking methods evaluated against experimental crystal structures."*
2. **Why Crystal Structures** — 2-3 sentence motivation (crystal structures isolate docking error from structure prediction error; mirrors real SBDD). Dataset table: runsNposes (~1,280), Astex Diverse (85), PoseBusters (428), DockGen (189).
3. **Supported Methods & Results** — Methods table (method, type, conda env). Preliminary runsNposes results table (% ≤ 2Å top-1, median RMSD). UniDock2 flagged as tentative.
4. **Installation & Quickstart** — `pip install -e .`, `.env.example` copy, minimal `dock_engine` usage for single molecule and dataset mode. Links to `docs/quickstart.md` and `docs/api.md`.
5. **Project Structure** — concise directory tree for `cogligandbench/`, `cogligand_config/`, `data/`, `forks/`, `docs/`, `tests/`.
6. **Citation & Acknowledgements** — BibTeX block, one-line acknowledgement of PoseBench upstream.

**Results to include (runsNposes, top-1 pose):**

| Method | % ≤ 2Å RMSD | Median RMSD |
|--------|------------|-------------|
| SurfDock | 59.5% | 1.56 Å |
| GNINA | 33.9% | 2.56 Å |
| ICM-RTCNN | 27.5% | 3.42 Å |
| ICM | 25.7% | 3.69 Å |
| Vina | 7.5% | 7.43 Å |
| UniDock2 | 2.0%† | 24.35 Å† |

†UniDock2 tentative — run artifact under investigation.

---

### 2. `docs/api.md` — New file (~150 lines)
**Audience:** Lab members already familiar with docking.

**Sections:**
1. `dock_engine` full signature, parameters, return value
2. Single-molecule vs dataset mode switching
3. Per-method kwargs table (exhaustiveness, num_poses, python_exec_path, etc.)
4. Output file naming conventions per method
5. Config override pattern (how kwargs map to YAML keys)
6. Standard RMSD calculation snippet (RDKit CalcRMS)

---

## Files to Moderately Rewrite

### 3. `docs/running_methods.md`
- Remove sections for deleted methods: DiffDock, FABind, NeuralPLexer, RFAA
- Fix `export PROJECT_ROOT=/path/to/PoseBench` → `/path/to/CogLigandBench`
- Keep: Vina, GNINA, ICM, Chai-1, DynamicBind, SurfDock sections
- Update config path references from `configs/model/` to `cogligand_config/model/`

### 4. `docs/rmsd_analysis.md`
- Replace all `/home/aoxu/projects/PoseBench` paths with `${PROJECT_ROOT}` or generic `/path/to/CogLigandBench`
- Expand scope from Vina/UniDock2 only to all 6 supported methods
- Clarify this is for crystal structure (holo) analysis

---

## Files to Delete

### 5. `notebooks/ORGANIZATION_PLAN.md`
- Entirely describes upstream PoseBench notebook organization
- Not applicable to CogLigandBench
- Action: `git rm`

---

## Minor Fixes

### 6. `docs/quickstart.md`
- Line ~12: `export PROJECT_ROOT=/path/to/PoseBench` → `/path/to/CogLigandBench`

### 7. `notebooks/04_utils/README_icm_rmsd.md`
- Replace `/home/aoxu/projects/PoseBench` path reference with generic CogLigandBench path

---

## Implementation Order (Option A — Priority-first)
1. Write `README.md` (most visible)
2. Write `docs/api.md` (new file)
3. Rewrite `docs/running_methods.md`
4. Rewrite `docs/rmsd_analysis.md`
5. Delete `notebooks/ORGANIZATION_PLAN.md`
6. Minor fixes: `docs/quickstart.md`, `notebooks/04_utils/README_icm_rmsd.md`
7. Commit all

---

## Success Criteria
- No "PoseBench" in README.md title, badges, or main content (one acknowledgement line is fine)
- All method sections in docs reference `cogligand_config/` paths, not `configs/`
- No hardcoded `/home/aoxu/` paths in any tracked `.md` file
- `docs/api.md` covers the full `dock_engine` interface with per-method kwargs
