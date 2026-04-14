# Figures

Visualizations referenced by the top-level `README.md`.

## Pose Comparison (4 PNGs)

| File | Method | Similarity regime | System |
|------|--------|-------------------|--------|
| `af3_high_sim.png`   | AlphaFold3 | High (SuCOS 85.2) | `7qhl__1__1.A__1.E` |
| `gnina_high_sim.png` | GNINA      | High (SuCOS 85.2) | `7qhl__1__1.A__1.E` |
| `af3_low_sim.png`    | AlphaFold3 | Low (SuCOS 3.0)   | `8cfb__1__1.A__1.L` |
| `gnina_low_sim.png`  | GNINA      | Low (SuCOS 3.0)   | `8cfb__1__1.A__1.L` |

Predicted ligand: **magenta sticks**. Crystal ligand: **green sticks**. Receptor: **grey cartoon**. Binding-pocket residues (within 5 Å of the crystal ligand): **thin white sticks**. Panels are framed on the native binding pocket — for the AF3 low-sim case the predicted ligand is ~38 Å away and therefore off-screen.

## How to regenerate

```bash
conda run -n base python scripts/generate_pose_comparison_figure.py
```

The script:
1. Reads `data/runsNposes_archive/zenodo_downloads/annotations.csv` for the SuCOS similarity (`sucos_shape_pocket_qcov` column, percentage 0–100).
2. Discovers systems with all four artifacts: crystal PDB+SDF, AF3 prediction, GNINA prediction.
3. Picks one HIGH-similarity system (SuCOS ≥ 70, both AF3 and GNINA succeed) and one LOW-similarity system (SuCOS ≤ 20, AF3 fails > 4 Å, GNINA succeeds < 2 Å). The LOW pick maximizes the AF3-vs-GNINA RMSD contrast.
4. Identifies the AF3 ligand chain by atom-count match against the crystal heavy atoms (AF3 may emit multiple HETATM chains).
5. Renders 4 PyMOL panels (1200×900, ray-traced) with the AF3 protein aligned to the crystal receptor before showing only its matching ligand chain.

The exact selection metadata (system IDs, SuCOS, RMSDs) is dumped to `selection_metadata.json` after each run.
