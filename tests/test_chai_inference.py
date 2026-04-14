"""Unit tests for the Chai-1 wrapper helpers (subprocess-free)."""

import os
from pathlib import Path

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(PROJECT_ROOT, "data", "runsNposes", "8gkf__1__1.A__1.J")
FIXTURE_PROTEIN = os.path.join(FIXTURE_DIR, "8gkf__1__1.A__1.J_protein.pdb")
FIXTURE_LIGAND = os.path.join(FIXTURE_DIR, "8gkf__1__1.A__1.J_ligand.sdf")


def _require_fixture():
    if not (os.path.exists(FIXTURE_PROTEIN) and os.path.exists(FIXTURE_LIGAND)):
        pytest.skip(f"Fixture not found at {FIXTURE_DIR}")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _write_fake_chai_pdb(sdf_path, pdb_path, resname="LIG", translation=(0.0, 0.0, 0.0)):
    """Build a minimal PDB with some dummy protein ATOM lines and LIG atoms
    from the first molecule in ``sdf_path``.

    The ``resname`` parameter controls the residue name for ligand atoms
    (default ``LIG``). Set to something else to test the "no LIG found" path.
    """
    from rdkit import Chem

    mol = next(Chem.SDMolSupplier(str(sdf_path), removeHs=True), None)
    assert mol is not None, f"Cannot read SDF: {sdf_path}"
    conf = mol.GetConformer()

    lines = []
    # A few dummy protein atoms (chain A, residue ALA)
    lines.append(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C"
    )
    lines.append(
        "ATOM      2  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N"
    )

    # Ligand atoms with the specified residue name
    for i, atom in enumerate(mol.GetAtoms(), start=3):
        pos = conf.GetAtomPosition(atom.GetIdx())
        x = pos.x + translation[0]
        y = pos.y + translation[1]
        z = pos.z + translation[2]
        elem = atom.GetSymbol()
        name = f" {elem}{i:<3}"[:4]
        lines.append(
            f"ATOM  {i:>5} {name} {resname:>3} C{1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
            f"{elem.rjust(2)}"
        )

    lines.append("END")
    Path(pdb_path).write_text("\n".join(lines) + "\n")


def _write_fake_chai_cif(sdf_path, cif_path, resname="LIG", translation=(0.0, 0.0, 0.0)):
    """Build a minimal mmCIF with LIG atoms from the SDF (for CIF code path)."""
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
    # Dummy protein atom
    lines.append("ATOM 1 C CA ALA A 1 0.000 0.000 0.000 1.00 0.00")

    # Ligand atoms
    for i, atom in enumerate(mol.GetAtoms(), start=2):
        pos = conf.GetAtomPosition(atom.GetIdx())
        x = pos.x + translation[0]
        y = pos.y + translation[1]
        z = pos.z + translation[2]
        elem = atom.GetSymbol()
        atom_name = f"{elem}{i}"
        lines.append(
            f"HETATM {i} {elem} {atom_name} {resname} C 1 "
            f"{x:.3f} {y:.3f} {z:.3f} 1.00 0.00"
        )

    Path(cif_path).write_text("\n".join(lines) + "\n")


def _write_fake_npz(path, aggregate_score):
    """Write a minimal NPZ file mimicking Chai-1's scores.model_idx_*.npz."""
    import numpy as np

    np.savez(str(path), aggregate_score=np.array([aggregate_score], dtype=np.float32))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSmilesFromSdf:
    def test_returns_canonical_smiles(self):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.chai_inference import _smiles_from_sdf

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        assert isinstance(smiles, str)
        assert len(smiles) > 0
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == smiles

    def test_raises_value_error_for_missing_file(self, tmp_path):
        from dockstrat.models.chai_inference import _smiles_from_sdf

        missing = tmp_path / "does_not_exist.sdf"
        with pytest.raises(ValueError, match="Cannot open SDF file"):
            _smiles_from_sdf(str(missing))


class TestWriteFasta:
    def test_single_chain_protein(self, tmp_path):
        from dockstrat.models.chai_inference import _write_fasta

        fasta_path = str(tmp_path / "test.fasta")
        _write_fasta(fasta_path, "sys1", ["MKVL"], "CCO")
        content = Path(fasta_path).read_text()
        assert ">protein|sys1-chain-1\n" in content
        assert "MKVL\n" in content
        assert ">ligand|sys1-chain-2\n" in content
        assert "CCO\n" in content

    def test_multi_chain_protein(self, tmp_path):
        from dockstrat.models.chai_inference import _write_fasta

        fasta_path = str(tmp_path / "test.fasta")
        _write_fasta(fasta_path, "sys2", ["AAA", "BBB"], "c1ccccc1")
        content = Path(fasta_path).read_text()
        assert ">protein|sys2-chain-1\n" in content
        assert ">protein|sys2-chain-2\n" in content
        assert ">ligand|sys2-chain-3\n" in content
        assert "AAA\n" in content
        assert "BBB\n" in content
        assert "c1ccccc1\n" in content


class TestExtractLigandFromChaiOutput:
    def test_recovers_canonical_smiles_from_pdb(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.chai_inference import (
            _extract_ligand_from_chai_output, _smiles_from_sdf,
        )

        pdb_path = tmp_path / "pred.model_idx_0.pdb"
        _write_fake_chai_pdb(FIXTURE_LIGAND, pdb_path)

        template_smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        template = Chem.MolFromSmiles(template_smiles)

        mol = _extract_ligand_from_chai_output(pdb_path, template_mol=template)
        assert mol is not None
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == template_smiles

    def test_recovers_canonical_smiles_from_cif(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.chai_inference import (
            _extract_ligand_from_chai_output, _smiles_from_sdf,
        )

        cif_path = tmp_path / "pred.model_idx_0.cif"
        _write_fake_chai_cif(FIXTURE_LIGAND, cif_path)

        template_smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        template = Chem.MolFromSmiles(template_smiles)

        mol = _extract_ligand_from_chai_output(cif_path, template_mol=template)
        assert mol is not None
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == template_smiles

    def test_raises_when_no_lig_atoms(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.chai_inference import (
            _extract_ligand_from_chai_output, _smiles_from_sdf,
        )

        # Write a PDB where ligand has residue name "XXX" instead of "LIG"
        pdb_path = tmp_path / "no_lig.pdb"
        _write_fake_chai_pdb(FIXTURE_LIGAND, pdb_path, resname="XXX")

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        with pytest.raises(ValueError, match="No ligand atoms"):
            _extract_ligand_from_chai_output(pdb_path, template_mol=template)


class TestExtractRankedLigandSdfs:
    def test_writes_rank_files_in_score_order(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.chai_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        chai_dir = tmp_path / "chai_out"
        chai_dir.mkdir()

        # Model 0: original coordinates, low score
        _write_fake_chai_pdb(FIXTURE_LIGAND, chai_dir / "pred.model_idx_0.pdb")
        _write_fake_npz(chai_dir / "scores.model_idx_0.npz", 0.40)

        # Model 1: translated coordinates, high score
        _write_fake_chai_pdb(
            FIXTURE_LIGAND, chai_dir / "pred.model_idx_1.pdb",
            translation=(10.0, 0.0, 0.0),
        )
        _write_fake_npz(chai_dir / "scores.model_idx_1.npz", 0.85)

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        out = tmp_path / "out"
        out.mkdir()

        _extract_ranked_ligand_sdfs(chai_dir, smiles=smiles, out_dir=out, num_poses=2)

        rank1 = out / "rank1.sdf"
        rank2 = out / "rank2.sdf"
        assert rank1.exists()
        assert rank2.exists()

        m1 = next(Chem.SDMolSupplier(str(rank1), removeHs=True), None)
        m2 = next(Chem.SDMolSupplier(str(rank2), removeHs=True), None)
        assert m1 is not None
        assert m2 is not None

        # rank1 should be the high-scored model 1 (translated +10 along x)
        c1 = m1.GetConformer()
        c2 = m2.GetConformer()
        centroid1_x = sum(c1.GetAtomPosition(i).x for i in range(m1.GetNumAtoms())) / m1.GetNumAtoms()
        centroid2_x = sum(c2.GetAtomPosition(i).x for i in range(m2.GetNumAtoms())) / m2.GetNumAtoms()
        assert centroid1_x - centroid2_x > 5.0, (
            f"rank1 should be the translated (high-scored) model but got "
            f"centroid_x diff = {centroid1_x - centroid2_x}"
        )

    def test_respects_num_poses_cap(self, tmp_path):
        _require_fixture()
        from dockstrat.models.chai_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        chai_dir = tmp_path / "chai_out"
        chai_dir.mkdir()
        for idx in range(3):
            _write_fake_chai_pdb(FIXTURE_LIGAND, chai_dir / f"pred.model_idx_{idx}.pdb")
            _write_fake_npz(chai_dir / f"scores.model_idx_{idx}.npz", 0.1 * (idx + 1))

        out = tmp_path / "out"
        out.mkdir()
        _extract_ranked_ligand_sdfs(
            chai_dir, smiles=_smiles_from_sdf(FIXTURE_LIGAND), out_dir=out, num_poses=2,
        )
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]

    def test_top_scored_failure_still_yields_rank1(self, tmp_path, monkeypatch):
        """If extraction fails for the top-scored model, the second-best
        should land at rank1.sdf (not rank2.sdf)."""
        _require_fixture()
        import dockstrat.models.chai_inference as chai_mod
        from dockstrat.models.chai_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        chai_dir = tmp_path / "chai_out"
        chai_dir.mkdir()
        for idx in range(2):
            _write_fake_chai_pdb(FIXTURE_LIGAND, chai_dir / f"pred.model_idx_{idx}.pdb")
            _write_fake_npz(chai_dir / f"scores.model_idx_{idx}.npz", 0.9 - idx * 0.4)

        real_extract = chai_mod._extract_ligand_from_chai_output
        call_count = {"n": 0}

        def flaky_extract(output_path, template_mol):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("simulated extraction failure on first call")
            return real_extract(output_path, template_mol)

        monkeypatch.setattr(chai_mod, "_extract_ligand_from_chai_output", flaky_extract)

        out = tmp_path / "out"
        out.mkdir()
        written = _extract_ranked_ligand_sdfs(
            chai_dir, smiles=_smiles_from_sdf(FIXTURE_LIGAND), out_dir=out, num_poses=2,
        )

        assert written == 1
        assert (out / "rank1.sdf").exists()
        assert not (out / "rank2.sdf").exists()


class TestRunSingle:
    def test_run_single_writes_rank_sdfs_with_mocked_subprocess(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.chai_inference as chai_mod

        config = {
            "python_exec_path": "/nonexistent/python",
            "num_poses_to_keep": 2,
        }

        # Stub subprocess: write fake Chai output files directly
        def fake_subprocess(fasta_path, out_dir, cfg):
            out_dir = Path(out_dir)
            for idx in range(2):
                _write_fake_chai_pdb(FIXTURE_LIGAND, out_dir / f"pred.model_idx_{idx}.pdb")
                _write_fake_npz(out_dir / f"scores.model_idx_{idx}.npz", 0.3 + idx * 0.4)

        monkeypatch.setattr(chai_mod, "_run_chai_subprocess", fake_subprocess)

        out = tmp_path / "out"
        out.mkdir()

        result = chai_mod.run_single(
            protein=FIXTURE_PROTEIN,
            ligand=FIXTURE_LIGAND,
            output_dir=str(out),
            config=config,
            prefix="8gkf_test",
        )
        assert result == str(out)
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]

    def test_run_single_raises_without_python_exec_path(self, tmp_path):
        _require_fixture()
        from dockstrat.models.chai_inference import run_single

        with pytest.raises(ValueError, match="python_exec_path"):
            run_single(
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=str(tmp_path),
                config={},
                prefix="test",
            )


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
        import dockstrat.models.chai_inference as chai_mod

        input_dir = tmp_path / "data"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        self._make_fake_dataset(input_dir, ["sys_a", "sys_b"])

        call_count = {"n": 0}

        def fake_subprocess(fasta_path, out_dir, cfg):
            call_count["n"] += 1
            out_dir = Path(out_dir)
            for idx in range(2):
                _write_fake_chai_pdb(FIXTURE_LIGAND, out_dir / f"pred.model_idx_{idx}.pdb")
                _write_fake_npz(out_dir / f"scores.model_idx_{idx}.npz", 0.3 + idx * 0.2)

        monkeypatch.setattr(chai_mod, "_run_chai_subprocess", fake_subprocess)

        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "python_exec_path": "/nonexistent/python",
            "num_poses_to_keep": 2,
            "skip_existing": True,
            "max_num_inputs": None,
            "repeat_index": 0,
            "dataset": "fake",
            "logging": {
                "level": "INFO",
                "file": "chai_test.log",
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        }
        chai_mod.run_dataset(config)

        assert call_count["n"] == 2
        assert (output_dir / "sys_a" / "rank1.sdf").exists()
        assert (output_dir / "sys_b" / "rank1.sdf").exists()

    def test_run_dataset_skips_already_completed_systems(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.chai_inference as chai_mod

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

        def fake_subprocess(fasta_path, out_dir, cfg):
            call_count["n"] += 1
            out_dir = Path(out_dir)
            for idx in range(1):
                _write_fake_chai_pdb(FIXTURE_LIGAND, out_dir / f"pred.model_idx_{idx}.pdb")
                _write_fake_npz(out_dir / f"scores.model_idx_{idx}.npz", 0.50)

        monkeypatch.setattr(chai_mod, "_run_chai_subprocess", fake_subprocess)

        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "python_exec_path": "/nonexistent/python",
            "num_poses_to_keep": 1,
            "skip_existing": True,
            "max_num_inputs": None,
            "repeat_index": 0,
            "dataset": "fake",
            "logging": {
                "level": "INFO",
                "file": "chai_test.log",
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        }
        chai_mod.run_dataset(config)

        assert call_count["n"] == 1
        assert (output_dir / "needs_work" / "rank1.sdf").exists()
        assert (output_dir / "already_done" / "rank1.sdf").read_text() == "dummy\n"
