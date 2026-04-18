"""Unit tests for the AlphaFold3 wrapper helpers (subprocess-free)."""

import json
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


class TestSmilesFromSdf:
    def test_returns_canonical_smiles(self):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.alphafold3_inference import _smiles_from_sdf

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        assert isinstance(smiles, str)
        assert len(smiles) > 0
        # Canonicalization should be a fixed point
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == smiles

    def test_raises_value_error_for_missing_file(self, tmp_path):
        from dockstrat.models.alphafold3_inference import _smiles_from_sdf

        missing = tmp_path / "does_not_exist.sdf"
        with pytest.raises(ValueError, match="Cannot open SDF file"):
            _smiles_from_sdf(str(missing))


class TestBuildAf3InputJson:
    def test_structure_has_required_top_level_keys(self):
        _require_fixture()
        from dockstrat.models.alphafold3_inference import _build_af3_input_json

        d = _build_af3_input_json("8gkf_test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        assert d["dialect"] == "alphafold3"
        assert d["version"] == 2
        assert d["name"] == "8gkf_test"
        assert isinstance(d["modelSeeds"], list)
        assert len(d["modelSeeds"]) >= 1
        assert "sequences" in d and isinstance(d["sequences"], list)

    def test_sequences_have_one_protein_per_chain_and_one_ligand(self):
        _require_fixture()
        from dockstrat.models.alphafold3_inference import _build_af3_input_json

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
        from dockstrat.models.alphafold3_inference import _build_af3_input_json

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
        from dockstrat.models.alphafold3_inference import _build_af3_input_json

        d = _build_af3_input_json("8gkf_test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded == d


def _write_tiny_cif_from_sdf(
    sdf_path, cif_path, chain_id="L", resname="LIG", use_auth_asym_id=True,
    translation=(0.0, 0.0, 0.0),
):
    """Build a minimal mmCIF containing chain ``chain_id`` HETATM records
    with the heavy-atom coordinates from the first molecule in ``sdf_path``.

    When ``use_auth_asym_id`` is True (default), the file includes BOTH
    ``_atom_site.auth_asym_id`` and ``_atom_site.label_asym_id`` columns.
    When False, only ``_atom_site.label_asym_id`` is written, exercising the
    fallback code path in ``_extract_ligand_from_cif``.
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
    ]
    if use_auth_asym_id:
        lines.append("_atom_site.auth_asym_id")
        lines.append("_atom_site.auth_seq_id")
    lines += [
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    ]
    for i, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        x = pos.x + translation[0]
        y = pos.y + translation[1]
        z = pos.z + translation[2]
        elem = atom.GetSymbol()
        # Atom name must be unique within the residue; use element + index
        atom_name = f"{elem}{i}"
        if use_auth_asym_id:
            lines.append(
                f"HETATM {i} {elem} {atom_name} {resname} {chain_id} 1 "
                f"{chain_id} 1 "
                f"{x:.3f} {y:.3f} {z:.3f} 1.00 0.00"
            )
        else:
            lines.append(
                f"HETATM {i} {elem} {atom_name} {resname} {chain_id} 1 "
                f"{x:.3f} {y:.3f} {z:.3f} 1.00 0.00"
            )
    cif_path.write_text("\n".join(lines) + "\n")


class TestExtractLigandFromCif:
    def test_recovers_canonical_smiles_from_synthetic_cif(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.alphafold3_inference import (
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
        from dockstrat.models.alphafold3_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        # Write a CIF where the ligand is on chain "Z" instead of "L"
        cif_path = tmp_path / "wrong_chain.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="Z")

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        with pytest.raises(ValueError, match="No ligand atoms"):
            _extract_ligand_from_cif(cif_path, template_mol=template)

    def test_uses_label_asym_id_when_auth_asym_id_missing(self, tmp_path):
        """Verify the fallback to label_asym_id when auth_asym_id column is absent."""
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.alphafold3_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        cif_path = tmp_path / "label_only.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, use_auth_asym_id=False)

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        mol = _extract_ligand_from_cif(cif_path, template_mol=template)
        assert mol is not None
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == _smiles_from_sdf(FIXTURE_LIGAND)


class TestExtractRankedLigandSdfs:
    def test_writes_rank_files_in_score_order(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.alphafold3_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        # Build a fake AF3 system output dir with two samples having
        # DISTINCT geometries so we can verify which one lands at rank1.
        af3_dir = tmp_path / "8gkf_test"
        af3_dir.mkdir()
        sample0 = af3_dir / "seed-1234_sample-0"
        sample1 = af3_dir / "seed-1234_sample-1"
        sample0.mkdir()
        sample1.mkdir()
        name = "8gkf_test"
        # Sample 0: original coordinates
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, sample0 / f"{name}_seed-1234_sample-0_model.cif")
        # Sample 1: translated by (10, 0, 0) Å so we can tell them apart
        _write_tiny_cif_from_sdf(
            FIXTURE_LIGAND, sample1 / f"{name}_seed-1234_sample-1_model.cif", translation=(10.0, 0.0, 0.0)
        )

        # ranking_scores: sample 1 (translated) ranked higher than sample 0
        (af3_dir / f"{name}_ranking_scores.csv").write_text(
            "seed,sample,ranking_score\n"
            "1234,0,0.40\n"
            "1234,1,0.85\n"
        )

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        out = tmp_path / "out"
        out.mkdir()

        _extract_ranked_ligand_sdfs(af3_dir, name=name, smiles=smiles, out_dir=out, num_poses=2)

        rank1 = out / "rank1.sdf"
        rank2 = out / "rank2.sdf"
        assert rank1.exists()
        assert rank2.exists()

        m1 = next(Chem.SDMolSupplier(str(rank1), removeHs=True), None)
        m2 = next(Chem.SDMolSupplier(str(rank2), removeHs=True), None)
        assert m1 is not None
        assert m2 is not None

        # rank1 should be the high-scored sample 1 (translated +10 Å along x).
        # Compare centroid x-coordinates: rank1 must be ~10 Å further along x
        # than rank2. This confirms ordering is descending by score.
        c1 = m1.GetConformer()
        c2 = m2.GetConformer()
        centroid1_x = sum(c1.GetAtomPosition(i).x for i in range(m1.GetNumAtoms())) / m1.GetNumAtoms()
        centroid2_x = sum(c2.GetAtomPosition(i).x for i in range(m2.GetNumAtoms())) / m2.GetNumAtoms()
        assert centroid1_x - centroid2_x > 5.0, (
            f"rank1 should be the translated (high-scored) sample but got "
            f"centroid_x diff = {centroid1_x - centroid2_x}"
        )

    def test_respects_num_poses_cap(self, tmp_path):
        _require_fixture()
        from dockstrat.models.alphafold3_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        name = "sys"
        af3_dir = tmp_path / name
        af3_dir.mkdir()
        for s in range(3):
            d = af3_dir / f"seed-1234_sample-{s}"
            d.mkdir()
            _write_tiny_cif_from_sdf(FIXTURE_LIGAND, d / f"{name}_seed-1234_sample-{s}_model.cif")
        (af3_dir / f"{name}_ranking_scores.csv").write_text(
            "seed,sample,ranking_score\n"
            "1234,0,0.10\n"
            "1234,1,0.50\n"
            "1234,2,0.30\n"
        )

        out = tmp_path / "out"
        out.mkdir()
        _extract_ranked_ligand_sdfs(
            af3_dir, name=name, smiles=_smiles_from_sdf(FIXTURE_LIGAND), out_dir=out, num_poses=2,
        )
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]  # only top-2 written

    def test_top_scored_failure_still_yields_rank1(self, tmp_path, monkeypatch):
        """If _extract_ligand_from_cif fails for the top-scored sample, the
        second-best sample should still land at rank1.sdf (not rank2.sdf)."""
        _require_fixture()
        import dockstrat.models.alphafold3_inference as af3_mod
        from dockstrat.models.alphafold3_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        name = "sys"
        af3_dir = tmp_path / name
        af3_dir.mkdir()
        for s in range(2):
            d = af3_dir / f"seed-1234_sample-{s}"
            d.mkdir()
            _write_tiny_cif_from_sdf(FIXTURE_LIGAND, d / f"{name}_seed-1234_sample-{s}_model.cif")
        (af3_dir / f"{name}_ranking_scores.csv").write_text(
            "seed,sample,ranking_score\n"
            "1234,0,0.90\n"  # top-scored, will be made to fail
            "1234,1,0.50\n"  # second-best, should land at rank1
        )

        real_extract = af3_mod._extract_ligand_from_cif
        call_count = {"n": 0}

        def flaky_extract(cif_path, template_mol):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("simulated extraction failure on first call")
            return real_extract(cif_path, template_mol)

        monkeypatch.setattr(af3_mod, "_extract_ligand_from_cif", flaky_extract)

        out = tmp_path / "out"
        out.mkdir()
        written = _extract_ranked_ligand_sdfs(
            af3_dir, name=name, smiles=_smiles_from_sdf(FIXTURE_LIGAND), out_dir=out, num_poses=2,
        )

        # Only one SDF should be written (since first extract failed)
        assert written == 1
        assert (out / "rank1.sdf").exists(), (
            "rank1.sdf should exist even when the top-scored sample's extraction fails"
        )
        assert not (out / "rank2.sdf").exists(), (
            "rank2.sdf should NOT exist when only one extraction succeeded"
        )


class TestRunSingle:
    def test_run_single_writes_rank_sdfs_with_mocked_subprocess(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.alphafold3_inference as af3_mod

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
            af3_dir = Path(out_dir) / sys_id
            af3_dir.mkdir(parents=True, exist_ok=True)
            for s in range(2):
                d = af3_dir / f"seed-1234_sample-{s}"
                d.mkdir()
                _write_tiny_cif_from_sdf(FIXTURE_LIGAND, d / f"{sys_id}_seed-1234_sample-{s}_model.cif")
            (af3_dir / f"{sys_id}_ranking_scores.csv").write_text(
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
        import dockstrat.models.alphafold3_inference as af3_mod

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
            af3_dir = Path(out_dir) / sys_id
            af3_dir.mkdir(parents=True, exist_ok=True)
            (af3_dir / "seed-1234_sample-0").mkdir()
            _write_tiny_cif_from_sdf(
                FIXTURE_LIGAND, af3_dir / "seed-1234_sample-0" / f"{sys_id}_seed-1234_sample-0_model.cif"
            )
            (af3_dir / f"{sys_id}_ranking_scores.csv").write_text(
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
        import dockstrat.models.alphafold3_inference as af3_mod

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
            af3_dir = Path(out_dir) / sys_id
            af3_dir.mkdir(parents=True, exist_ok=True)
            (af3_dir / "seed-1234_sample-0").mkdir()
            _write_tiny_cif_from_sdf(
                FIXTURE_LIGAND, af3_dir / "seed-1234_sample-0" / f"{sys_id}_seed-1234_sample-0_model.cif"
            )
            (af3_dir / f"{sys_id}_ranking_scores.csv").write_text(
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
