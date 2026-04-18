"""Unit tests for the Boltz wrapper helpers (subprocess-free)."""

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


def _write_tiny_cif_from_sdf(
    sdf_path, cif_path, chain_id="B", resname="LIG", use_auth_asym_id=True,
    translation=(0.0, 0.0, 0.0),
):
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


class TestSmilesFromSdf:
    def test_returns_canonical_smiles(self):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.boltz_inference import _smiles_from_sdf

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        assert isinstance(smiles, str)
        assert len(smiles) > 0
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == smiles

    def test_raises_value_error_for_missing_file(self, tmp_path):
        from dockstrat.models.boltz_inference import _smiles_from_sdf

        missing = tmp_path / "does_not_exist.sdf"
        with pytest.raises(ValueError, match="Cannot open SDF file"):
            _smiles_from_sdf(str(missing))


class TestBuildBoltzInputYaml:
    def test_single_chain_protein_structure(self):
        _require_fixture()
        from dockstrat.models.boltz_inference import _build_boltz_input_yaml

        d, lig_chain = _build_boltz_input_yaml("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        assert d["version"] == 1
        assert "sequences" in d
        proteins = [s for s in d["sequences"] if "protein" in s]
        ligands = [s for s in d["sequences"] if "ligand" in s]
        assert len(proteins) >= 1
        assert len(ligands) == 1
        assert ligands[0]["ligand"]["id"] == lig_chain
        assert ligands[0]["ligand"]["smiles"]

    def test_ligand_chain_id_is_next_after_proteins(self):
        _require_fixture()
        from dockstrat.models.boltz_inference import _build_boltz_input_yaml

        d, lig_chain = _build_boltz_input_yaml("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        proteins = [s for s in d["sequences"] if "protein" in s]
        expected_lig_chain = chr(ord("A") + len(proteins))
        assert lig_chain == expected_lig_chain

    def test_no_msa_sets_msa_empty_on_proteins(self):
        _require_fixture()
        from dockstrat.models.boltz_inference import _build_boltz_input_yaml

        d, _ = _build_boltz_input_yaml("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        for entry in d["sequences"]:
            if "protein" in entry:
                assert entry["protein"]["msa"] == "empty"

    def test_boltz2_includes_affinity_properties(self):
        _require_fixture()
        from dockstrat.models.boltz_inference import _build_boltz_input_yaml

        d, lig_chain = _build_boltz_input_yaml(
            "test", FIXTURE_PROTEIN, FIXTURE_LIGAND, model="boltz2"
        )
        assert "properties" in d
        assert len(d["properties"]) == 1
        assert d["properties"][0]["affinity"]["binder"] == lig_chain

    def test_boltz1_has_no_affinity_properties(self):
        _require_fixture()
        from dockstrat.models.boltz_inference import _build_boltz_input_yaml

        d, _ = _build_boltz_input_yaml(
            "test", FIXTURE_PROTEIN, FIXTURE_LIGAND, model="boltz1"
        )
        assert "properties" not in d

    def test_yaml_round_trips(self):
        _require_fixture()
        import yaml
        from dockstrat.models.boltz_inference import _build_boltz_input_yaml

        d, _ = _build_boltz_input_yaml("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        encoded = yaml.safe_dump(d)
        decoded = yaml.safe_load(encoded)
        assert decoded == d


class TestExtractLigandFromCif:
    def test_recovers_canonical_smiles_from_synthetic_cif(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.boltz_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        cif_path = tmp_path / "tiny.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="B")

        template_smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        template = Chem.MolFromSmiles(template_smiles)

        mol = _extract_ligand_from_cif(cif_path, "B", template_mol=template)
        assert mol is not None
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == template_smiles

    def test_raises_when_ligand_chain_missing(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.boltz_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        cif_path = tmp_path / "wrong_chain.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="Z")

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        with pytest.raises(ValueError, match="No ligand atoms"):
            _extract_ligand_from_cif(cif_path, "B", template_mol=template)

    def test_uses_label_asym_id_when_auth_asym_id_missing(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.boltz_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        cif_path = tmp_path / "label_only.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="B", use_auth_asym_id=False)

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        mol = _extract_ligand_from_cif(cif_path, "B", template_mol=template)
        assert mol is not None
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == _smiles_from_sdf(FIXTURE_LIGAND)

    def test_works_with_different_chain_ids(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.boltz_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        for chain in ("C", "D", "E"):
            cif_path = tmp_path / f"chain_{chain}.cif"
            _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id=chain)

            template_smiles = _smiles_from_sdf(FIXTURE_LIGAND)
            template = Chem.MolFromSmiles(template_smiles)
            mol = _extract_ligand_from_cif(cif_path, chain, template_mol=template)
            assert mol is not None


class TestExtractRankedLigandSdfs:
    def test_writes_rank_files_in_score_order(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.boltz_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        stem = "test_sys"
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()

        # Model 0: original coordinates, lower score
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, pred_dir / f"{stem}_model_0.cif")
        (pred_dir / f"confidence_{stem}_model_0.json").write_text(
            json.dumps({"confidence_score": 0.40})
        )
        # Model 1: translated coordinates, higher score
        _write_tiny_cif_from_sdf(
            FIXTURE_LIGAND, pred_dir / f"{stem}_model_1.cif",
            translation=(10.0, 0.0, 0.0),
        )
        (pred_dir / f"confidence_{stem}_model_1.json").write_text(
            json.dumps({"confidence_score": 0.85})
        )

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        out = tmp_path / "out"
        out.mkdir()

        _extract_ranked_ligand_sdfs(
            pred_dir, stem=stem, smiles=smiles, ligand_chain_id="B",
            out_dir=out, num_poses=2,
        )

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
        assert centroid1_x - centroid2_x > 5.0

    def test_respects_num_poses_cap(self, tmp_path):
        _require_fixture()
        from dockstrat.models.boltz_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        stem = "sys"
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        for s in range(3):
            _write_tiny_cif_from_sdf(FIXTURE_LIGAND, pred_dir / f"{stem}_model_{s}.cif")
            (pred_dir / f"confidence_{stem}_model_{s}.json").write_text(
                json.dumps({"confidence_score": 0.1 * (s + 1)})
            )

        out = tmp_path / "out"
        out.mkdir()
        _extract_ranked_ligand_sdfs(
            pred_dir, stem=stem, smiles=_smiles_from_sdf(FIXTURE_LIGAND),
            ligand_chain_id="B", out_dir=out, num_poses=2,
        )
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]

    def test_top_scored_failure_still_yields_rank1(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.boltz_inference as boltz_mod
        from dockstrat.models.boltz_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        stem = "sys"
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        for s in range(2):
            _write_tiny_cif_from_sdf(FIXTURE_LIGAND, pred_dir / f"{stem}_model_{s}.cif")
            (pred_dir / f"confidence_{stem}_model_{s}.json").write_text(
                json.dumps({"confidence_score": 0.90 - s * 0.40})
            )

        real_extract = boltz_mod._extract_ligand_from_cif
        call_count = {"n": 0}

        def flaky_extract(cif_path, ligand_chain_id, template_mol):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("simulated extraction failure")
            return real_extract(cif_path, ligand_chain_id, template_mol)

        monkeypatch.setattr(boltz_mod, "_extract_ligand_from_cif", flaky_extract)

        out = tmp_path / "out"
        out.mkdir()
        written = _extract_ranked_ligand_sdfs(
            pred_dir, stem=stem, smiles=_smiles_from_sdf(FIXTURE_LIGAND),
            ligand_chain_id="B", out_dir=out, num_poses=2,
        )

        assert written == 1
        assert (out / "rank1.sdf").exists()
        assert not (out / "rank2.sdf").exists()

    def test_attaches_affinity_properties_for_boltz2(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.boltz_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        stem = "sys"
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, pred_dir / f"{stem}_model_0.cif")
        (pred_dir / f"confidence_{stem}_model_0.json").write_text(
            json.dumps({"confidence_score": 0.80})
        )
        affinity_json = pred_dir / f"affinity_{stem}.json"
        affinity_json.write_text(json.dumps({
            "affinity_pred_value": -2.5,
            "affinity_probability_binary": 0.92,
        }))

        out = tmp_path / "out"
        out.mkdir()
        _extract_ranked_ligand_sdfs(
            pred_dir, stem=stem, smiles=_smiles_from_sdf(FIXTURE_LIGAND),
            ligand_chain_id="B", out_dir=out, num_poses=1,
            affinity_json_path=str(affinity_json),
        )

        mol = next(Chem.SDMolSupplier(str(out / "rank1.sdf")), None)
        assert mol is not None
        assert mol.HasProp("boltz2_affinity_pred_value")
        assert mol.HasProp("boltz2_affinity_probability_binary")
        assert float(mol.GetProp("boltz2_affinity_pred_value")) == pytest.approx(-2.5)


class TestRunSingle:
    def test_run_single_writes_rank_sdfs_with_mocked_subprocess(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.boltz_inference as boltz_mod

        config = {
            "boltz_binary": "/nonexistent/boltz",
            "model": "boltz1",
            "diffusion_samples": 2,
            "num_poses_to_keep": 2,
        }

        def fake_subprocess(yaml_path, out_dir, cfg):
            """Build a fake Boltz-like output tree."""
            import yaml as _yaml
            with open(yaml_path) as fh:
                payload = _yaml.safe_load(fh)

            # Determine stem from the yaml filename
            stem = Path(yaml_path).stem
            boltz_out = Path(out_dir) / f"boltz_results_{stem}" / "predictions" / stem
            boltz_out.mkdir(parents=True, exist_ok=True)

            for s in range(2):
                _write_tiny_cif_from_sdf(
                    FIXTURE_LIGAND, boltz_out / f"{stem}_model_{s}.cif",
                    translation=(s * 10.0, 0.0, 0.0),
                )
                (boltz_out / f"confidence_{stem}_model_{s}.json").write_text(
                    json.dumps({"confidence_score": 0.30 + s * 0.40})
                )

        monkeypatch.setattr(boltz_mod, "_run_boltz_subprocess", fake_subprocess)

        out = tmp_path / "out"
        out.mkdir()

        result = boltz_mod.run_single(
            protein=FIXTURE_PROTEIN,
            ligand=FIXTURE_LIGAND,
            output_dir=str(out),
            config=config,
            prefix="8gkf_test",
        )
        assert result == str(out)
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]

    def test_run_single_raises_without_boltz_binary(self, tmp_path):
        from dockstrat.models.boltz_inference import run_single

        with pytest.raises(ValueError, match="boltz_binary"):
            run_single(
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=str(tmp_path),
                config={},
            )


class TestRunDataset:
    def _make_fake_dataset(self, root, system_ids):
        import shutil
        for sid in system_ids:
            sys_dir = root / sid
            sys_dir.mkdir(parents=True)
            shutil.copy(FIXTURE_PROTEIN, sys_dir / f"{sid}_protein.pdb")
            shutil.copy(FIXTURE_LIGAND, sys_dir / f"{sid}_ligand.sdf")

    def test_run_dataset_iterates_systems_and_writes_rank_sdfs(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.boltz_inference as boltz_mod

        input_dir = tmp_path / "data"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        self._make_fake_dataset(input_dir, ["sys_a", "sys_b"])

        call_count = {"n": 0}

        def fake_subprocess(yaml_path, out_dir, cfg):
            call_count["n"] += 1
            stem = Path(yaml_path).stem
            boltz_out = Path(out_dir) / f"boltz_results_{stem}" / "predictions" / stem
            boltz_out.mkdir(parents=True, exist_ok=True)
            _write_tiny_cif_from_sdf(
                FIXTURE_LIGAND, boltz_out / f"{stem}_model_0.cif"
            )
            (boltz_out / f"confidence_{stem}_model_0.json").write_text(
                json.dumps({"confidence_score": 0.50})
            )

        monkeypatch.setattr(boltz_mod, "_run_boltz_subprocess", fake_subprocess)

        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "boltz_binary": "/nonexistent/boltz",
            "model": "boltz1",
            "diffusion_samples": 1,
            "num_poses_to_keep": 1,
            "skip_existing": True,
            "max_num_inputs": None,
            "repeat_index": 0,
            "dataset": "fake",
            "logging": {
                "level": "INFO",
                "file": "boltz_test.log",
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        }
        boltz_mod.run_dataset(config)

        assert call_count["n"] == 2
        assert (output_dir / "sys_a" / "rank1.sdf").exists()
        assert (output_dir / "sys_b" / "rank1.sdf").exists()

    def test_run_dataset_skips_already_completed_systems(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.boltz_inference as boltz_mod

        input_dir = tmp_path / "data"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        self._make_fake_dataset(input_dir, ["already_done", "needs_work"])

        done_dir = output_dir / "already_done"
        done_dir.mkdir()
        (done_dir / "rank1.sdf").write_text("dummy\n")

        call_count = {"n": 0}

        def fake_subprocess(yaml_path, out_dir, cfg):
            call_count["n"] += 1
            stem = Path(yaml_path).stem
            boltz_out = Path(out_dir) / f"boltz_results_{stem}" / "predictions" / stem
            boltz_out.mkdir(parents=True, exist_ok=True)
            _write_tiny_cif_from_sdf(
                FIXTURE_LIGAND, boltz_out / f"{stem}_model_0.cif"
            )
            (boltz_out / f"confidence_{stem}_model_0.json").write_text(
                json.dumps({"confidence_score": 0.50})
            )

        monkeypatch.setattr(boltz_mod, "_run_boltz_subprocess", fake_subprocess)

        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "boltz_binary": "/nonexistent/boltz",
            "model": "boltz1",
            "diffusion_samples": 1,
            "num_poses_to_keep": 1,
            "skip_existing": True,
            "max_num_inputs": None,
            "repeat_index": 0,
            "dataset": "fake",
            "logging": {
                "level": "INFO",
                "file": "boltz_test.log",
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        }
        boltz_mod.run_dataset(config)

        assert call_count["n"] == 1
        assert (output_dir / "needs_work" / "rank1.sdf").exists()
        assert (output_dir / "already_done" / "rank1.sdf").read_text() == "dummy\n"
