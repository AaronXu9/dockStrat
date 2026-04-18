"""Unit tests for the Protenix wrapper helpers (subprocess-free)."""

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
    sdf_path, cif_path, chain_id="B0", resname="LIG", use_auth_asym_id=True,
    translation=(0.0, 0.0, 0.0),
):
    """Build a minimal mmCIF with Protenix-style chain IDs (e.g. B0)."""
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
        from dockstrat.models.protenix_inference import _smiles_from_sdf

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        assert isinstance(smiles, str)
        assert len(smiles) > 0
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == smiles

    def test_raises_value_error_for_missing_file(self, tmp_path):
        from dockstrat.models.protenix_inference import _smiles_from_sdf

        missing = tmp_path / "does_not_exist.sdf"
        with pytest.raises(ValueError, match="Cannot open SDF file"):
            _smiles_from_sdf(str(missing))


class TestBuildProtenixInputJson:
    def test_structure_is_list_with_one_element(self):
        _require_fixture()
        from dockstrat.models.protenix_inference import _build_protenix_input_json

        d, _ = _build_protenix_input_json("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        assert isinstance(d, list)
        assert len(d) == 1
        assert "name" in d[0]
        assert d[0]["name"] == "test"

    def test_uses_proteinChain_not_protein(self):
        _require_fixture()
        from dockstrat.models.protenix_inference import _build_protenix_input_json

        d, _ = _build_protenix_input_json("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        seqs = d[0]["sequences"]
        protein_entries = [s for s in seqs if "proteinChain" in s]
        assert len(protein_entries) >= 1
        for p in protein_entries:
            assert "proteinChain" in p
            assert "protein" not in p
            assert p["proteinChain"]["count"] == 1
            assert len(p["proteinChain"]["sequence"]) > 0

    def test_ligand_uses_ligand_field_for_smiles(self):
        _require_fixture()
        from dockstrat.models.protenix_inference import _build_protenix_input_json

        d, _ = _build_protenix_input_json("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        seqs = d[0]["sequences"]
        ligand_entries = [s for s in seqs if "ligand" in s]
        assert len(ligand_entries) == 1
        lig = ligand_entries[0]["ligand"]
        assert "ligand" in lig  # SMILES stored in ligand.ligand
        assert lig["count"] == 1
        assert len(lig["ligand"]) > 0

    def test_ligand_chain_id_uses_letter_zero_format(self):
        _require_fixture()
        from dockstrat.models.protenix_inference import _build_protenix_input_json

        d, lig_chain = _build_protenix_input_json("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        seqs = d[0]["sequences"]
        num_proteins = len([s for s in seqs if "proteinChain" in s])
        expected = f"{chr(ord('A') + num_proteins)}0"
        assert lig_chain == expected
        assert lig_chain.endswith("0")

    def test_json_round_trips(self):
        _require_fixture()
        from dockstrat.models.protenix_inference import _build_protenix_input_json

        d, _ = _build_protenix_input_json("test", FIXTURE_PROTEIN, FIXTURE_LIGAND)
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded == d


class TestDiscoverLigandChainId:
    def test_returns_expected_chain_when_present(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.protenix_inference import (
            _discover_ligand_chain_id, _smiles_from_sdf,
        )

        cif_path = tmp_path / "test.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="B0")

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        result = _discover_ligand_chain_id(cif_path, "B0", template)
        assert result == "B0"

    def test_fallback_scans_hetatm_chains(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.protenix_inference import (
            _discover_ligand_chain_id, _smiles_from_sdf,
        )

        # Write CIF with chain C0, but expect B0 — should fallback and find C0
        cif_path = tmp_path / "test.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="C0")

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        result = _discover_ligand_chain_id(cif_path, "B0", template)
        assert result == "C0"

    def test_raises_when_no_ligand_chain(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.protenix_inference import (
            _discover_ligand_chain_id, _smiles_from_sdf,
        )

        # Write a CIF with only ATOM (protein) records — no HETATM
        cif_path = tmp_path / "no_lig.cif"
        lines = [
            "data_test", "loop_",
            "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
            "_atom_site.label_atom_id", "_atom_site.label_comp_id",
            "_atom_site.label_asym_id", "_atom_site.label_seq_id",
            "_atom_site.auth_asym_id", "_atom_site.auth_seq_id",
            "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
            "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
            "ATOM 1 C CA ALA A0 1 A0 1 0.000 0.000 0.000 1.00 0.00",
        ]
        cif_path.write_text("\n".join(lines) + "\n")

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        with pytest.raises(ValueError, match="No ligand chain found"):
            _discover_ligand_chain_id(cif_path, "B0", template)


class TestExtractLigandFromCif:
    def test_recovers_canonical_smiles_with_b0_chain(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.protenix_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        cif_path = tmp_path / "tiny.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="B0")

        template_smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        template = Chem.MolFromSmiles(template_smiles)

        mol = _extract_ligand_from_cif(cif_path, "B0", template_mol=template)
        assert mol is not None
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == template_smiles

    def test_raises_when_chain_missing(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.protenix_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        cif_path = tmp_path / "wrong.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="Z0")

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        with pytest.raises(ValueError, match="No ligand atoms"):
            _extract_ligand_from_cif(cif_path, "B0", template_mol=template)

    def test_uses_label_asym_id_when_auth_asym_id_missing(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.protenix_inference import (
            _extract_ligand_from_cif, _smiles_from_sdf,
        )

        cif_path = tmp_path / "label_only.cif"
        _write_tiny_cif_from_sdf(FIXTURE_LIGAND, cif_path, chain_id="B0", use_auth_asym_id=False)

        template = Chem.MolFromSmiles(_smiles_from_sdf(FIXTURE_LIGAND))
        mol = _extract_ligand_from_cif(cif_path, "B0", template_mol=template)
        assert mol is not None


def _make_protenix_output(tmp_path, job_name, seed, samples, fixture_ligand, translations=None):
    """Build a fake Protenix raw output tree for testing."""
    job_dir = tmp_path / job_name
    seed_dir = job_dir / f"seed_{seed}"
    pred_dir = seed_dir / "predictions"
    pred_dir.mkdir(parents=True)

    for s in range(samples):
        score = 0.1 * (s + 1)
        if translations and s < len(translations):
            t = translations[s]
        else:
            t = (0.0, 0.0, 0.0)

        # Write confidence JSON (v2.0.0 naming: no seed in filename)
        (pred_dir / f"{job_name}_summary_confidence_sample_{s}.json").write_text(
            json.dumps({"ranking_score": score})
        )
        # Write CIF (v2.0.0 naming: no seed in filename)
        _write_tiny_cif_from_sdf(
            fixture_ligand,
            pred_dir / f"{job_name}_sample_{s}.cif",
            translation=t,
        )

    return job_dir


class TestExtractRankedLigandSdfs:
    def test_writes_rank_files_in_score_order(self, tmp_path):
        _require_fixture()
        from rdkit import Chem
        from dockstrat.models.protenix_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        job_dir = _make_protenix_output(
            tmp_path, "sys", 42, 2, FIXTURE_LIGAND,
            translations=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        )

        smiles = _smiles_from_sdf(FIXTURE_LIGAND)
        out = tmp_path / "out"
        out.mkdir()

        _extract_ranked_ligand_sdfs(
            job_dir, job_name="sys", smiles=smiles, ligand_chain_id="B0",
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

        # rank1 should be sample 1 (higher score, translated +10 along x)
        c1 = m1.GetConformer()
        c2 = m2.GetConformer()
        centroid1_x = sum(c1.GetAtomPosition(i).x for i in range(m1.GetNumAtoms())) / m1.GetNumAtoms()
        centroid2_x = sum(c2.GetAtomPosition(i).x for i in range(m2.GetNumAtoms())) / m2.GetNumAtoms()
        assert centroid1_x - centroid2_x > 5.0

    def test_respects_num_poses_cap(self, tmp_path):
        _require_fixture()
        from dockstrat.models.protenix_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        job_dir = _make_protenix_output(tmp_path, "sys", 42, 3, FIXTURE_LIGAND)

        out = tmp_path / "out"
        out.mkdir()
        _extract_ranked_ligand_sdfs(
            job_dir, job_name="sys", smiles=_smiles_from_sdf(FIXTURE_LIGAND),
            ligand_chain_id="B0", out_dir=out, num_poses=2,
        )
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]

    def test_extracts_from_cif(self, tmp_path):
        """Extraction from CIF works when no pre-extracted SDF exists."""
        _require_fixture()
        from dockstrat.models.protenix_inference import (
            _extract_ranked_ligand_sdfs, _smiles_from_sdf,
        )

        job_dir = _make_protenix_output(tmp_path, "sys", 42, 1, FIXTURE_LIGAND)

        out = tmp_path / "out"
        out.mkdir()
        written = _extract_ranked_ligand_sdfs(
            job_dir, job_name="sys", smiles=_smiles_from_sdf(FIXTURE_LIGAND),
            ligand_chain_id="B0", out_dir=out, num_poses=1,
        )
        assert written == 1
        assert (out / "rank1.sdf").exists()


class TestRunSingle:
    def test_run_single_writes_rank_sdfs_with_mocked_subprocess(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.protenix_inference as ptx_mod

        config = {
            "protenix_binary": "/nonexistent/protenix",
            "model_name": "protenix-v2",
            "num_poses_to_keep": 2,
        }

        def fake_subprocess(json_path, out_dir, cfg):
            """Build a fake Protenix raw output tree."""
            with open(json_path) as fh:
                payload = json.load(fh)
            job_name = payload[0]["name"]
            _make_protenix_output(
                Path(out_dir), job_name, 42, 2, FIXTURE_LIGAND,
                translations=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
            )

        monkeypatch.setattr(ptx_mod, "_run_protenix_subprocess", fake_subprocess)

        out = tmp_path / "out"
        out.mkdir()

        result = ptx_mod.run_single(
            protein=FIXTURE_PROTEIN,
            ligand=FIXTURE_LIGAND,
            output_dir=str(out),
            config=config,
            prefix="8gkf_test",
        )
        assert result == str(out)
        sdfs = sorted(p.name for p in out.glob("rank*.sdf"))
        assert sdfs == ["rank1.sdf", "rank2.sdf"]

    def test_run_single_raises_without_protenix_binary(self, tmp_path):
        from dockstrat.models.protenix_inference import run_single

        with pytest.raises(ValueError, match="protenix_binary"):
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
        import dockstrat.models.protenix_inference as ptx_mod

        input_dir = tmp_path / "data"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        self._make_fake_dataset(input_dir, ["sys_a", "sys_b"])

        call_count = {"n": 0}

        def fake_subprocess(json_path, out_dir, cfg):
            call_count["n"] += 1
            with open(json_path) as fh:
                payload = json.load(fh)
            job_name = payload[0]["name"]
            _make_protenix_output(Path(out_dir), job_name, 42, 1, FIXTURE_LIGAND)

        monkeypatch.setattr(ptx_mod, "_run_protenix_subprocess", fake_subprocess)

        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "protenix_binary": "/nonexistent/protenix",
            "model_name": "protenix-v2",
            "num_poses_to_keep": 1,
            "skip_existing": True,
            "max_num_inputs": None,
            "repeat_index": 0,
            "dataset": "fake",
            "logging": {
                "level": "INFO",
                "file": "protenix_test.log",
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        }
        ptx_mod.run_dataset(config)

        assert call_count["n"] == 2
        assert (output_dir / "sys_a" / "rank1.sdf").exists()
        assert (output_dir / "sys_b" / "rank1.sdf").exists()

    def test_run_dataset_skips_already_completed_systems(self, tmp_path, monkeypatch):
        _require_fixture()
        import dockstrat.models.protenix_inference as ptx_mod

        input_dir = tmp_path / "data"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        self._make_fake_dataset(input_dir, ["already_done", "needs_work"])

        done_dir = output_dir / "already_done"
        done_dir.mkdir()
        (done_dir / "rank1.sdf").write_text("dummy\n")

        call_count = {"n": 0}

        def fake_subprocess(json_path, out_dir, cfg):
            call_count["n"] += 1
            with open(json_path) as fh:
                payload = json.load(fh)
            job_name = payload[0]["name"]
            _make_protenix_output(Path(out_dir), job_name, 42, 1, FIXTURE_LIGAND)

        monkeypatch.setattr(ptx_mod, "_run_protenix_subprocess", fake_subprocess)

        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "protenix_binary": "/nonexistent/protenix",
            "model_name": "protenix-v2",
            "num_poses_to_keep": 1,
            "skip_existing": True,
            "max_num_inputs": None,
            "repeat_index": 0,
            "dataset": "fake",
            "logging": {
                "level": "INFO",
                "file": "protenix_test.log",
                "console": False,
                "log_dir": str(tmp_path / "logs"),
            },
        }
        ptx_mod.run_dataset(config)

        assert call_count["n"] == 1
        assert (output_dir / "needs_work" / "rank1.sdf").exists()
        assert (output_dir / "already_done" / "rank1.sdf").read_text() == "dummy\n"
