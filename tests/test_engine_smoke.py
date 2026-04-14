"""
Smoke tests for the dockstrat engine.

Level 1 (fast, no actual docking): tests imports, config loading, and argument validation.
Level 2 (slow, requires binaries): end-to-end dock on 8gkf fixture.
  Run with: pytest tests/test_engine_smoke.py -m slow
"""

import glob
import importlib
import os
import tempfile

import pytest

from dockstrat import dock_engine
from dockstrat.engine import (
    SUPPORTED_METHODS,
    _CONFIG_PATHS,
    _METHOD_MODULES,
    _load_config,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(PROJECT_ROOT, "data", "runsNposes", "8gkf__1__1.A__1.J")
FIXTURE_PROTEIN = os.path.join(FIXTURE_DIR, "8gkf__1__1.A__1.J_protein.pdb")
FIXTURE_LIGAND = os.path.join(FIXTURE_DIR, "8gkf__1__1.A__1.J_ligand.sdf")
CHAI_PYTHON = os.path.join(PROJECT_ROOT, "envs", "chai", "bin", "python")


# ── Level 1: fast, no docking ────────────────────────────────────────────────

class TestImports:
    def test_dock_engine_importable(self):
        from dockstrat import dock_engine
        assert callable(dock_engine)

    def test_register_resolvers_importable(self):
        from dockstrat import register_custom_omegaconf_resolvers
        assert callable(register_custom_omegaconf_resolvers)

    def test_no_posebench_imports(self):
        """Ensure no dockstrat module imports from the old posebench package."""
        import ast
        import glob
        pkg_dir = os.path.join(PROJECT_ROOT, "dockstrat")
        for py_file in glob.glob(os.path.join(pkg_dir, "**", "*.py"), recursive=True):
            with open(py_file) as f:
                source = f.read()
            tree = ast.parse(source, filename=py_file)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, "module", "") or ""
                    names = [alias.name for alias in getattr(node, "names", [])]
                    all_names = [module] + names
                    for name in all_names:
                        assert not (name or "").startswith("posebench"), (
                            f"Found 'posebench' import in {py_file}:{node.lineno}: {name}"
                        )


class TestConfigLoading:
    def test_all_config_files_exist(self):
        for method, path in _CONFIG_PATHS.items():
            assert os.path.exists(path), f"Config missing for {method}: {path}"

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_config_loads(self, method):
        cfg = _load_config(method, {})
        assert isinstance(cfg, dict)

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_config_override(self, method):
        cfg = _load_config(method, {"skip_existing": False})
        assert cfg.get("skip_existing") is False


class TestMethodModules:
    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_module_imports(self, method):
        mod = importlib.import_module(_METHOD_MODULES[method])
        assert hasattr(mod, "run_single"), f"{method} missing run_single"
        assert hasattr(mod, "run_dataset"), f"{method} missing run_dataset"


class TestArgumentValidation:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            dock_engine("not_a_method", protein="a.pdb", ligand="b.sdf")

    def test_missing_protein_ligand_raises(self):
        with pytest.raises(ValueError):
            dock_engine("vina")

    def test_conflicting_args_raises(self):
        with pytest.raises(ValueError):
            dock_engine("vina", dataset="runsNposes", protein="a.pdb", ligand="b.sdf")


# ── Level 2: slow, requires binaries ─────────────────────────────────────────

@pytest.mark.slow
class TestEndToEnd:
    """Full docking runs using the 8gkf fixture (smallest system in runsNposes)."""

    def setup_method(self):
        if not os.path.exists(FIXTURE_PROTEIN):
            pytest.skip(f"Fixture not found: {FIXTURE_PROTEIN}")
        if not os.path.exists(FIXTURE_LIGAND):
            pytest.skip(f"Fixture not found: {FIXTURE_LIGAND}")

    def test_vina_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dock_engine(
                "vina",
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=tmp,
                prefix="8gkf_test",
                top_n=3,
            )
            assert result is not None
            sdfs = [f for f in os.listdir(result) if f.endswith(".sdf")]
            assert len(sdfs) >= 1, f"No SDF poses found in {result}"

    def test_gnina_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dock_engine(
                "gnina",
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=tmp,
                prefix="8gkf_test",
                top_n=3,
            )
            assert result is not None
            sdfs = [f for f in os.listdir(result) if f.endswith(".sdf")]
            assert len(sdfs) >= 1, f"No SDF poses found in {result}"

    def test_unidock2_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dock_engine(
                "unidock2",
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=tmp,
                prefix="8gkf_test",
                num_poses=3,
            )
            assert result is not None
            sdfs = [f for f in os.listdir(result) if f.endswith(".sdf")]
            assert len(sdfs) >= 1, f"No SDF poses found in {result}"

    def test_chai_single(self):
        if not os.path.exists(CHAI_PYTHON):
            pytest.skip(f"Chai Python not found: {CHAI_PYTHON}")
        with tempfile.TemporaryDirectory() as tmp:
            result = dock_engine(
                "chai",
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=tmp,
                prefix="8gkf_test",
                python_exec_path=CHAI_PYTHON,
            )
            assert result is not None
            sdfs = glob.glob(os.path.join(result, "rank*.sdf"))
            assert len(sdfs) >= 1, f"No rank*.sdf files found in {result}"

    def test_dynamicbind_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dock_engine(
                "dynamicbind",
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=tmp,
                prefix="8gkf_test",
                samples_per_complex=1,
                savings_per_complex=1,
            )
            assert result is not None
            sdfs = glob.glob(os.path.join(result, "rank*.sdf")) + \
                   glob.glob(os.path.join(result, "rank*_ligand_lddt*.sdf"))
            assert len(sdfs) >= 1, f"No rank*.sdf files found in {result}"

    def test_surfdock_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dock_engine(
                "surfdock",
                protein=FIXTURE_PROTEIN,
                ligand=FIXTURE_LIGAND,
                output_dir=tmp,
                prefix="8gkf_test",
                num_poses=1,
                samples_per_complex=1,
                batch_size=1,
            )
            assert result is not None
            sdfs = glob.glob(os.path.join(result, "rank*.sdf"))
            assert len(sdfs) >= 1, f"No rank*.sdf files found in {result}"

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
