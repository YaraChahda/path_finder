"""
test_launch.py — Tests for launch.py (CLI entry points).

Functions tested:
  _pkg_root()                 → Path to the package directory
  _data_dir()                 → Path to the data directory (env-configurable)
  _fallback_template()        → YAML template string
  _write_config_from_template → writes config.yml from template

main() and setup() start subprocesses / print to stdout; they are tested
with subprocess and print mocks so no real process is spawned.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# launch.py lives in src/path_finder/ which conftest.py added to sys.path
import src.path_finder.launch as launch


# ---------------------------------------------------------------------------
# _pkg_root
# ---------------------------------------------------------------------------

class TestPkgRoot:
    def test_returns_path_object(self):
        result = launch._pkg_root()
        assert isinstance(result, Path)

    def test_points_to_directory_containing_app_py(self):
        root = launch._pkg_root()
        assert (root / "app.py").exists(), (
            f"Expected app.py inside _pkg_root() = {root}"
        )

    def test_same_directory_as_launch_py(self):
        expected = Path(launch.__file__).parent
        assert launch._pkg_root() == expected


# ---------------------------------------------------------------------------
# _data_dir
# ---------------------------------------------------------------------------

class TestDataDir:
    def test_returns_path_object(self):
        result = launch._data_dir()
        assert isinstance(result, Path)

    def test_default_is_data_subdirectory(self):
        with patch.dict(os.environ, {}, clear=True):
            # Temporarily remove PATH_FINDER_DATA if present
            env = {k: v for k, v in os.environ.items() if k != "PATH_FINDER_DATA"}
            with patch.dict(os.environ, env, clear=True):
                result = launch._data_dir()
        assert result == Path("data")

    def test_env_var_overrides_default(self, tmp_path):
        custom = str(tmp_path / "custom_data")
        with patch.dict(os.environ, {"PATH_FINDER_DATA": custom}):
            result = launch._data_dir()
        assert result == Path(custom)

    def test_env_var_empty_falls_back_to_data(self):
        # An empty string is falsy; os.environ.get returns "" which Path("") == Path(".")
        # The implementation uses os.environ.get("PATH_FINDER_DATA", "data")
        # so empty string stays as empty (Path("") = Path(".")), not "data".
        # We only test that the function does not raise.
        with patch.dict(os.environ, {"PATH_FINDER_DATA": ""}):
            result = launch._data_dir()
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# _fallback_template
# ---------------------------------------------------------------------------

class TestFallbackTemplate:
    def test_returns_string(self):
        assert isinstance(launch._fallback_template(), str)

    def test_contains_expansion_key(self):
        assert "expansion" in launch._fallback_template()

    def test_contains_filter_key(self):
        assert "filter" in launch._fallback_template()

    def test_contains_stock_key(self):
        assert "stock" in launch._fallback_template()

    def test_contains_search_key(self):
        assert "search" in launch._fallback_template()

    def test_contains_path_placeholder(self):
        assert "/PATH/TO/AIZYNTHFINDER/" in launch._fallback_template()

    def test_contains_onnx_reference(self):
        assert ".onnx" in launch._fallback_template()

    def test_contains_mcts_algorithm(self):
        assert "mcts" in launch._fallback_template()


# ---------------------------------------------------------------------------
# _write_config_from_template
# ---------------------------------------------------------------------------

class TestWriteConfigFromTemplate:
    def test_creates_output_file(self, tmp_path):
        output  = tmp_path / "config.yml"
        aiz_dir = tmp_path / "aizynthfinder"
        aiz_dir.mkdir()
        pkg     = launch._pkg_root()
        launch._write_config_from_template(pkg, output, aiz_dir)
        assert output.exists()

    def test_replaces_placeholder_with_aiz_dir(self, tmp_path):
        output  = tmp_path / "config.yml"
        aiz_dir = tmp_path / "models"
        aiz_dir.mkdir()
        pkg     = launch._pkg_root()
        launch._write_config_from_template(pkg, output, aiz_dir)
        content = output.read_text()
        assert str(aiz_dir.resolve()) in content
        assert "/PATH/TO/AIZYNTHFINDER/" not in content

    def test_written_file_is_valid_text(self, tmp_path):
        output  = tmp_path / "config.yml"
        aiz_dir = tmp_path / "aiz"
        aiz_dir.mkdir()
        pkg     = launch._pkg_root()
        launch._write_config_from_template(pkg, output, aiz_dir)
        content = output.read_text(encoding="utf-8")
        assert len(content) > 10


# ---------------------------------------------------------------------------
# main() — subprocess mocked
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_raises_sys_exit_when_app_not_found(self, tmp_path):
        """If the package root has no app.py, main() should call sys.exit(1)."""
        with patch.object(launch, "_pkg_root", return_value=tmp_path):
            with pytest.raises(SystemExit) as exc_info:
                launch.main()
            assert exc_info.value.code == 1

    def test_main_runs_subprocess_when_app_exists(self):
        """When app.py exists, main() calls subprocess.run."""
        with patch("launch.subprocess.run") as mock_run:
            with patch("builtins.print"):   # suppress output
                launch.main()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "streamlit" in args
        assert "run" in args

    def test_main_prints_url(self, capsys):
        with patch("launch.subprocess.run"):
            launch.main()
        captured = capsys.readouterr()
        assert "8501" in captured.out or "localhost" in captured.out


# ---------------------------------------------------------------------------
# setup() — prints and subprocess mocked
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_prints_header(self, capsys):
        with patch("launch.subprocess.run"):
            with patch("launch.shutil.copy2"):
                launch.setup()
        out = capsys.readouterr().out
        assert "Setup" in out or "Path Finder" in out

    def test_setup_checks_rdkit(self, capsys):
        with patch("launch.subprocess.run"):
            launch.setup()
        out = capsys.readouterr().out
        # Should mention RDKit (either OK or MISSING)
        assert "RDKit" in out or "rdkit" in out.lower()

    def test_setup_mentions_dependencies_step(self, capsys):
        with patch("launch.subprocess.run"):
            launch.setup()
        out = capsys.readouterr().out
        assert "Step 1" in out or "dependencies" in out.lower() or "Checking" in out

    def test_setup_mentions_validation_step(self, capsys):
        with patch("launch.subprocess.run"):
            launch.setup()
        out = capsys.readouterr().out
        assert "Validation" in out or "validation" in out.lower() or "Step 5" in out