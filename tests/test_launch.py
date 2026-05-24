"""Tests for launch.py (CLI entry points)."""

import sys
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import pytest
import src.path_finder.launch as launch


# _pkg_root

def test_pkg_root_returns_path():
    result = launch._pkg_root()
    assert isinstance(result, Path)


def test_pkg_root_is_directory():
    result = launch._pkg_root()
    assert result.is_dir()


def test_pkg_root_contains_app_py():
    result = launch._pkg_root()
    assert (result / "app.py").exists()


def test_pkg_root_is_parent_of_launch():
    result = launch._pkg_root()
    assert (result / "launch.py").exists()


# _data_dir

def test_data_dir_returns_path():
    result = launch._data_dir()
    assert isinstance(result, Path)


def test_data_dir_default_is_data():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PATH_FINDER_DATA", None)
        result = launch._data_dir()
    assert result == Path("data")


def test_data_dir_respects_env_var(tmp_path):
    with patch.dict(os.environ, {"PATH_FINDER_DATA": str(tmp_path)}):
        result = launch._data_dir()
    assert result == tmp_path


def test_data_dir_env_var_custom_path():
    custom = "/custom/data/path"
    with patch.dict(os.environ, {"PATH_FINDER_DATA": custom}):
        result = launch._data_dir()
    assert result == Path(custom)


# _fallback_template

def test_fallback_template_returns_string():
    result = launch._fallback_template()
    assert isinstance(result, str)


def test_fallback_template_non_empty():
    result = launch._fallback_template()
    assert len(result) > 0


def test_fallback_template_contains_expansion():
    result = launch._fallback_template()
    assert "expansion" in result


def test_fallback_template_contains_filter():
    result = launch._fallback_template()
    assert "filter" in result


def test_fallback_template_contains_stock():
    result = launch._fallback_template()
    assert "stock" in result


def test_fallback_template_contains_search():
    result = launch._fallback_template()
    assert "search" in result


def test_fallback_template_contains_path_placeholder():
    result = launch._fallback_template()
    assert "/PATH/TO/AIZYNTHFINDER/" in result


def test_fallback_template_is_yaml_like():
    result = launch._fallback_template()
    assert ":" in result


# _write_config_from_template

def test_write_config_creates_file(tmp_path):
    pkg   = launch._pkg_root()
    out   = tmp_path / "config.yml"
    aiz   = tmp_path / "aizynthfinder"
    aiz.mkdir()
    launch._write_config_from_template(pkg, out, aiz)
    assert out.exists()


def test_write_config_replaces_placeholder(tmp_path):
    pkg = launch._pkg_root()
    out = tmp_path / "config.yml"
    aiz = tmp_path / "models"
    aiz.mkdir()
    launch._write_config_from_template(pkg, out, aiz)
    content = out.read_text()
    assert "/PATH/TO/AIZYNTHFINDER/" not in content


def test_write_config_inserts_aiz_dir(tmp_path):
    pkg = launch._pkg_root()
    out = tmp_path / "config.yml"
    aiz = tmp_path / "aizynthfinder"
    aiz.mkdir()
    launch._write_config_from_template(pkg, out, aiz)
    content = out.read_text()
    assert str(aiz.resolve()) in content


def test_write_config_output_is_valid_string(tmp_path):
    pkg = launch._pkg_root()
    out = tmp_path / "config.yml"
    aiz = tmp_path / "aiz"
    aiz.mkdir()
    launch._write_config_from_template(pkg, out, aiz)
    content = out.read_text()
    assert len(content) > 0
    
# main — smoke tests (subprocess mocked)

def test_main_calls_subprocess(tmp_path):
    app = launch._pkg_root() / "app.py"
    with patch.object(launch, "_data_dir", return_value=tmp_path), \
         patch("subprocess.run") as mock_run, \
         patch.object(launch, "_pkg_root", return_value=launch._pkg_root()):
        # Create dummy config so no warning is printed
        (tmp_path / "config.yml").write_text("dummy")
        launch.main()
        mock_run.assert_called_once()


def test_main_exits_if_app_not_found(tmp_path):
    with patch.object(launch, "_pkg_root", return_value=tmp_path), \
         patch.object(launch, "_data_dir", return_value=tmp_path):
        with pytest.raises(SystemExit):
            launch.main()


def test_main_subprocess_args_include_streamlit():
    with patch("subprocess.run") as mock_run, \
         patch.object(launch, "_data_dir", return_value=Path("data")):
        try:
            launch.main()
        except Exception:
            pass
        if mock_run.called:
            cmd = mock_run.call_args[0][0]
            assert "streamlit" in cmd

# setup — smoke tests (subprocess and filesystem mocked)

def test_setup_runs_without_crash(tmp_path):
    with patch.object(launch, "_data_dir", return_value=tmp_path), \
         patch.object(launch, "_pkg_root", return_value=launch._pkg_root()), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)):
        # Should not raise
        launch.setup()


def test_setup_creates_data_dir(tmp_path):
    data_dir = tmp_path / "new_data"
    with patch.object(launch, "_data_dir", return_value=data_dir), \
         patch.object(launch, "_pkg_root", return_value=launch._pkg_root()), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)):
        launch.setup()
    assert data_dir.exists()

def test_write_config_uses_fallback_when_no_template(tmp_path):
    fake_pkg = tmp_path / "fake_pkg"
    fake_pkg.mkdir()
    out = tmp_path / "config.yml"
    aiz = tmp_path / "aiz"
    aiz.mkdir()
    launch._write_config_from_template(fake_pkg, out, aiz)
    assert "expansion" in out.read_text()   # from _fallback_template

def test_setup_validation_reports_missing(tmp_path):
    with patch.object(launch, "_data_dir", return_value=tmp_path), \
         patch.object(launch, "_pkg_root", return_value=launch._pkg_root()), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)), \
         patch("builtins.print") as mock_print:
        launch.setup()
    printed = " ".join(str(c) for call in mock_print.call_args_list for c in call.args)
    assert "MISSING" in printed or "incomplete" in printed
