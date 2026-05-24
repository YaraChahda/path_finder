"""Tests for molecule_rendering.py."""

import sys
import os
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import pytest
from path_finder.molecule_rendering import (
    mol_png,
    mol_b64_or_text_svg,
    fallback_data_uri,
    is_trivial_smiles,
    MODULE_OK,
)

RDKIT_SKIP = pytest.mark.skipif(not MODULE_OK, reason="RDKit not available")

VALID_SMILES = "c1ccccc1"           # benzene
COMPLEX_SMILES = "OC1C=C[C@@]23c4cc(OC)ccc4CN(C)C[C@@H]2[C@@H]1O3"  # galanthamine
INVALID_SMILES = "not_a_smiles!!!"
SINGLE_ATOM = "[Na+]"
DIATOMIC = "Cl"


# MODULE_OK

def test_module_ok_is_bool():
    assert isinstance(MODULE_OK, bool)


# mol_png

def test_mol_png_returns_none_for_empty_smiles():
    assert mol_png("") is None


def test_mol_png_returns_none_for_invalid_smiles():
    assert mol_png(INVALID_SMILES) is None


@RDKIT_SKIP
def test_mol_png_returns_bytes_for_valid_smiles():
    result = mol_png(VALID_SMILES, 200, 150)
    assert isinstance(result, bytes)
    assert len(result) > 0


@RDKIT_SKIP
def test_mol_png_starts_with_png_header():
    result = mol_png(VALID_SMILES, 100, 100)
    assert result is not None
    assert result[:4] == b"\x89PNG"


@RDKIT_SKIP
def test_mol_png_works_with_complex_molecule():
    result = mol_png(COMPLEX_SMILES, 300, 200)
    assert isinstance(result, bytes)


@RDKIT_SKIP
def test_mol_png_default_size():
    result = mol_png(VALID_SMILES)
    assert result is not None


# mol_b64_or_text_svg

def test_mol_b64_returns_string():
    result = mol_b64_or_text_svg("", 100, 80)
    assert isinstance(result, str)


def test_mol_b64_returns_data_uri_for_empty():
    result = mol_b64_or_text_svg("", 100, 80)
    assert result.startswith("data:image/png;base64,")


def test_mol_b64_returns_data_uri_for_invalid():
    result = mol_b64_or_text_svg(INVALID_SMILES, 100, 80)
    assert result.startswith("data:image/png;base64,")


@RDKIT_SKIP
def test_mol_b64_returns_data_uri_for_valid():
    result = mol_b64_or_text_svg(VALID_SMILES, 100, 80)
    assert result.startswith("data:image/png;base64,")


@RDKIT_SKIP
def test_mol_b64_base64_is_decodable():
    result = mol_b64_or_text_svg(VALID_SMILES, 100, 80)
    b64_part = result.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert len(decoded) > 0


@RDKIT_SKIP
def test_mol_b64_single_atom_no_crash():
    result = mol_b64_or_text_svg(SINGLE_ATOM, 80, 60)
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,")


# fallback_data_uri

def test_fallback_returns_string():
    result = fallback_data_uri("test", 100, 80)
    assert isinstance(result, str)


def test_fallback_starts_with_data_uri():
    result = fallback_data_uri("test", 100, 80)
    assert result.startswith("data:image/png;base64,")


def test_fallback_base64_decodable():
    result = fallback_data_uri("ABC", 50, 40)
    b64_part = result.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert len(decoded) > 0


def test_fallback_long_text_truncated_gracefully():
    long_text = "A" * 100
    result = fallback_data_uri(long_text, 100, 80)
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,")


def test_fallback_empty_text():
    result = fallback_data_uri("", 100, 80)
    assert isinstance(result, str)


def test_fallback_various_sizes():
    for w, h in [(50, 40), (200, 150), (400, 300)]:
        result = fallback_data_uri("test", w, h)
        assert result.startswith("data:image/png;base64,")


# is_trivial_smiles

def test_is_trivial_empty_string():
    assert is_trivial_smiles("") is True


def test_is_trivial_single_atom():
    assert is_trivial_smiles("[Na+]") is True


def test_is_trivial_invalid_smiles():
    assert is_trivial_smiles(INVALID_SMILES) is True


@RDKIT_SKIP
def test_is_trivial_diatomic():
    assert is_trivial_smiles("Cl") is True


@RDKIT_SKIP
def test_is_trivial_benzene_is_not_trivial():
    assert is_trivial_smiles(VALID_SMILES) is False


@RDKIT_SKIP
def test_is_trivial_complex_molecule_is_not_trivial():
    assert is_trivial_smiles(COMPLEX_SMILES) is False


@RDKIT_SKIP
def test_is_trivial_water_is_trivial():
    assert is_trivial_smiles("O") is True


@RDKIT_SKIP
def test_is_trivial_two_atom_molecule():
    assert is_trivial_smiles("Cl") is True


@RDKIT_SKIP
def test_is_trivial_salt_all_small_fragments():
    assert is_trivial_smiles("[Na+].[Cl-]") is True


@RDKIT_SKIP
def test_is_trivial_returns_bool():
    result = is_trivial_smiles(VALID_SMILES)
    assert isinstance(result, bool)

def test_fallback_data_uri_pil_failure():
    """If PIL raises inside fallback_data_uri the 1×1 PNG is returned."""
    import sys
    from unittest.mock import patch
    from path_finder import molecule_rendering as mr
    with patch.dict(sys.modules, {"PIL":None,"PIL.Image":None,"PIL.ImageDraw":None}):
        result = mr.fallback_data_uri("test", 100, 80)
    assert result.startswith("data:image/png;base64,")
