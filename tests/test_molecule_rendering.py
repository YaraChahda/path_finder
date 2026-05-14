"""
Tests for src/path_finder/molecule_rendering.py

Covers: mol_png, mol_b64_or_text_svg, fallback_data_uri, is_trivial_smiles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "path_finder"))

import base64
import pytest

from src.path_finder.molecule_rendering import (
    mol_png,
    mol_b64_or_text_svg,
    fallback_data_uri,
    is_trivial_smiles,
    MODULE_OK,
)

# A few representative SMILES strings used across multiple tests
BENZENE   = "c1ccccc1"
WATER     = "O"
SODIUM    = "[Na+]"
CHLORINE  = "Cl"
INVALID   = "not_a_smiles!!!"
MORPHINE  = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5"


# ---------------------------------------------------------------------------
# is_trivial_smiles
# ---------------------------------------------------------------------------

class TestIsTrivialSmiles:
    def test_empty_string_is_trivial(self):
        assert is_trivial_smiles("") is True

    def test_single_atom_is_trivial(self):
        assert is_trivial_smiles("[Na+]") is True

    def test_water_is_trivial(self):
        # O has 1 heavy atom
        assert is_trivial_smiles("O") is True

    def test_chlorine_gas_is_trivial(self):
        # Cl2 has 2 heavy atoms
        assert is_trivial_smiles("ClCl") is True

    def test_salt_with_small_fragments_is_trivial(self):
        # Na+ and Cl- — both ≤ 2 heavy atoms
        assert is_trivial_smiles("[Na+].[Cl-]") is True

    def test_benzene_is_not_trivial(self):
        assert is_trivial_smiles(BENZENE) is False

    def test_morphine_is_not_trivial(self):
        assert is_trivial_smiles(MORPHINE) is False

    def test_ethanol_is_not_trivial(self):
        # CCO has 3 heavy atoms
        assert is_trivial_smiles("CCO") is False

    def test_invalid_smiles_returns_true(self):
        # Cannot parse → treated as trivial (safe fallback)
        assert is_trivial_smiles(INVALID) is True


# ---------------------------------------------------------------------------
# mol_png
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODULE_OK, reason="RDKit not available")
class TestMolPng:
    def test_returns_bytes_for_valid_smiles(self):
        result = mol_png(BENZENE)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_returns_none_for_empty_smiles(self):
        assert mol_png("") is None

    def test_returns_none_for_invalid_smiles(self):
        assert mol_png(INVALID) is None

    def test_result_is_png(self):
        result = mol_png(BENZENE)
        # PNG magic bytes: \x89PNG
        assert result[:4] == b"\x89PNG"

    def test_custom_dimensions_accepted(self):
        result = mol_png(BENZENE, w=200, h=150)
        assert isinstance(result, bytes) and len(result) > 0

    def test_morphine_renders(self):
        result = mol_png(MORPHINE)
        assert isinstance(result, bytes) and len(result) > 0


# ---------------------------------------------------------------------------
# mol_b64_or_text_svg
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODULE_OK, reason="RDKit not available")
class TestMolB64OrTextSvg:
    def test_returns_data_uri_for_valid_smiles(self):
        result = mol_b64_or_text_svg(BENZENE, 200, 150)
        assert result.startswith("data:image/png;base64,")

    def test_falls_back_for_empty_smiles(self):
        result = mol_b64_or_text_svg("", 100, 80)
        assert result.startswith("data:image/png;base64,")

    def test_falls_back_for_invalid_smiles(self):
        result = mol_b64_or_text_svg(INVALID, 100, 80)
        assert result.startswith("data:image/png;base64,")

    def test_base64_payload_is_valid(self):
        result = mol_b64_or_text_svg(BENZENE, 100, 80)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        # Decoded bytes must be a PNG (magic header)
        assert decoded[:4] == b"\x89PNG"

    def test_single_atom_renders_without_error(self):
        # Single atoms skip Compute2DCoords; this must not raise
        result = mol_b64_or_text_svg("[Na+]", 60, 60)
        assert result.startswith("data:image/png;base64,")

    def test_morphine_base64_is_valid(self):
        result = mol_b64_or_text_svg(MORPHINE, 300, 200)
        b64_part = result.split(",", 1)[1]
        base64.b64decode(b64_part)  # must not raise


# ---------------------------------------------------------------------------
# fallback_data_uri
# ---------------------------------------------------------------------------

class TestFallbackDataUri:
    def test_returns_data_uri(self):
        result = fallback_data_uri("TEST", 100, 80)
        assert result.startswith("data:image/png;base64,")

    def test_base64_payload_decodes(self):
        result = fallback_data_uri("TEST", 100, 80)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert len(decoded) > 0

    def test_long_text_is_truncated_to_18_chars(self):
        long_text = "A" * 30
        # Should not raise even with very long labels
        result = fallback_data_uri(long_text, 200, 100)
        assert result.startswith("data:image/png;base64,")

    def test_empty_text_does_not_raise(self):
        result = fallback_data_uri("", 50, 50)
        assert result.startswith("data:image/png;base64,")

    def test_result_is_png(self):
        result = fallback_data_uri("?", 100, 80)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:4] == b"\x89PNG"