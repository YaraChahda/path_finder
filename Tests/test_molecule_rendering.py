"""
test_molecule_rendering.py — Tests for molecule_rendering.py.

All tests are gated on MODULE_OK (RDKit availability).  The fallback helpers
(fallback_data_uri, is_trivial_smiles with empty input) are always tested
because they contain graceful-degradation paths that must work even when
RDKit is absent.
"""

import base64
import pytest

import src.path_finder.molecule_rendering as mr

rdkit_available = pytest.mark.skipif(
    not mr.MODULE_OK, reason="RDKit not installed"
)


# ---------------------------------------------------------------------------
# is_trivial_smiles
# ---------------------------------------------------------------------------

class TestIsTrivialSmiles:
    def test_empty_string_is_trivial(self):
        assert mr.is_trivial_smiles("") is True

    def test_none_like_empty_is_trivial(self):
        # Falsy strings are trivial regardless of RDKit
        assert mr.is_trivial_smiles("") is True

    @rdkit_available
    def test_single_atom_is_trivial(self):
        assert mr.is_trivial_smiles("[Na+]") is True

    @rdkit_available
    def test_diatomic_is_trivial(self):
        assert mr.is_trivial_smiles("Cl") is True  # chlorine single atom

    @rdkit_available
    def test_two_heavy_atoms_is_trivial(self):
        # HCl — two atoms, should be trivial
        assert mr.is_trivial_smiles("ClCl") is True

    @rdkit_available
    def test_benzene_is_not_trivial(self):
        assert mr.is_trivial_smiles("c1ccccc1") is False

    @rdkit_available
    def test_complex_molecule_is_not_trivial(self):
        caffeine = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
        assert mr.is_trivial_smiles(caffeine) is False

    @rdkit_available
    def test_salt_all_trivial_fragments_is_trivial(self):
        # Two trivial fragments separated by '.'
        assert mr.is_trivial_smiles("[Na+].[Cl-]") is True

    @rdkit_available
    def test_invalid_smiles_returns_true(self):
        # Unparseable SMILES → treated as trivial (safe default)
        assert mr.is_trivial_smiles("not_a_smiles!!!") is True


# ---------------------------------------------------------------------------
# fallback_data_uri
# ---------------------------------------------------------------------------

class TestFallbackDataUri:
    def test_returns_string(self):
        result = mr.fallback_data_uri("test", 100, 80)
        assert isinstance(result, str)

    def test_is_data_uri(self):
        result = mr.fallback_data_uri("test", 100, 80)
        assert result.startswith("data:image/png;base64,")

    def test_base64_is_valid(self):
        result = mr.fallback_data_uri("ABC", 50, 50)
        b64_part = result.split(",", 1)[1]
        # Should not raise
        decoded = base64.b64decode(b64_part)
        assert len(decoded) > 0

    def test_long_text_is_truncated_gracefully(self):
        long_text = "A" * 100
        result = mr.fallback_data_uri(long_text, 100, 80)
        assert result.startswith("data:image/png;base64,")

    def test_empty_text(self):
        result = mr.fallback_data_uri("", 100, 80)
        assert result.startswith("data:")

    def test_small_dimensions(self):
        result = mr.fallback_data_uri("X", 10, 10)
        assert result.startswith("data:")


# ---------------------------------------------------------------------------
# mol_png
# ---------------------------------------------------------------------------

class TestMolPng:
    def test_empty_smiles_returns_none(self):
        assert mr.mol_png("") is None

    @rdkit_available
    def test_invalid_smiles_returns_none(self):
        assert mr.mol_png("not!valid") is None

    @rdkit_available
    def test_valid_smiles_returns_bytes(self):
        result = mr.mol_png("c1ccccc1", 100, 80)
        assert isinstance(result, bytes)
        assert len(result) > 0

    @rdkit_available
    def test_png_bytes_start_with_png_header(self):
        result = mr.mol_png("c1ccccc1", 100, 80)
        # PNG magic bytes: 0x89 0x50 0x4E 0x47
        assert result[:4] == b"\x89PNG"

    @rdkit_available
    def test_custom_dimensions_accepted(self):
        result = mr.mol_png("CCO", 200, 150)
        assert result is not None

    @rdkit_available
    def test_complex_molecule(self):
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        result = mr.mol_png(aspirin, 300, 200)
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# mol_b64_or_text_svg
# ---------------------------------------------------------------------------

class TestMolB64OrTextSvg:
    def test_empty_smiles_returns_fallback_uri(self):
        result = mr.mol_b64_or_text_svg("", 100, 80)
        assert result.startswith("data:")

    @rdkit_available
    def test_invalid_smiles_returns_fallback_uri(self):
        result = mr.mol_b64_or_text_svg("not!valid", 100, 80)
        assert result.startswith("data:")

    @rdkit_available
    def test_valid_smiles_returns_base64_png(self):
        result = mr.mol_b64_or_text_svg("c1ccccc1", 100, 80)
        assert result.startswith("data:image/png;base64,")

    @rdkit_available
    def test_base64_payload_is_valid(self):
        result = mr.mol_b64_or_text_svg("CCO", 100, 80)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:4] == b"\x89PNG"

    @rdkit_available
    def test_single_atom_molecule(self):
        # Single-atom molecules skip Compute2DCoords; must not crash
        result = mr.mol_b64_or_text_svg("[Na+]", 60, 50)
        assert result.startswith("data:")

    @rdkit_available
    def test_result_is_string(self):
        result = mr.mol_b64_or_text_svg("CC", 80, 60)
        assert isinstance(result, str)