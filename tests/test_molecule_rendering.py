"""
Tests for path_finder.molecule_rendering

All four public functions are exercised:
  mol_png               — high-res PNG bytes
  mol_b64_or_text_svg   — base64 PNG data-URI
  fallback_data_uri     — grey placeholder data-URI
  is_trivial_smiles     — small-molecule detector

RDKit and Pillow are both expected; the whole module is skipped when
either is missing.
"""

import base64
import pytest

rdkit = pytest.importorskip("rdkit",  reason="RDKit not installed")
PIL   = pytest.importorskip("PIL",    reason="Pillow not installed")

from path_finder.molecule_rendering import (  # noqa: E402
    mol_png,
    mol_b64_or_text_svg,
    fallback_data_uri,
    is_trivial_smiles,
    MODULE_OK,
)

BENZENE = "c1ccccc1"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SINGLE_ATOM = "[Na+]"
DIATOMIC = "Cl"
BAD_SMILES = "NOT_A_MOLECULE"


# ===========================================================================
# mol_png
# ===========================================================================

class TestMolPng:
    def test_returns_bytes_for_valid_smiles(self):
        result = mol_png(BENZENE)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_returns_none_for_empty_string(self):
        assert mol_png("") is None

    def test_returns_none_for_invalid_smiles(self):
        assert mol_png(BAD_SMILES) is None

    def test_returns_bytes_for_aspirin(self):
        result = mol_png(ASPIRIN)
        assert result is not None
        assert isinstance(result, bytes)

    def test_output_is_valid_png(self):
        result = mol_png(ETHANOL)
        assert result is not None
        # PNG magic bytes: \x89PNG
        assert result[:4] == b"\x89PNG"

    def test_custom_size(self):
        result_small = mol_png(BENZENE, w=100, h=100)
        result_large = mol_png(BENZENE, w=400, h=400)
        # Larger canvas → more bytes
        assert result_large is not None
        assert result_small is not None
        assert len(result_large) >= len(result_small)


# ===========================================================================
# mol_b64_or_text_svg
# ===========================================================================

class TestMolB64OrTextSvg:
    def test_returns_string(self):
        result = mol_b64_or_text_svg(BENZENE, 200, 150)
        assert isinstance(result, str)

    def test_valid_smiles_returns_data_uri(self):
        result = mol_b64_or_text_svg(ASPIRIN, 200, 150)
        assert result.startswith("data:image/png;base64,")

    def test_invalid_smiles_returns_fallback(self):
        result = mol_b64_or_text_svg(BAD_SMILES, 200, 150)
        assert result.startswith("data:image/png;base64,")

    def test_empty_smiles_returns_fallback(self):
        result = mol_b64_or_text_svg("", 100, 100)
        assert result.startswith("data:image/png;base64,")

    def test_base64_portion_is_valid(self):
        result = mol_b64_or_text_svg(ETHANOL, 100, 80)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:4] == b"\x89PNG"

    def test_single_atom_handled_gracefully(self):
        result = mol_b64_or_text_svg(SINGLE_ATOM, 80, 60)
        assert result.startswith("data:image/png;base64,")


# ===========================================================================
# fallback_data_uri
# ===========================================================================

class TestFallbackDataUri:
    def test_returns_string(self):
        result = fallback_data_uri("TEST", 100, 80)
        assert isinstance(result, str)

    def test_returns_data_uri(self):
        result = fallback_data_uri("ABC", 200, 150)
        assert result.startswith("data:image/png;base64,")

    def test_base64_decodes_to_valid_png(self):
        result = fallback_data_uri("hello", 120, 60)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:4] == b"\x89PNG"

    def test_long_text_truncated(self):
        # Should not raise even with a very long string
        result = fallback_data_uri("A" * 100, 100, 50)
        assert result.startswith("data:image/png;base64,")

    def test_empty_text(self):
        result = fallback_data_uri("", 100, 50)
        assert result.startswith("data:image/png;base64,")

    def test_different_sizes_produce_different_outputs(self):
        r_small = fallback_data_uri("X", 50, 50)
        r_large = fallback_data_uri("X", 400, 400)
        # Different image sizes → different base64 blobs
        assert r_small != r_large


# ===========================================================================
# is_trivial_smiles
# ===========================================================================

class TestIsTrivialSmiles:
    def test_single_atom_is_trivial(self):
        assert is_trivial_smiles(SINGLE_ATOM) is True

    def test_diatomic_is_trivial(self):
        assert is_trivial_smiles("Cl") is True

    def test_dihydrogen_is_trivial(self):
        assert is_trivial_smiles("[H][H]") is True

    def test_benzene_is_not_trivial(self):
        assert is_trivial_smiles(BENZENE) is False

    def test_aspirin_is_not_trivial(self):
        assert is_trivial_smiles(ASPIRIN) is False

    def test_ethanol_is_not_trivial(self):
        assert is_trivial_smiles(ETHANOL) is False

    def test_empty_string_is_trivial(self):
        assert is_trivial_smiles("") is True

    def test_invalid_smiles_is_trivial(self):
        assert is_trivial_smiles(BAD_SMILES) is True

    def test_salt_with_small_fragments_is_trivial(self):
        # "[Na+].[Cl-]" — both fragments have 1 heavy atom
        assert is_trivial_smiles("[Na+].[Cl-]") is True

    def test_salt_with_large_fragment_not_trivial(self):
        # Sodium acetate: [Na+].CC(=O)[O-]
        assert is_trivial_smiles("[Na+].CC(=O)[O-]") is False