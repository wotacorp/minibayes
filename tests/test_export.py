"""Tests for save/load functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from minibayes.results import InferenceResult
from minibayes.utils.export import load_npz, save_npz, to_json


def _make_result() -> InferenceResult:
    """Create a sample InferenceResult for testing."""
    rng = np.random.default_rng(42)
    # Always (num_chains, num_samples) shape
    samples: dict[str, NDArray[np.float64]] = {
        "alpha": rng.standard_normal((1, 100)),
        "beta": rng.standard_normal((1, 100)),
    }
    samples_unconstrained: dict[str, NDArray[np.float64]] = {
        "alpha": samples["alpha"].copy(),
        "beta": samples["beta"].copy(),
    }
    return InferenceResult(
        samples=samples,
        samples_unconstrained=samples_unconstrained,
        acceptance_rate=np.array([0.25]),
        num_samples=100,
        num_warmup=50,
        num_chains=1,
        sampler="adaptive_mh",
        elapsed_time=1.5,
    )


def _make_multichain_result() -> InferenceResult:
    """Create a multi-chain InferenceResult for testing."""
    rng = np.random.default_rng(42)
    samples: dict[str, NDArray[np.float64]] = {
        "alpha": rng.standard_normal((4, 100)),
        "beta": rng.standard_normal((4, 100)),
    }
    samples_unconstrained: dict[str, NDArray[np.float64]] = {
        "alpha": samples["alpha"].copy(),
        "beta": samples["beta"].copy(),
    }
    return InferenceResult(
        samples=samples,
        samples_unconstrained=samples_unconstrained,
        acceptance_rate=np.array([0.23, 0.25, 0.24, 0.22]),
        num_samples=100,
        num_warmup=50,
        num_chains=4,
        sampler="adaptive_mh",
        elapsed_time=5.0,
    )


class TestExport:
    """Tests for export functions."""

    def test_save_load_npz_roundtrip(self) -> None:
        """Test save/load NPZ preserves data."""
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.npz"
            save_npz(result, str(path))

            loaded = load_npz(str(path))

            assert loaded.num_samples == result.num_samples
            assert loaded.num_warmup == result.num_warmup
            assert loaded.num_chains == result.num_chains
            assert loaded.sampler == result.sampler
            assert loaded.elapsed_time == result.elapsed_time
            np.testing.assert_array_almost_equal(loaded.acceptance_rate, result.acceptance_rate)
            assert set(loaded.samples.keys()) == set(result.samples.keys())
            for k in result.samples:
                np.testing.assert_array_almost_equal(loaded.samples[k], result.samples[k])

    def test_save_load_npz_multichain(self) -> None:
        """Test save/load NPZ preserves multi-chain data."""
        result = _make_multichain_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.npz"
            save_npz(result, str(path))

            loaded = load_npz(str(path))

            assert loaded.num_chains == 4
            assert loaded.samples["alpha"].shape == (4, 100)
            np.testing.assert_array_almost_equal(loaded.acceptance_rate, result.acceptance_rate)

    def test_to_dict(self) -> None:
        """Test to_dict returns serializable dict."""
        result = _make_result()
        d = to_json(result)

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Check structure
        assert "samples" in d
        assert "samples_unconstrained" in d
        assert "acceptance_rate" in d
        assert "num_samples" in d
        assert d["num_samples"] == 100
        assert d["sampler"] == "adaptive_mh"

    def test_inference_result_save_method(self) -> None:
        """Test InferenceResult.save() method."""
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test NPZ format
            npz_path = Path(tmpdir) / "result.npz"
            result.save(str(npz_path), format="npz")
            assert npz_path.exists()

            # Test JSON format
            json_path = Path(tmpdir) / "result.json"
            result.save(str(json_path), format="json")
            assert json_path.exists()

            # Verify JSON content
            with open(json_path) as f:
                data = json.load(f)
            assert data["num_samples"] == 100

    def test_inference_result_load_method(self) -> None:
        """Test InferenceResult.load() method."""
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test NPZ roundtrip
            npz_path = Path(tmpdir) / "result.npz"
            result.save(str(npz_path), format="npz")
            loaded_npz = InferenceResult.load(str(npz_path))
            assert loaded_npz.num_samples == result.num_samples

            # Test JSON roundtrip
            json_path = Path(tmpdir) / "result.json"
            result.save(str(json_path), format="json")
            loaded_json = InferenceResult.load(str(json_path))
            assert loaded_json.num_samples == result.num_samples
            assert loaded_json.sampler == result.sampler

    def test_inference_result_to_dict(self) -> None:
        """Test InferenceResult.to_dict() method."""
        result = _make_result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["num_samples"] == 100

    def test_inference_result_summary(self) -> None:
        """Test InferenceResult.summary() method."""
        result = _make_result()
        s = result.summary()

        assert "alpha" in s
        assert "beta" in s
        assert "mean" in s["alpha"]
        assert "std" in s["alpha"]
        assert "ess" in s["alpha"]


class TestDerivedParameters:
    """Tests for derived parameters functionality."""

    def test_add_derived_basic(self) -> None:
        """Test adding a derived parameter."""
        result = _make_result()
        # Create derived param with matching shape (1, 100)
        derived_samples: NDArray[np.float64] = result.samples["alpha"] + result.samples["beta"]
        result.add_derived("sum_ab", derived_samples)

        assert "sum_ab" in result.derived
        np.testing.assert_array_equal(result.derived["sum_ab"], derived_samples)

    def test_add_derived_multichain(self) -> None:
        """Test adding derived parameter with multiple chains."""
        result = _make_multichain_result()
        # Create derived param with matching shape (4, 100)
        derived_samples: NDArray[np.float64] = result.samples["alpha"] * 2
        result.add_derived("double_alpha", derived_samples)

        assert result.derived["double_alpha"].shape == (4, 100)

    def test_add_derived_rejects_existing_sample_name(self) -> None:
        """Test that add_derived rejects names that exist in samples."""
        result = _make_result()
        derived_samples: NDArray[np.float64] = result.samples["alpha"].copy()

        import pytest

        with pytest.raises(ValueError, match="already exists in samples"):
            result.add_derived("alpha", derived_samples)

    def test_add_derived_rejects_duplicate_derived_name(self) -> None:
        """Test that add_derived rejects duplicate derived names."""
        result = _make_result()
        derived_samples: NDArray[np.float64] = result.samples["alpha"].copy()
        result.add_derived("rho", derived_samples)

        import pytest

        with pytest.raises(ValueError, match="already exists in derived"):
            result.add_derived("rho", derived_samples)

    def test_add_derived_rejects_wrong_shape(self) -> None:
        """Test that add_derived rejects wrong shape."""
        result = _make_result()
        # Wrong shape: (2, 50) instead of (1, 100)
        wrong_shape: NDArray[np.float64] = np.random.default_rng(0).standard_normal((2, 50))

        import pytest

        with pytest.raises(ValueError, match="Shape must be"):
            result.add_derived("bad", wrong_shape)

    def test_summary_includes_derived(self) -> None:
        """Test that summary() includes derived parameters."""
        result = _make_result()
        derived_samples: NDArray[np.float64] = result.samples["alpha"] * 2
        result.add_derived("double_alpha", derived_samples)

        s = result.summary()

        assert "alpha" in s
        assert "double_alpha" in s
        assert "mean" in s["double_alpha"]
        assert "ess" in s["double_alpha"]

    def test_summary_filter_derived(self) -> None:
        """Test that summary() can filter to just derived params."""
        result = _make_result()
        derived_samples: NDArray[np.float64] = result.samples["alpha"] + result.samples["beta"]
        result.add_derived("sum_ab", derived_samples)

        s = result.summary(params=["sum_ab"])

        assert "sum_ab" in s
        assert "alpha" not in s

    def test_save_load_npz_with_derived(self) -> None:
        """Test save/load NPZ preserves derived parameters."""
        result = _make_result()
        derived_samples: NDArray[np.float64] = result.samples["alpha"] * 3
        result.add_derived("triple_alpha", derived_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.npz"
            save_npz(result, str(path))
            loaded = load_npz(str(path))

            assert "triple_alpha" in loaded.derived
            np.testing.assert_array_almost_equal(
                loaded.derived["triple_alpha"], result.derived["triple_alpha"]
            )

    def test_save_load_json_with_derived(self) -> None:
        """Test save/load JSON preserves derived parameters."""
        result = _make_result()
        derived_samples: NDArray[np.float64] = result.samples["alpha"] - result.samples["beta"]
        result.add_derived("diff_ab", derived_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            result.save(str(path), format="json")
            loaded = InferenceResult.load(str(path))

            assert "diff_ab" in loaded.derived
            np.testing.assert_array_almost_equal(
                loaded.derived["diff_ab"], result.derived["diff_ab"]
            )

    def test_to_json_includes_derived(self) -> None:
        """Test to_json includes derived parameters."""
        result = _make_result()
        derived_samples: NDArray[np.float64] = result.samples["alpha"].copy()
        result.add_derived("rho", derived_samples)

        d = to_json(result)

        assert "derived" in d
        assert "rho" in d["derived"]
