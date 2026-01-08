"""Tests for convergence diagnostics."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from minibayes.diagnostics import effective_sample_size, r_hat, summary


class TestEffectiveSampleSize:
    """Tests for effective_sample_size function."""

    def test_independent_samples(self) -> None:
        """Test ESS ≈ n for independent samples."""
        rng = np.random.default_rng(42)
        n = 1000
        samples: NDArray[np.float64] = rng.standard_normal(n)
        ess = effective_sample_size(samples)
        # For iid samples, ESS should be close to n
        # Allow some variance, but should be > 0.7 * n
        assert ess > 0.7 * n, f"ESS {ess} too low for iid samples"
        assert ess <= n, f"ESS {ess} should not exceed n"

    def test_correlated_samples(self) -> None:
        """Test ESS < n for correlated samples."""
        # Create AR(1) process with high correlation
        rng = np.random.default_rng(42)
        n = 1000
        phi = 0.95  # High autocorrelation
        samples: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        samples[0] = rng.standard_normal()
        for i in range(1, n):
            samples[i] = phi * samples[i - 1] + rng.standard_normal()

        ess = effective_sample_size(samples)
        # ESS should be much smaller than n for highly correlated samples
        assert ess < 0.5 * n, f"ESS {ess} too high for correlated samples"
        assert ess >= 1.0, f"ESS {ess} should be at least 1"

    def test_constant_samples(self) -> None:
        """Test ESS for constant (degenerate) samples."""
        samples: NDArray[np.float64] = np.ones(100, dtype=np.float64)
        ess = effective_sample_size(samples)
        # For constant samples, variance is 0, so ESS = n (our implementation)
        assert ess == 100.0, f"ESS for constant samples should be n, got {ess}"

    def test_small_sample(self) -> None:
        """Test ESS returns n for very small samples."""
        samples: NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
        ess = effective_sample_size(samples)
        assert ess == 3.0, f"ESS for n<4 should equal n, got {ess}"


class TestRHat:
    """Tests for r_hat function."""

    def test_converged_chains(self) -> None:
        """Test R-hat ≈ 1.0 for converged chains."""
        rng = np.random.default_rng(42)
        n_chains = 4
        n_samples = 500
        # All chains from same distribution
        chains: NDArray[np.float64] = rng.standard_normal((n_chains, n_samples))
        rhat = r_hat(chains)
        # R-hat should be very close to 1.0
        assert abs(rhat - 1.0) < 0.05, f"R-hat {rhat} not close to 1.0"

    def test_diverged_chains(self) -> None:
        """Test R-hat > 1.0 for diverged chains."""
        rng = np.random.default_rng(42)
        n_chains = 4
        n_samples = 500
        chains: NDArray[np.float64] = np.zeros((n_chains, n_samples), dtype=np.float64)
        # Each chain has different mean
        for i in range(n_chains):
            chains[i, :] = rng.standard_normal(n_samples) + i * 5.0

        rhat = r_hat(chains)
        # R-hat should be significantly > 1.0
        assert rhat > 1.5, f"R-hat {rhat} should be > 1.5 for diverged chains"

    def test_single_chain_returns_nan(self) -> None:
        """Test R-hat returns NaN for single chain."""
        rng = np.random.default_rng(42)
        chains: NDArray[np.float64] = rng.standard_normal((1, 100))
        rhat = r_hat(chains)
        assert math.isnan(rhat), f"R-hat for single chain should be NaN, got {rhat}"

    def test_wrong_dimension_raises(self) -> None:
        """Test R-hat raises for 1D input."""
        samples: NDArray[np.float64] = np.ones(100, dtype=np.float64)
        with pytest.raises(ValueError, match="must be 2D"):
            r_hat(samples)


class TestSummary:
    """Tests for summary function."""

    def test_summary_keys(self) -> None:
        """Test summary returns expected keys."""
        rng = np.random.default_rng(42)
        samples: dict[str, NDArray[np.float64]] = {
            "x": rng.standard_normal(100),
            "y": rng.standard_normal(100),
        }
        result = summary(samples)

        assert "x" in result
        assert "y" in result
        assert "mean" in result["x"]
        assert "std" in result["x"]
        assert "5%" in result["x"]
        assert "50%" in result["x"]
        assert "95%" in result["x"]
        assert "ess" in result["x"]

    def test_summary_percentiles(self) -> None:
        """Test custom percentiles."""
        rng = np.random.default_rng(42)
        samples: dict[str, NDArray[np.float64]] = {
            "x": rng.standard_normal(100),
        }
        result = summary(samples, percentiles=[10, 90])

        assert "10%" in result["x"]
        assert "90%" in result["x"]
        assert "5%" not in result["x"]

    def test_summary_multichain_includes_rhat(self) -> None:
        """Test summary includes R-hat for multi-chain samples."""
        rng = np.random.default_rng(42)
        samples: dict[str, NDArray[np.float64]] = {
            "x": rng.standard_normal((4, 100)),  # 4 chains
        }
        result = summary(samples)

        assert "r_hat" in result["x"]
        assert abs(result["x"]["r_hat"] - 1.0) < 0.1

    def test_summary_values_reasonable(self) -> None:
        """Test summary computes reasonable values."""
        rng = np.random.default_rng(42)
        # Standard normal: mean ≈ 0, std ≈ 1
        samples: dict[str, NDArray[np.float64]] = {
            "x": rng.standard_normal(10000),
        }
        result = summary(samples)

        assert abs(result["x"]["mean"]) < 0.1
        assert abs(result["x"]["std"] - 1.0) < 0.1
        assert result["x"]["50%"] < result["x"]["95%"]
        assert result["x"]["5%"] < result["x"]["50%"]
