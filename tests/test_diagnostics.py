"""Tests for convergence diagnostics."""

import pytest


class TestEffectiveSampleSize:
    """Tests for effective_sample_size function."""

    def test_independent_samples(self) -> None:
        """Test ESS ≈ n for independent samples."""
        pytest.skip("Not implemented")

    def test_correlated_samples(self) -> None:
        """Test ESS < n for correlated samples."""
        pytest.skip("Not implemented")

    def test_constant_samples(self) -> None:
        """Test ESS for constant (degenerate) samples."""
        pytest.skip("Not implemented")


class TestRHat:
    """Tests for r_hat function."""

    def test_converged_chains(self) -> None:
        """Test R-hat ≈ 1.0 for converged chains."""
        pytest.skip("Not implemented")

    def test_diverged_chains(self) -> None:
        """Test R-hat > 1.0 for diverged chains."""
        pytest.skip("Not implemented")

    def test_single_chain_raises(self) -> None:
        """Test R-hat requires multiple chains."""
        pytest.skip("Not implemented")


class TestSummary:
    """Tests for summary function."""

    def test_summary_keys(self) -> None:
        """Test summary returns expected keys."""
        pytest.skip("Not implemented")

    def test_summary_percentiles(self) -> None:
        """Test custom percentiles."""
        pytest.skip("Not implemented")
