"""Tests for utility functions."""

import pytest


class TestEnsureRng:
    """Tests for ensure_rng function."""

    def test_with_seed(self) -> None:
        """Test ensure_rng with integer seed."""
        pytest.skip("Not implemented")

    def test_with_generator(self) -> None:
        """Test ensure_rng with existing Generator."""
        pytest.skip("Not implemented")

    def test_with_none(self) -> None:
        """Test ensure_rng with None."""
        pytest.skip("Not implemented")


class TestCheckFinite:
    """Tests for check_finite function."""

    def test_finite_value(self) -> None:
        """Test check_finite with finite value."""
        pytest.skip("Not implemented")

    def test_nan_raises(self) -> None:
        """Test check_finite raises on NaN."""
        pytest.skip("Not implemented")

    def test_inf_raises(self) -> None:
        """Test check_finite raises on Inf."""
        pytest.skip("Not implemented")


class TestLogSumExp:
    """Tests for log_sum_exp function."""

    def test_simple_case(self) -> None:
        """Test log_sum_exp on simple array."""
        pytest.skip("Not implemented")

    def test_numerical_stability(self) -> None:
        """Test log_sum_exp handles large values."""
        pytest.skip("Not implemented")
