"""Tests for utility functions."""

import numpy as np
import pytest

from minibayes.exceptions import NumericalError
from minibayes.utils import check_finite, ensure_rng, log_sum_exp


class TestEnsureRng:
    """Tests for ensure_rng function."""

    def test_with_seed(self) -> None:
        """Test ensure_rng with integer seed."""
        rng = ensure_rng(42)
        assert isinstance(rng, np.random.Generator)
        # Same seed should produce same sequence
        rng1 = ensure_rng(42)
        rng2 = ensure_rng(42)
        assert rng1.random() == rng2.random()

    def test_with_generator(self) -> None:
        """Test ensure_rng with existing Generator."""
        original = np.random.default_rng(123)
        result = ensure_rng(original)
        assert result is original

    def test_with_none(self) -> None:
        """Test ensure_rng with None."""
        rng = ensure_rng(None)
        assert isinstance(rng, np.random.Generator)


class TestCheckFinite:
    """Tests for check_finite function."""

    def test_finite_value(self) -> None:
        """Test check_finite with finite value."""
        check_finite(1.0, "test")
        check_finite(0.0, "test")
        check_finite(-1e10, "test")

    def test_nan_raises(self) -> None:
        """Test check_finite raises on NaN."""
        with pytest.raises(NumericalError, match="not finite"):
            check_finite(np.nan, "test_value")

    def test_inf_raises(self) -> None:
        """Test check_finite raises on Inf."""
        with pytest.raises(NumericalError, match="not finite"):
            check_finite(np.inf, "test_value")
        with pytest.raises(NumericalError, match="not finite"):
            check_finite(-np.inf, "test_value")


class TestLogSumExp:
    """Tests for log_sum_exp function."""

    def test_simple_case(self) -> None:
        """Test log_sum_exp on simple array."""
        x = np.array([1.0, 2.0, 3.0])
        result = log_sum_exp(x)
        expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
        assert np.isclose(result, expected)

    def test_numerical_stability(self) -> None:
        """Test log_sum_exp handles large values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = log_sum_exp(x)
        # Should not overflow - result should be approximately 1002 + log(1 + e^-1 + e^-2)
        expected = 1002.0 + np.log(1 + np.exp(-1) + np.exp(-2))
        assert np.isclose(result, expected)

    def test_all_negative_inf(self) -> None:
        """Test log_sum_exp with all -inf."""
        x = np.array([-np.inf, -np.inf])
        result = log_sum_exp(x)
        assert result == float("-inf")

    def test_single_element(self) -> None:
        """Test log_sum_exp with single element."""
        x = np.array([5.0])
        result = log_sum_exp(x)
        assert np.isclose(result, 5.0)
