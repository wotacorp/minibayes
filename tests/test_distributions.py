"""Tests for probability distributions."""

import pytest


class TestNormal:
    """Tests for Normal distribution."""

    def test_log_prob(self) -> None:
        """Test log_prob against scipy.stats."""
        pytest.skip("Not implemented")

    def test_sample_shape(self) -> None:
        """Test sample output shape."""
        pytest.skip("Not implemented")

    def test_support(self) -> None:
        """Test support is REAL."""
        pytest.skip("Not implemented")


class TestHalfNormal:
    """Tests for HalfNormal distribution."""

    def test_log_prob(self) -> None:
        """Test log_prob against scipy.stats."""
        pytest.skip("Not implemented")

    def test_sample_positive(self) -> None:
        """Test samples are positive."""
        pytest.skip("Not implemented")

    def test_support(self) -> None:
        """Test support is POSITIVE."""
        pytest.skip("Not implemented")


class TestExponential:
    """Tests for Exponential distribution."""

    def test_log_prob(self) -> None:
        """Test log_prob against scipy.stats."""
        pytest.skip("Not implemented")

    def test_support(self) -> None:
        """Test support is POSITIVE."""
        pytest.skip("Not implemented")


class TestBeta:
    """Tests for Beta distribution."""

    def test_log_prob(self) -> None:
        """Test log_prob against scipy.stats."""
        pytest.skip("Not implemented")

    def test_support(self) -> None:
        """Test support is UNIT."""
        pytest.skip("Not implemented")


class TestGamma:
    """Tests for Gamma distribution."""

    def test_log_prob(self) -> None:
        """Test log_prob against scipy.stats."""
        pytest.skip("Not implemented")

    def test_support(self) -> None:
        """Test support is POSITIVE."""
        pytest.skip("Not implemented")


class TestUniform:
    """Tests for Uniform distribution."""

    def test_log_prob(self) -> None:
        """Test log_prob against scipy.stats."""
        pytest.skip("Not implemented")

    def test_support(self) -> None:
        """Test support is BOUNDED."""
        pytest.skip("Not implemented")
