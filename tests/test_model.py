"""Tests for Model class."""

import pytest


class TestModel:
    """Tests for Model class."""

    def test_param_names(self) -> None:
        """Test param_names returns prior keys."""
        pytest.skip("Not implemented")

    def test_sample_prior(self) -> None:
        """Test sample_prior returns valid samples."""
        pytest.skip("Not implemented")

    def test_log_prior(self) -> None:
        """Test log_prior computes sum of prior log_probs."""
        pytest.skip("Not implemented")

    def test_log_likelihood(self) -> None:
        """Test log_likelihood calls user function."""
        pytest.skip("Not implemented")

    def test_log_prob(self) -> None:
        """Test log_prob = log_prior + log_likelihood."""
        pytest.skip("Not implemented")

    def test_transforms_from_support(self) -> None:
        """Test transforms are inferred from distribution support."""
        pytest.skip("Not implemented")

    def test_to_unconstrained_roundtrip(self) -> None:
        """Test to_constrained(to_unconstrained(x)) == x."""
        pytest.skip("Not implemented")

    def test_log_prob_unconstrained_jacobian(self) -> None:
        """Test log_prob_unconstrained includes Jacobian correction."""
        pytest.skip("Not implemented")

    def test_validate_params_correct_names(self) -> None:
        """Test validate_params checks parameter names."""
        pytest.skip("Not implemented")

    def test_validate_params_within_support(self) -> None:
        """Test validate_params checks values are in support."""
        pytest.skip("Not implemented")
