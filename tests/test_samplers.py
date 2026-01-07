"""Tests for MCMC samplers."""

import pytest


class TestMetropolisHastings:
    """Tests for MetropolisHastings sampler."""

    def test_step_returns_dict_and_bool(self) -> None:
        """Test step returns (params, accepted)."""
        pytest.skip("Not implemented")

    def test_accepts_higher_probability(self) -> None:
        """Test proposals with higher log_prob are accepted."""
        pytest.skip("Not implemented")

    def test_normal_normal_posterior(self) -> None:
        """Test on Normal-Normal model with known posterior."""
        pytest.skip("Not implemented")


class TestAdaptiveMetropolis:
    """Tests for AdaptiveMetropolis sampler."""

    def test_step_returns_dict_and_bool(self) -> None:
        """Test step returns (params, accepted)."""
        pytest.skip("Not implemented")

    def test_warmup_adapts_covariance(self) -> None:
        """Test warmup_step adapts proposal covariance."""
        pytest.skip("Not implemented")

    def test_normal_normal_posterior(self) -> None:
        """Test on Normal-Normal model with known posterior."""
        pytest.skip("Not implemented")

    def test_acceptance_rate_near_target(self) -> None:
        """Test acceptance rate approaches target after warmup."""
        pytest.skip("Not implemented")
