"""Tests for MCMC samplers."""

import numpy as np
import pytest

from minibayes.exceptions import ModelSpecError
from minibayes.samplers import AdaptiveMetropolis, MetropolisHastings


class TestMetropolisHastings:
    """Tests for MetropolisHastings sampler."""

    def test_step_returns_dict_and_bool(self, rng: np.random.Generator) -> None:
        """Test step returns (params, accepted)."""
        sampler = MetropolisHastings(proposal_scale=1.0)
        current: dict[str, float] = {"mu": 0.0}

        def log_prob(p: dict[str, float]) -> float:
            return -0.5 * p["mu"] ** 2  # Standard normal

        new_state, accepted = sampler.step(current, log_prob, rng)
        assert isinstance(new_state, dict)
        assert isinstance(accepted, bool)
        assert "mu" in new_state

    def test_accepts_higher_probability(self, rng: np.random.Generator) -> None:
        """Test proposals with higher log_prob are accepted."""
        sampler = MetropolisHastings(proposal_scale=0.001)  # tiny steps
        current: dict[str, float] = {"x": 10.0}

        def log_prob(p: dict[str, float]) -> float:
            return -(p["x"] ** 2)  # max at 0

        # Run many steps; should accept most moves toward 0
        accepts = 0
        state = current.copy()
        for _ in range(100):
            state, acc = sampler.step(state, log_prob, rng)
            if acc:
                accepts += 1
        assert accepts > 50  # should accept most proposals

    def test_normal_normal_posterior(self, rng: np.random.Generator) -> None:
        """Test on Normal-Normal model with known posterior."""
        # Prior: N(0, 1), Likelihood: N(mu, 1) with data
        # Analytical posterior: N(n*x_bar/(n+1), 1/(n+1))
        data = [1.5, 2.0, 1.8, 2.2, 1.9]
        n = len(data)
        x_bar: float = sum(data) / n
        posterior_mean: float = n * x_bar / (n + 1)  # ~1.57

        def log_prob(p: dict[str, float]) -> float:
            mu: float = p["mu"]
            log_prior: float = -0.5 * mu**2
            log_lik: float = sum(-0.5 * (x - mu) ** 2 for x in data)
            return log_prior + log_lik

        sampler = MetropolisHastings(proposal_scale=0.5)
        samples: list[float] = []
        state: dict[str, float] = {"mu": 0.0}

        # Warmup
        for _ in range(500):
            state, _ = sampler.step(state, log_prob, rng)

        # Sample
        for _ in range(2000):
            state, _ = sampler.step(state, log_prob, rng)
            samples.append(state["mu"])

        empirical_mean: float = float(np.mean(samples))
        assert abs(empirical_mean - posterior_mean) < 0.1

    def test_per_parameter_scales(self, rng: np.random.Generator) -> None:
        """Test per-parameter proposal scales."""
        sampler = MetropolisHastings(proposal_scale={"a": 0.1, "b": 2.0})
        current: dict[str, float] = {"a": 0.0, "b": 0.0}

        def log_prob(p: dict[str, float]) -> float:
            return -0.5 * (p["a"] ** 2 + p["b"] ** 2)

        new_state, _ = sampler.step(current, log_prob, rng)
        assert "a" in new_state
        assert "b" in new_state

    def test_invalid_scale_raises(self) -> None:
        """Test that invalid proposal_scale raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            MetropolisHastings(proposal_scale=-1.0)

        with pytest.raises(ModelSpecError):
            MetropolisHastings(proposal_scale=0.0)

        with pytest.raises(ModelSpecError):
            MetropolisHastings(proposal_scale={"x": -0.5})

    def test_warmup_step_same_as_step(self, rng: np.random.Generator) -> None:
        """Test warmup_step delegates to step for basic MH."""
        sampler = MetropolisHastings(proposal_scale=1.0)
        current: dict[str, float] = {"mu": 0.0}

        def log_prob(p: dict[str, float]) -> float:
            return -0.5 * p["mu"] ** 2

        # warmup_step should work the same as step
        new_state, accepted = sampler.warmup_step(current, log_prob, rng, step_num=0)
        assert isinstance(new_state, dict)
        assert isinstance(accepted, bool)


class TestAdaptiveMetropolis:
    """Tests for AdaptiveMetropolis sampler."""

    def test_step_returns_dict_and_bool(self, rng: np.random.Generator) -> None:
        """Test step returns (params, accepted)."""
        sampler = AdaptiveMetropolis(initial_scale=1.0)
        current: dict[str, float] = {"mu": 0.0}

        def log_prob(p: dict[str, float]) -> float:
            return -0.5 * p["mu"] ** 2

        new_state, accepted = sampler.step(current, log_prob, rng)
        assert isinstance(new_state, dict)
        assert isinstance(accepted, bool)
        assert "mu" in new_state

    def test_warmup_adapts_covariance(self, rng: np.random.Generator) -> None:
        """Test warmup_step adapts proposal covariance."""
        sampler = AdaptiveMetropolis(initial_scale=1.0)
        state: dict[str, float] = {"x": 0.0, "y": 0.0}

        def log_prob(p: dict[str, float]) -> float:
            return -0.5 * (p["x"] ** 2 + p["y"] ** 2)

        # Run warmup (need at least 50 steps for adaptation to kick in)
        for i in range(200):
            state, _ = sampler.warmup_step(state, log_prob, rng, step_num=i)

        # Covariance should be set after warmup
        assert sampler._cov is not None
        assert sampler._cov.shape == (2, 2)

    def test_normal_normal_posterior(self, rng: np.random.Generator) -> None:
        """Test on Normal-Normal model with known posterior."""
        data = [1.5, 2.0, 1.8, 2.2, 1.9]
        n = len(data)
        x_bar: float = sum(data) / n
        posterior_mean: float = n * x_bar / (n + 1)

        def log_prob(p: dict[str, float]) -> float:
            mu: float = p["mu"]
            log_prior: float = -0.5 * mu**2
            log_lik: float = sum(-0.5 * (x - mu) ** 2 for x in data)
            return log_prior + log_lik

        sampler = AdaptiveMetropolis(initial_scale=1.0)
        state: dict[str, float] = {"mu": 0.0}

        # Warmup
        for i in range(500):
            state, _ = sampler.warmup_step(state, log_prob, rng, step_num=i)
        sampler.freeze()

        # Sample
        samples: list[float] = []
        for _ in range(2000):
            state, _ = sampler.step(state, log_prob, rng)
            samples.append(state["mu"])

        empirical_mean: float = float(np.mean(samples))
        assert abs(empirical_mean - posterior_mean) < 0.15

    def test_acceptance_rate_near_target(self, rng: np.random.Generator) -> None:
        """Test acceptance rate is reasonable after warmup."""
        sampler = AdaptiveMetropolis(initial_scale=0.1)
        state: dict[str, float] = {"x": 0.0, "y": 0.0}

        def log_prob(p: dict[str, float]) -> float:
            return -0.5 * (p["x"] ** 2 + p["y"] ** 2)

        # Long warmup to allow adaptation
        for i in range(1000):
            state, _ = sampler.warmup_step(state, log_prob, rng, step_num=i)
        sampler.freeze()

        # Check acceptance rate during sampling
        accepts = 0
        for _ in range(1000):
            state, accepted = sampler.step(state, log_prob, rng)
            accepts += int(accepted)

        acceptance_rate = accepts / 1000
        # Should be in reasonable range (10-60%)
        assert 0.10 < acceptance_rate < 0.60

    def test_freeze_clears_history(self, rng: np.random.Generator) -> None:
        """Test freeze() clears sample history."""
        sampler = AdaptiveMetropolis(initial_scale=1.0)
        state: dict[str, float] = {"x": 0.0}

        def log_prob(p: dict[str, float]) -> float:
            return -0.5 * p["x"] ** 2

        # Run some warmup steps
        for i in range(100):
            state, _ = sampler.warmup_step(state, log_prob, rng, step_num=i)

        assert len(sampler._sample_history) > 0

        sampler.freeze()

        assert len(sampler._sample_history) == 0
        assert sampler._frozen is True

    def test_invalid_initial_scale_raises(self) -> None:
        """Test that invalid initial_scale raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            AdaptiveMetropolis(initial_scale=-1.0)

        with pytest.raises(ModelSpecError):
            AdaptiveMetropolis(initial_scale=0.0)

    def test_invalid_target_acceptance_raises(self) -> None:
        """Test that invalid target_acceptance raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            AdaptiveMetropolis(target_acceptance=0.0)

        with pytest.raises(ModelSpecError):
            AdaptiveMetropolis(target_acceptance=1.0)

        with pytest.raises(ModelSpecError):
            AdaptiveMetropolis(target_acceptance=-0.5)
