"""Tests for Model class."""

import numpy as np
import pytest
from scipy import stats

from minibayes import dist
from minibayes.exceptions import ModelSpecError
from minibayes.model import Model
from minibayes.transforms import (
    AffineTransform,
    IdentityTransform,
    LogitTransform,
    LogTransform,
)


def simple_likelihood(params: dict[str, float], data: object) -> float:
    """Simple likelihood for testing: ignores data, returns constant."""
    del data  # unused
    del params  # unused
    return -1.0


def normal_likelihood(params: dict[str, float], data: object) -> float:
    """Normal likelihood for linear model."""
    y = data
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    mu = params["mu"]
    sigma = params["sigma"]
    d = dist.Normal(loc=mu, scale=sigma)
    result = d.log_prob(y)
    if isinstance(result, np.ndarray):
        return float(np.sum(result))
    return float(result)


class TestModel:
    """Tests for Model class."""

    def test_param_names(self) -> None:
        """Test param_names returns prior keys in order."""
        priors = {
            "alpha": dist.Normal(0, 1),
            "beta": dist.Normal(0, 5),
            "sigma": dist.HalfNormal(1),
        }
        model = Model(priors=priors, likelihood=simple_likelihood)
        assert model.param_names == ["alpha", "beta", "sigma"]

    def test_sample_prior(self) -> None:
        """Test sample_prior returns valid samples."""
        priors = {
            "mu": dist.Normal(0, 10),
            "sigma": dist.HalfNormal(5),
        }
        model = Model(priors=priors, likelihood=simple_likelihood)
        rng = np.random.default_rng(42)
        sample = model.sample_prior(rng=rng)

        assert set(sample.keys()) == {"mu", "sigma"}
        assert np.isfinite(sample["mu"])
        assert np.isfinite(sample["sigma"])
        assert sample["sigma"] > 0  # HalfNormal is positive

    def test_prior_means(self) -> None:
        """Test prior_means returns mean of each prior."""
        priors = {
            "mu": dist.Normal(loc=5.0, scale=2.0),
            "sigma": dist.HalfNormal(scale=3.0),
            "p": dist.Beta(alpha=2.0, beta=3.0),
        }
        model = Model(priors=priors, likelihood=simple_likelihood)
        means = model.prior_means()

        assert set(means.keys()) == {"mu", "sigma", "p"}
        np.testing.assert_allclose(means["mu"], 5.0, rtol=1e-10)
        np.testing.assert_allclose(means["sigma"], 3.0 * np.sqrt(2 / np.pi), rtol=1e-10)
        np.testing.assert_allclose(means["p"], 2.0 / 5.0, rtol=1e-10)

    def test_log_prior(self) -> None:
        """Test log_prior computes sum of prior log_probs."""
        priors = {
            "mu": dist.Normal(0, 1),
            "sigma": dist.HalfNormal(1),
        }
        model = Model(priors=priors, likelihood=simple_likelihood)
        params = {"mu": 0.5, "sigma": 1.0}

        result = model.log_prior(params)

        # Compare to scipy
        expected_mu: float = float(stats.norm(0, 1).logpdf(0.5))
        expected_sigma: float = float(stats.halfnorm(scale=1).logpdf(1.0))
        expected = expected_mu + expected_sigma

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_likelihood(self) -> None:
        """Test log_likelihood calls user function."""
        priors = {"mu": dist.Normal(0, 1)}
        call_log: list[tuple[dict[str, float], object]] = []

        def tracking_likelihood(params: dict[str, float], data: object) -> float:
            call_log.append((params.copy(), data))
            return -2.5

        model = Model(priors=priors, likelihood=tracking_likelihood)
        params = {"mu": 1.0}
        data = "test_data"

        result = model.log_likelihood(params, data)

        assert result == -2.5
        assert len(call_log) == 1
        assert call_log[0][0] == params
        assert call_log[0][1] == data

    def test_log_prob(self) -> None:
        """Test log_prob = log_prior + log_likelihood."""
        priors = {
            "mu": dist.Normal(0, 1),
            "sigma": dist.HalfNormal(1),
        }

        def fixed_likelihood(params: dict[str, float], data: object) -> float:
            del params, data
            return -5.0

        model = Model(priors=priors, likelihood=fixed_likelihood)
        params = {"mu": 0.0, "sigma": 1.0}
        data = None

        result = model.log_prob(params, data)
        expected = model.log_prior(params) + model.log_likelihood(params, data)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_transforms_from_support(self) -> None:
        """Test transforms are inferred from distribution support."""
        priors = {
            "real_param": dist.Normal(0, 1),  # REAL -> Identity
            "positive_param": dist.HalfNormal(1),  # POSITIVE -> Log
            "unit_param": dist.Beta(2, 2),  # UNIT -> Logit
            "bounded_param": dist.Uniform(0, 10),  # BOUNDED -> Affine
        }
        model = Model(priors=priors, likelihood=simple_likelihood)
        transforms = model.transforms

        assert isinstance(transforms["real_param"], IdentityTransform)
        assert isinstance(transforms["positive_param"], LogTransform)
        assert isinstance(transforms["unit_param"], LogitTransform)
        assert isinstance(transforms["bounded_param"], AffineTransform)

    def test_to_unconstrained_roundtrip(self) -> None:
        """Test to_constrained(to_unconstrained(x)) == x."""
        priors = {
            "mu": dist.Normal(0, 10),
            "sigma": dist.HalfNormal(5),
            "p": dist.Beta(2, 2),
        }
        model = Model(priors=priors, likelihood=simple_likelihood)

        # Test with various parameter values
        params = {"mu": 2.5, "sigma": 1.5, "p": 0.3}

        unconstrained = model.to_unconstrained(params)
        recovered = model.to_constrained(unconstrained)

        for name in params:
            np.testing.assert_allclose(recovered[name], params[name], rtol=1e-10)

    def test_log_prob_unconstrained_jacobian(self) -> None:
        """Test log_prob_unconstrained includes Jacobian correction."""
        # Use HalfNormal (log transform) to verify Jacobian
        priors = {"sigma": dist.HalfNormal(1)}

        def zero_likelihood(params: dict[str, float], data: object) -> float:
            del params, data
            return 0.0

        model = Model(priors=priors, likelihood=zero_likelihood)

        # In constrained space: sigma = 2.0
        constrained_params = {"sigma": 2.0}
        unconstrained_params = model.to_unconstrained(constrained_params)

        # log_prob in constrained space
        lp_constrained = model.log_prob(constrained_params, None)

        # log_prob in unconstrained space
        lp_unconstrained = model.log_prob_unconstrained(unconstrained_params, None)

        # For log transform: Jacobian = log(sigma)
        # So log_prob_unconstrained = log_prob_constrained + log(sigma)
        expected_jacobian: float = float(np.log(2.0))
        expected = lp_constrained + expected_jacobian

        np.testing.assert_allclose(lp_unconstrained, expected, rtol=1e-10)

    def test_validate_params_correct_names(self) -> None:
        """Test validate_params checks parameter names."""
        priors = {
            "mu": dist.Normal(0, 1),
            "sigma": dist.HalfNormal(1),
        }
        model = Model(priors=priors, likelihood=simple_likelihood)

        # Valid params should return True
        valid_params = {"mu": 0.0, "sigma": 1.0}
        assert model.validate_params(valid_params) is True

        # Missing parameter should raise
        with pytest.raises(ModelSpecError, match="Missing parameters"):
            model.validate_params({"mu": 0.0})

        # Extra parameter should raise
        with pytest.raises(ModelSpecError, match="Unknown parameters"):
            model.validate_params({"mu": 0.0, "sigma": 1.0, "extra": 0.0})

    def test_validate_params_within_support(self) -> None:
        """Test validate_params checks values are in support."""
        priors = {
            "mu": dist.Normal(0, 1),
            "sigma": dist.HalfNormal(1),
        }
        model = Model(priors=priors, likelihood=simple_likelihood)

        # Negative sigma is outside support
        with pytest.raises(ModelSpecError, match="outside support"):
            model.validate_params({"mu": 0.0, "sigma": -1.0})

        # Zero sigma is also outside support for HalfNormal
        with pytest.raises(ModelSpecError, match="outside support"):
            model.validate_params({"mu": 0.0, "sigma": 0.0})


class TestModelEdgeCases:
    """Additional edge case tests for Model class."""

    def test_empty_priors_raises(self) -> None:
        """Empty priors dict raises ModelSpecError."""
        with pytest.raises(ModelSpecError, match="non-empty"):
            Model(priors={}, likelihood=simple_likelihood)

    def test_single_param_model(self) -> None:
        """Model works with single parameter."""
        priors = {"theta": dist.Normal(0, 1)}
        model = Model(priors=priors, likelihood=simple_likelihood)

        assert model.param_names == ["theta"]
        sample = model.sample_prior()
        assert "theta" in sample

    def test_sample_prior_reproducibility(self) -> None:
        """Same seed produces same samples."""
        priors = {"mu": dist.Normal(0, 10), "sigma": dist.HalfNormal(5)}
        model = Model(priors=priors, likelihood=simple_likelihood)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        sample1 = model.sample_prior(rng=rng1)
        sample2 = model.sample_prior(rng=rng2)

        assert sample1["mu"] == sample2["mu"]
        assert sample1["sigma"] == sample2["sigma"]

    def test_log_prob_unconstrained_identity_no_jacobian(self) -> None:
        """For REAL support, no Jacobian correction is added."""
        priors = {"mu": dist.Normal(0, 1)}

        def zero_likelihood(params: dict[str, float], data: object) -> float:
            del params, data
            return 0.0

        model = Model(priors=priors, likelihood=zero_likelihood)

        params = {"mu": 2.0}
        unconstrained = model.to_unconstrained(params)

        # For identity transform, constrained == unconstrained
        assert unconstrained["mu"] == params["mu"]

        # And log_prob_unconstrained == log_prob (no Jacobian adjustment)
        lp_constrained = model.log_prob(params, None)
        lp_unconstrained = model.log_prob_unconstrained(unconstrained, None)

        np.testing.assert_allclose(lp_unconstrained, lp_constrained, rtol=1e-10)
