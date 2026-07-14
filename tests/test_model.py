# Copyright 2026 WOTA CORP.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Model class."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from minibayes import dist
from minibayes.exceptions import ModelSpecError
from minibayes.model import Model, StructuredParams
from minibayes.params import ParamContext, ParamMode
from minibayes.transforms import (
    AffineTransform,
    IdentityTransform,
    LogitTransform,
    LogTransform,
)


def simple_log_likelihood(
    params: StructuredParams, data: object
) -> NDArray[np.float64]:
    """Simple log-likelihood for testing: ignores data, returns constant array."""
    del data  # unused
    del params  # unused
    return np.array([-1.0], dtype=np.float64)


def normal_log_likelihood(
    params: StructuredParams, data: object
) -> NDArray[np.float64]:
    """Normal log-likelihood for linear model."""
    y = data
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    mu = params["mu"]
    sigma = params["sigma"]
    if not isinstance(mu, float):
        mu = float(mu)
    if not isinstance(sigma, float):
        sigma = float(sigma)
    d = dist.Normal(loc=mu, scale=sigma)
    return d.log_prob(y)


class TestParamContext:
    """Tests for ParamContext class."""

    def test_sample_mode_scalar(self) -> None:
        """Test ParamContext in sample mode with scalar params."""
        ctx = ParamContext(mode=ParamMode.SAMPLE, rng=np.random.default_rng(42))

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))
            p("sigma", dist.HalfNormal(1))

        priors(ctx)

        assert set(ctx.values.keys()) == {"mu", "sigma"}
        assert isinstance(ctx.values["mu"], float)
        assert isinstance(ctx.values["sigma"], float)
        assert ctx.values["sigma"] > 0

    def test_sample_mode_vector(self) -> None:
        """Test ParamContext in sample mode with vector params."""
        ctx = ParamContext(mode=ParamMode.SAMPLE, rng=np.random.default_rng(42))

        def priors(p: ParamContext) -> None:
            p("theta", dist.Normal(0, 1), size=5)

        priors(ctx)

        assert "theta" in ctx.values
        theta = ctx.values["theta"]
        assert isinstance(theta, np.ndarray)
        assert theta.shape == (5,)

    def test_evaluate_mode(self) -> None:
        """Test ParamContext in evaluate mode."""
        values = {"mu": 1.0, "sigma": 0.5}
        ctx = ParamContext(mode=ParamMode.EVALUATE, values=values)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("sigma", dist.HalfNormal(1))

        priors(ctx)

        # Check log_prob is accumulated
        expected_mu: float = float(stats.norm(0, 1).logpdf(1.0))
        expected_sigma: float = float(stats.halfnorm(scale=1).logpdf(0.5))
        expected = expected_mu + expected_sigma

        np.testing.assert_allclose(ctx.log_prob, expected, rtol=1e-10)

    def test_hierarchical_sampling(self) -> None:
        """Test that hierarchical dependencies work in sample mode."""
        ctx = ParamContext(mode=ParamMode.SAMPLE, rng=np.random.default_rng(42))

        def priors(p: ParamContext) -> None:
            mu = p("mu", dist.Normal(0, 5))
            sigma = p("sigma", dist.HalfNormal(1))
            # theta depends on mu and sigma
            p("theta", dist.Normal(mu, sigma), size=3)

        priors(ctx)

        # All params should be present
        assert set(ctx.values.keys()) == {"mu", "sigma", "theta"}
        # theta should be an array
        assert isinstance(ctx.values["theta"], np.ndarray)
        assert ctx.values["theta"].shape == (3,)

    def test_duplicate_param_raises(self) -> None:
        """Registering same param twice raises error."""
        ctx = ParamContext(mode=ParamMode.SAMPLE)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("mu", dist.Normal(0, 1))  # Duplicate!

        with pytest.raises(ModelSpecError, match="already registered"):
            priors(ctx)

    def test_param_info(self) -> None:
        """Test param_info records correct metadata."""
        ctx = ParamContext(mode=ParamMode.SAMPLE)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))
            p("theta", dist.Normal(0, 1), size=3)

        priors(ctx)

        info = ctx.param_info
        assert info["mu"].is_vector is False
        assert info["mu"].size == 1
        assert info["theta"].is_vector is True
        assert info["theta"].size == 3


class TestModel:
    """Tests for Model class."""

    def test_param_names(self) -> None:
        """Test param_names returns prior keys in order."""

        def priors(p: ParamContext) -> None:
            p("alpha", dist.Normal(0, 1))
            p("beta", dist.Normal(0, 5))
            p("sigma", dist.HalfNormal(1))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)
        assert model.param_names == ["alpha", "beta", "sigma"]

    def test_sample_prior(self) -> None:
        """Test sample_prior returns valid samples."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 10))
            p("sigma", dist.HalfNormal(5))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)
        rng = np.random.default_rng(42)
        sample = model.sample_prior(rng=rng)

        assert set(sample.keys()) == {"mu", "sigma"}
        assert np.isfinite(sample["mu"])
        assert np.isfinite(sample["sigma"])
        assert sample["sigma"] > 0  # HalfNormal is positive

    def test_prior_means(self) -> None:
        """Test prior_means returns mean of each prior."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(loc=5.0, scale=2.0))
            p("sigma", dist.HalfNormal(scale=3.0))
            p("prob", dist.Beta(alpha=2.0, beta=3.0))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)
        means = model.prior_means()

        assert set(means.keys()) == {"mu", "sigma", "prob"}
        np.testing.assert_allclose(means["mu"], 5.0, rtol=1e-10)
        np.testing.assert_allclose(means["sigma"], 3.0 * np.sqrt(2 / np.pi), rtol=1e-10)
        np.testing.assert_allclose(means["prob"], 2.0 / 5.0, rtol=1e-10)

    def test_log_prior(self) -> None:
        """Test log_prior computes sum of prior log_probs."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("sigma", dist.HalfNormal(1))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)
        params: StructuredParams = {"mu": 0.5, "sigma": 1.0}

        result = model.log_prior(params)

        # Compare to scipy
        expected_mu: float = float(stats.norm(0, 1).logpdf(0.5))
        expected_sigma: float = float(stats.halfnorm(scale=1).logpdf(1.0))
        expected = expected_mu + expected_sigma

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_likelihood(self) -> None:
        """Test log_likelihood calls user function."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))

        call_log: list[tuple[StructuredParams, object]] = []

        def tracking_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            call_log.append((dict(params), data))
            return np.array([-2.5], dtype=np.float64)

        model = Model(priors=priors, log_likelihood=tracking_likelihood)
        params: StructuredParams = {"mu": 1.0}
        data = "test_data"

        result = model.log_likelihood(params, data)

        np.testing.assert_allclose(result, np.array([-2.5]))
        assert len(call_log) == 1
        assert call_log[0][0] == params
        assert call_log[0][1] == data

    def test_log_prob(self) -> None:
        """Test log_prob = log_prior + sum(log_likelihood)."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("sigma", dist.HalfNormal(1))

        def fixed_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            del params, data
            return np.array([-5.0], dtype=np.float64)

        model = Model(priors=priors, log_likelihood=fixed_likelihood)
        params: StructuredParams = {"mu": 0.0, "sigma": 1.0}
        data = None

        result = model.log_prob(params, data)
        expected = model.log_prior(params) + float(
            np.sum(model.log_likelihood(params, data))
        )

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_transforms_from_support(self) -> None:
        """Test transforms are inferred from distribution support."""

        def priors(p: ParamContext) -> None:
            p("real_param", dist.Normal(0, 1))  # REAL -> Identity
            p("positive_param", dist.HalfNormal(1))  # POSITIVE -> Log
            p("unit_param", dist.Beta(2, 2))  # UNIT -> Logit
            p("bounded_param", dist.Uniform(0, 10))  # BOUNDED -> Affine

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)
        transforms = model.transforms

        assert isinstance(transforms["real_param"], IdentityTransform)
        assert isinstance(transforms["positive_param"], LogTransform)
        assert isinstance(transforms["unit_param"], LogitTransform)
        assert isinstance(transforms["bounded_param"], AffineTransform)

    def test_to_unconstrained_roundtrip(self) -> None:
        """Test to_constrained(to_unconstrained(x)) == x."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 10))
            p("sigma", dist.HalfNormal(5))
            p("prob", dist.Beta(2, 2))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        # Test with various parameter values
        params: StructuredParams = {"mu": 2.5, "sigma": 1.5, "prob": 0.3}

        unconstrained = model.to_unconstrained(params)
        recovered = model.to_constrained(unconstrained)

        for name in params:
            np.testing.assert_allclose(recovered[name], params[name], rtol=1e-10)

    def test_log_prob_unconstrained_jacobian(self) -> None:
        """Test log_prob_unconstrained includes Jacobian correction."""

        # Use HalfNormal (log transform) to verify Jacobian
        def priors(p: ParamContext) -> None:
            p("sigma", dist.HalfNormal(1))

        def zero_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            del params, data
            return np.array([0.0], dtype=np.float64)

        model = Model(priors=priors, log_likelihood=zero_likelihood)

        # In constrained space: sigma = 2.0
        constrained_params: StructuredParams = {"sigma": 2.0}
        flat_unconstrained = model.to_flat_unconstrained(constrained_params)

        # log_prob in constrained space
        lp_constrained = model.log_prob(constrained_params, None)

        # log_prob in unconstrained space
        lp_unconstrained = model.log_prob_unconstrained(flat_unconstrained, None)

        # For log transform: Jacobian = log(sigma)
        # So log_prob_unconstrained = log_prob_constrained + log(sigma)
        expected_jacobian: float = float(np.log(2.0))
        expected = lp_constrained + expected_jacobian

        np.testing.assert_allclose(lp_unconstrained, expected, rtol=1e-10)

    def test_validate_params_correct_names(self) -> None:
        """Test validate_params checks parameter names."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("sigma", dist.HalfNormal(1))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        # Valid params should return True
        valid_params: StructuredParams = {"mu": 0.0, "sigma": 1.0}
        assert model.validate_params(valid_params) is True

        # Missing parameter should raise
        with pytest.raises(ModelSpecError, match="Missing parameters"):
            model.validate_params({"mu": 0.0})

        # Extra parameter should raise
        with pytest.raises(ModelSpecError, match="Unknown parameters"):
            model.validate_params({"mu": 0.0, "sigma": 1.0, "extra": 0.0})

    def test_validate_params_within_support(self) -> None:
        """Test validate_params checks values are in support."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("sigma", dist.HalfNormal(1))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        # Negative sigma is outside support
        with pytest.raises(ModelSpecError, match="outside support"):
            model.validate_params({"mu": 0.0, "sigma": -1.0})

        # Zero sigma is also outside support for HalfNormal
        with pytest.raises(ModelSpecError, match="outside support"):
            model.validate_params({"mu": 0.0, "sigma": 0.0})


class TestModelEdgeCases:
    """Additional edge case tests for Model class."""

    def test_empty_priors_raises(self) -> None:
        """Empty priors function raises ModelSpecError."""

        def empty_priors(p: ParamContext) -> None:
            pass  # No params registered

        with pytest.raises(ModelSpecError, match="at least one parameter"):
            Model(priors=empty_priors, log_likelihood=simple_log_likelihood)

    def test_single_param_model(self) -> None:
        """Model works with single parameter."""

        def priors(p: ParamContext) -> None:
            p("theta", dist.Normal(0, 1))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        assert model.param_names == ["theta"]
        sample = model.sample_prior()
        assert "theta" in sample

    def test_sample_prior_reproducibility(self) -> None:
        """Same seed produces same samples."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 10))
            p("sigma", dist.HalfNormal(5))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        sample1 = model.sample_prior(rng=rng1)
        sample2 = model.sample_prior(rng=rng2)

        assert sample1["mu"] == sample2["mu"]
        assert sample1["sigma"] == sample2["sigma"]

    def test_log_prob_unconstrained_identity_no_jacobian(self) -> None:
        """For REAL support, no Jacobian correction is added."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))

        def zero_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            del params, data
            return np.array([0.0], dtype=np.float64)

        model = Model(priors=priors, log_likelihood=zero_likelihood)

        params: StructuredParams = {"mu": 2.0}
        flat_unconstrained = model.to_flat_unconstrained(params)

        # For identity transform, constrained == unconstrained
        assert flat_unconstrained["mu"] == params["mu"]

        # And log_prob_unconstrained == log_prob (no Jacobian adjustment)
        lp_constrained = model.log_prob(params, None)
        lp_unconstrained = model.log_prob_unconstrained(flat_unconstrained, None)

        np.testing.assert_allclose(lp_unconstrained, lp_constrained, rtol=1e-10)


class TestHierarchicalModel:
    """Tests for hierarchical model support."""

    def test_vector_parameter_sample(self) -> None:
        """Test sampling vector parameters."""

        def priors(p: ParamContext) -> None:
            p("theta", dist.Normal(0, 1), size=5)

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        sample = model.sample_prior(rng=np.random.default_rng(42))

        assert "theta" in sample
        theta = sample["theta"]
        assert isinstance(theta, np.ndarray)
        assert theta.shape == (5,)

    def test_vector_parameter_flat_names(self) -> None:
        """Test flat_param_names expands vector params."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("theta", dist.Normal(0, 1), size=3)
            p("sigma", dist.HalfNormal(1))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        expected = ["mu", "theta[0]", "theta[1]", "theta[2]", "sigma"]
        assert model.flat_param_names == expected

    def test_flatten_unflatten_roundtrip(self) -> None:
        """Test flatten/unflatten preserves values."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("theta", dist.Normal(0, 1), size=3)
            p("sigma", dist.HalfNormal(1))

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        params: StructuredParams = {
            "mu": 1.5,
            "theta": np.array([0.1, 0.2, 0.3]),
            "sigma": 2.0,
        }

        flat = model.to_flat_unconstrained(params)
        recovered = model.from_flat_unconstrained(flat)

        np.testing.assert_allclose(recovered["mu"], params["mu"], rtol=1e-10)
        np.testing.assert_allclose(recovered["sigma"], params["sigma"], rtol=1e-10)
        np.testing.assert_allclose(recovered["theta"], params["theta"], rtol=1e-10)

    def test_hierarchical_log_prior(self) -> None:
        """Test log_prior with hierarchical dependencies."""

        def priors(p: ParamContext) -> None:
            mu = p("mu", dist.Normal(0, 5))
            sigma = p("sigma", dist.HalfNormal(1))
            p("theta", dist.Normal(mu, sigma), size=3)

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        # Create params where theta is drawn from Normal(mu, sigma)
        params: StructuredParams = {
            "mu": 2.0,
            "sigma": 0.5,
            "theta": np.array([1.8, 2.1, 2.3]),
        }

        lp = model.log_prior(params)

        # Manual calculation
        lp_mu: float = float(stats.norm(0, 5).logpdf(2.0))
        lp_sigma: float = float(stats.halfnorm(scale=1).logpdf(0.5))
        # theta[i] ~ Normal(2.0, 0.5)
        lp_theta: float = float(np.sum(stats.norm(2.0, 0.5).logpdf(params["theta"])))
        expected = lp_mu + lp_sigma + lp_theta

        np.testing.assert_allclose(lp, expected, rtol=1e-10)

    def test_hierarchical_log_prob_unconstrained(self) -> None:
        """Test log_prob_unconstrained with vector params."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("theta", dist.Normal(0, 1), size=2)

        def zero_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            del params, data
            return np.array([0.0], dtype=np.float64)

        model = Model(priors=priors, log_likelihood=zero_likelihood)

        params: StructuredParams = {
            "mu": 1.0,
            "theta": np.array([0.5, -0.5]),
        }

        flat = model.to_flat_unconstrained(params)
        lp_unc = model.log_prob_unconstrained(flat, None)

        # For REAL support params, no Jacobian correction
        # So should equal log_prior
        lp_prior = model.log_prior(params)
        np.testing.assert_allclose(lp_unc, lp_prior, rtol=1e-10)

    def test_param_info_property(self) -> None:
        """Test param_info returns correct metadata."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("theta", dist.Normal(0, 1), size=5)

        model = Model(priors=priors, log_likelihood=simple_log_likelihood)

        info = model.param_info

        assert info["mu"].is_vector is False
        assert info["mu"].size == 1
        assert isinstance(info["mu"].distribution, dist.Normal)

        assert info["theta"].is_vector is True
        assert info["theta"].size == 5
        assert isinstance(info["theta"].distribution, dist.Normal)

    def test_eight_schools_structure(self) -> None:
        """Test Eight Schools-like hierarchical structure."""
        J = 8  # Number of schools

        def priors(p: ParamContext) -> None:
            mu = p("mu", dist.Normal(0, 5))
            tau = p("tau", dist.HalfNormal(5))
            p("theta", dist.Normal(mu, tau), size=J)

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            y, sigma = data
            theta = params["theta"]
            assert isinstance(theta, np.ndarray)
            # Pointwise Normal log-probs
            ll: NDArray[np.float64] = np.zeros(J, dtype=np.float64)
            for j in range(J):
                ll[j] = float(dist.Normal(theta[j], sigma[j]).log_prob(y[j]))
            return ll

        model = Model(priors=priors, log_likelihood=log_likelihood)

        # Check structure
        assert model.param_names == ["mu", "tau", "theta"]
        assert len(model.flat_param_names) == 2 + J  # mu, tau, theta[0-7]

        # Sample and check shapes
        sample = model.sample_prior(rng=np.random.default_rng(42))
        assert isinstance(sample["mu"], float)
        assert isinstance(sample["tau"], float)
        assert sample["tau"] > 0
        assert isinstance(sample["theta"], np.ndarray)
        assert sample["theta"].shape == (J,)

        # Test log_prob computation
        y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
        sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)
        data = (y, sigma)

        lp = model.log_prob(sample, data)
        assert np.isfinite(lp)
