"""Tests for probability distributions."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from minibayes import dist
from minibayes.distributions import Support
from minibayes.exceptions import ModelSpecError


class TestNormal:
    """Tests for Normal distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        d = dist.Normal(loc=2.0, scale=3.0)
        scipy_d = stats.norm(loc=2.0, scale=3.0)
        x: NDArray[np.float64] = np.array([-1.0, 0.0, 2.0, 5.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_scalar(self) -> None:
        """log_prob works with scalar input."""
        d = dist.Normal(loc=0.0, scale=1.0)
        result = d.log_prob(0.0)
        assert isinstance(result, float)

    def test_sample_shape(self) -> None:
        """sample returns correct shape."""
        d = dist.Normal(loc=0.0, scale=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (100,)

    def test_support(self) -> None:
        """Support is REAL."""
        d = dist.Normal(loc=0.0, scale=1.0)
        assert d.support == Support.REAL

    def test_invalid_scale_raises(self) -> None:
        """Invalid scale raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.Normal(loc=0.0, scale=-1.0)

    def test_mean(self) -> None:
        """Mean equals loc parameter."""
        d = dist.Normal(loc=2.5, scale=3.0)
        assert d.mean == 2.5

    def test_obs_logp(self) -> None:
        """obs_logp returns sum of log_prob as float."""
        d = dist.Normal(loc=0.0, scale=1.0)
        data: NDArray[np.float64] = np.array([0.0, 1.0, -1.0])
        result = d.obs_logp(data)
        expected = float(np.sum(d.log_prob(data)))
        assert isinstance(result, float)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_obs_logp_scalar(self) -> None:
        """obs_logp works with scalar input."""
        d = dist.Normal(loc=0.0, scale=1.0)
        result = d.obs_logp(0.5)
        assert isinstance(result, float)
        np.testing.assert_allclose(result, d.log_prob(0.5), rtol=1e-10)


class TestHalfNormal:
    """Tests for HalfNormal distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        d = dist.HalfNormal(scale=2.0)
        scipy_d = stats.halfnorm(scale=2.0)
        x: NDArray[np.float64] = np.array([0.1, 1.0, 2.0, 5.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_negative_is_neg_inf(self) -> None:
        """log_prob returns -inf for negative values."""
        d = dist.HalfNormal(scale=1.0)
        assert d.log_prob(-1.0) == float("-inf")
        assert d.log_prob(0.0) == float("-inf")

    def test_sample_positive(self) -> None:
        """Samples are positive."""
        d = dist.HalfNormal(scale=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all(samples > 0)

    def test_support(self) -> None:
        """Support is POSITIVE."""
        d = dist.HalfNormal(scale=1.0)
        assert d.support == Support.POSITIVE

    def test_mean(self) -> None:
        """Mean equals scale * sqrt(2/pi)."""
        scale = 2.0
        d = dist.HalfNormal(scale=scale)
        expected = scale * np.sqrt(2.0 / np.pi)
        np.testing.assert_allclose(d.mean, expected, rtol=1e-10)


class TestExponential:
    """Tests for Exponential distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        rate = 2.0
        d = dist.Exponential(rate=rate)
        scipy_d = stats.expon(scale=1.0 / rate)
        x: NDArray[np.float64] = np.array([0.1, 0.5, 1.0, 2.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_negative_is_neg_inf(self) -> None:
        """log_prob returns -inf for non-positive values."""
        d = dist.Exponential(rate=1.0)
        assert d.log_prob(-1.0) == float("-inf")
        assert d.log_prob(0.0) == float("-inf")

    def test_sample_positive(self) -> None:
        """Samples are positive."""
        d = dist.Exponential(rate=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all(samples > 0)

    def test_support(self) -> None:
        """Support is POSITIVE."""
        d = dist.Exponential(rate=1.0)
        assert d.support == Support.POSITIVE

    def test_mean(self) -> None:
        """Mean equals 1/rate."""
        rate = 2.0
        d = dist.Exponential(rate=rate)
        expected = 1.0 / rate
        np.testing.assert_allclose(d.mean, expected, rtol=1e-10)


class TestGamma:
    """Tests for Gamma distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        shape, rate = 2.0, 3.0
        d = dist.Gamma(shape=shape, rate=rate)
        scipy_d = stats.gamma(a=shape, scale=1.0 / rate)
        x: NDArray[np.float64] = np.array([0.1, 0.5, 1.0, 2.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_negative_is_neg_inf(self) -> None:
        """log_prob returns -inf for non-positive values."""
        d = dist.Gamma(shape=2.0, rate=1.0)
        assert d.log_prob(-1.0) == float("-inf")
        assert d.log_prob(0.0) == float("-inf")

    def test_sample_positive(self) -> None:
        """Samples are positive."""
        d = dist.Gamma(shape=2.0, rate=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all(samples > 0)

    def test_support(self) -> None:
        """Support is POSITIVE."""
        d = dist.Gamma(shape=2.0, rate=1.0)
        assert d.support == Support.POSITIVE

    def test_mean(self) -> None:
        """Mean equals shape/rate."""
        shape, rate = 3.0, 2.0
        d = dist.Gamma(shape=shape, rate=rate)
        expected = shape / rate
        np.testing.assert_allclose(d.mean, expected, rtol=1e-10)


class TestBeta:
    """Tests for Beta distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        alpha, beta_param = 2.0, 5.0
        d = dist.Beta(alpha=alpha, beta=beta_param)
        scipy_d = stats.beta(a=alpha, b=beta_param)
        x: NDArray[np.float64] = np.array([0.1, 0.3, 0.5, 0.9])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_boundaries_neg_inf(self) -> None:
        """log_prob returns -inf at and outside boundaries."""
        d = dist.Beta(alpha=2.0, beta=2.0)
        assert d.log_prob(0.0) == float("-inf")
        assert d.log_prob(1.0) == float("-inf")
        assert d.log_prob(-0.1) == float("-inf")
        assert d.log_prob(1.1) == float("-inf")

    def test_sample_in_unit_interval(self) -> None:
        """Samples are in (0, 1)."""
        d = dist.Beta(alpha=2.0, beta=2.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all((samples > 0) & (samples < 1))

    def test_support(self) -> None:
        """Support is UNIT."""
        d = dist.Beta(alpha=2.0, beta=2.0)
        assert d.support == Support.UNIT

    def test_mean(self) -> None:
        """Mean equals alpha/(alpha+beta)."""
        alpha, beta_param = 2.0, 5.0
        d = dist.Beta(alpha=alpha, beta=beta_param)
        expected = alpha / (alpha + beta_param)
        np.testing.assert_allclose(d.mean, expected, rtol=1e-10)


class TestUniform:
    """Tests for Uniform distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        low, high = 2.0, 5.0
        d = dist.Uniform(low=low, high=high)
        scipy_d = stats.uniform(loc=low, scale=high - low)
        x: NDArray[np.float64] = np.array([2.5, 3.0, 4.5])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_outside_bounds_neg_inf(self) -> None:
        """log_prob returns -inf outside [low, high]."""
        d = dist.Uniform(low=0.0, high=1.0)
        assert d.log_prob(-0.1) == float("-inf")
        assert d.log_prob(1.1) == float("-inf")

    def test_log_prob_at_bounds(self) -> None:
        """log_prob is valid at boundaries (closed interval)."""
        d = dist.Uniform(low=0.0, high=1.0)
        expected = -np.log(1.0)
        assert d.log_prob(0.0) == expected
        assert d.log_prob(1.0) == expected

    def test_sample_in_bounds(self) -> None:
        """Samples are in [low, high]."""
        d = dist.Uniform(low=2.0, high=5.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all((samples >= 2.0) & (samples <= 5.0))

    def test_support(self) -> None:
        """Support is BOUNDED."""
        d = dist.Uniform(low=0.0, high=1.0)
        assert d.support == Support.BOUNDED

    def test_default_transform(self) -> None:
        """default_transform returns AffineTransform with correct bounds."""
        from minibayes.transforms import AffineTransform

        d = dist.Uniform(low=2.0, high=5.0)
        transform = d.default_transform()
        assert isinstance(transform, AffineTransform)
        assert transform.low == 2.0
        assert transform.high == 5.0

    def test_mean(self) -> None:
        """Mean equals (low+high)/2."""
        low, high = 2.0, 8.0
        d = dist.Uniform(low=low, high=high)
        expected = (low + high) / 2.0
        np.testing.assert_allclose(d.mean, expected, rtol=1e-10)
