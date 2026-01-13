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


class TestStudentT:
    """Tests for StudentT distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        df, loc, scale = 5.0, 2.0, 3.0
        d = dist.StudentT(df=df, loc=loc, scale=scale)
        scipy_d = stats.t(df=df, loc=loc, scale=scale)
        x: NDArray[np.float64] = np.array([-2.0, 0.0, 2.0, 5.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_scalar(self) -> None:
        """log_prob works with scalar input."""
        d = dist.StudentT(df=3.0, loc=0.0, scale=1.0)
        result = d.log_prob(0.0)
        assert isinstance(result, float)

    def test_sample_shape(self) -> None:
        """sample returns correct shape."""
        d = dist.StudentT(df=5.0, loc=0.0, scale=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (100,)

    def test_support(self) -> None:
        """Support is REAL."""
        d = dist.StudentT(df=5.0, loc=0.0, scale=1.0)
        assert d.support == Support.REAL

    def test_mean(self) -> None:
        """Mean equals loc when df > 1."""
        d = dist.StudentT(df=5.0, loc=2.5, scale=1.0)
        assert d.mean == 2.5

    def test_mean_undefined_for_df_le_1(self) -> None:
        """Mean is NaN when df <= 1."""
        d = dist.StudentT(df=1.0, loc=0.0, scale=1.0)
        assert np.isnan(d.mean)

    def test_invalid_df_raises(self) -> None:
        """Invalid df raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.StudentT(df=-1.0)

    def test_invalid_scale_raises(self) -> None:
        """Invalid scale raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.StudentT(df=5.0, scale=-1.0)


class TestLogNormal:
    """Tests for LogNormal distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        loc, scale = 1.0, 0.5
        d = dist.LogNormal(loc=loc, scale=scale)
        scipy_d = stats.lognorm(s=scale, scale=np.exp(loc))
        x: NDArray[np.float64] = np.array([0.5, 1.0, 2.0, 5.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_negative_is_neg_inf(self) -> None:
        """log_prob returns -inf for non-positive values."""
        d = dist.LogNormal(loc=0.0, scale=1.0)
        assert d.log_prob(-1.0) == float("-inf")
        assert d.log_prob(0.0) == float("-inf")

    def test_sample_positive(self) -> None:
        """Samples are positive."""
        d = dist.LogNormal(loc=0.0, scale=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all(samples > 0)

    def test_support(self) -> None:
        """Support is POSITIVE."""
        d = dist.LogNormal(loc=0.0, scale=1.0)
        assert d.support == Support.POSITIVE

    def test_mean(self) -> None:
        """Mean equals exp(loc + scale^2/2)."""
        loc, scale = 1.0, 0.5
        d = dist.LogNormal(loc=loc, scale=scale)
        expected = np.exp(loc + 0.5 * scale * scale)
        np.testing.assert_allclose(d.mean, expected, rtol=1e-10)

    def test_invalid_scale_raises(self) -> None:
        """Invalid scale raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.LogNormal(scale=-1.0)


class TestCauchy:
    """Tests for Cauchy distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        loc, scale = 2.0, 3.0
        d = dist.Cauchy(loc=loc, scale=scale)
        scipy_d = stats.cauchy(loc=loc, scale=scale)
        x: NDArray[np.float64] = np.array([-5.0, 0.0, 2.0, 10.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_scalar(self) -> None:
        """log_prob works with scalar input."""
        d = dist.Cauchy(loc=0.0, scale=1.0)
        result = d.log_prob(0.0)
        assert isinstance(result, float)

    def test_sample_shape(self) -> None:
        """sample returns correct shape."""
        d = dist.Cauchy(loc=0.0, scale=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (100,)

    def test_support(self) -> None:
        """Support is REAL."""
        d = dist.Cauchy(loc=0.0, scale=1.0)
        assert d.support == Support.REAL

    def test_mean_is_nan(self) -> None:
        """Mean is NaN (undefined for Cauchy)."""
        d = dist.Cauchy(loc=0.0, scale=1.0)
        assert np.isnan(d.mean)

    def test_invalid_scale_raises(self) -> None:
        """Invalid scale raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.Cauchy(scale=-1.0)


class TestInverseGamma:
    """Tests for InverseGamma distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        shape, scale = 3.0, 2.0
        d = dist.InverseGamma(shape=shape, scale=scale)
        scipy_d = stats.invgamma(a=shape, scale=scale)
        x: NDArray[np.float64] = np.array([0.5, 1.0, 2.0, 5.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_negative_is_neg_inf(self) -> None:
        """log_prob returns -inf for non-positive values."""
        d = dist.InverseGamma(shape=2.0, scale=1.0)
        assert d.log_prob(-1.0) == float("-inf")
        assert d.log_prob(0.0) == float("-inf")

    def test_sample_positive(self) -> None:
        """Samples are positive."""
        d = dist.InverseGamma(shape=2.0, scale=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all(samples > 0)

    def test_support(self) -> None:
        """Support is POSITIVE."""
        d = dist.InverseGamma(shape=2.0, scale=1.0)
        assert d.support == Support.POSITIVE

    def test_mean(self) -> None:
        """Mean equals scale/(shape-1) when shape > 1."""
        shape, scale = 3.0, 4.0
        d = dist.InverseGamma(shape=shape, scale=scale)
        expected = scale / (shape - 1)
        np.testing.assert_allclose(d.mean, expected, rtol=1e-10)

    def test_mean_infinite_for_shape_le_1(self) -> None:
        """Mean is infinite when shape <= 1."""
        d = dist.InverseGamma(shape=1.0, scale=1.0)
        assert d.mean == float("inf")

    def test_invalid_shape_raises(self) -> None:
        """Invalid shape raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.InverseGamma(shape=-1.0)

    def test_invalid_scale_raises(self) -> None:
        """Invalid scale raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.InverseGamma(shape=2.0, scale=-1.0)


class TestLaplace:
    """Tests for Laplace distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        loc, scale = 2.0, 3.0
        d = dist.Laplace(loc=loc, scale=scale)
        scipy_d = stats.laplace(loc=loc, scale=scale)
        x: NDArray[np.float64] = np.array([-5.0, 0.0, 2.0, 10.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_scalar(self) -> None:
        """log_prob works with scalar input."""
        d = dist.Laplace(loc=0.0, scale=1.0)
        result = d.log_prob(0.0)
        assert isinstance(result, float)

    def test_sample_shape(self) -> None:
        """sample returns correct shape."""
        d = dist.Laplace(loc=0.0, scale=1.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (100,)

    def test_support(self) -> None:
        """Support is REAL."""
        d = dist.Laplace(loc=0.0, scale=1.0)
        assert d.support == Support.REAL

    def test_mean(self) -> None:
        """Mean equals loc."""
        d = dist.Laplace(loc=2.5, scale=1.0)
        assert d.mean == 2.5

    def test_invalid_scale_raises(self) -> None:
        """Invalid scale raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.Laplace(scale=-1.0)


class TestBernoulli:
    """Tests for Bernoulli distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        prob = 0.7
        d = dist.Bernoulli(prob=prob)
        scipy_d = stats.bernoulli(p=prob)
        x: NDArray[np.float64] = np.array([0.0, 1.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpmf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_invalid_is_neg_inf(self) -> None:
        """log_prob returns -inf for values not in {0, 1}."""
        d = dist.Bernoulli(prob=0.5)
        assert d.log_prob(-1.0) == float("-inf")
        assert d.log_prob(0.5) == float("-inf")
        assert d.log_prob(2.0) == float("-inf")

    def test_sample_binary(self) -> None:
        """Samples are in {0, 1}."""
        d = dist.Bernoulli(prob=0.5)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all((samples == 0) | (samples == 1))

    def test_support(self) -> None:
        """Support is BINARY."""
        d = dist.Bernoulli(prob=0.5)
        assert d.support == Support.BINARY

    def test_mean(self) -> None:
        """Mean equals prob."""
        d = dist.Bernoulli(prob=0.7)
        assert d.mean == 0.7

    def test_invalid_prob_raises(self) -> None:
        """Invalid prob raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.Bernoulli(prob=-0.1)
        with pytest.raises(ModelSpecError):
            dist.Bernoulli(prob=1.1)


class TestPoisson:
    """Tests for Poisson distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference."""
        rate = 3.0
        d = dist.Poisson(rate=rate)
        scipy_d = stats.poisson(mu=rate)
        x: NDArray[np.float64] = np.array([0.0, 1.0, 3.0, 5.0, 10.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpmf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_invalid_is_neg_inf(self) -> None:
        """log_prob returns -inf for non-natural numbers."""
        d = dist.Poisson(rate=3.0)
        assert d.log_prob(-1.0) == float("-inf")
        assert d.log_prob(1.5) == float("-inf")

    def test_sample_natural(self) -> None:
        """Samples are non-negative integers."""
        d = dist.Poisson(rate=3.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert np.all(samples >= 0)
        assert np.all(samples == np.floor(samples))

    def test_support(self) -> None:
        """Support is NATURAL."""
        d = dist.Poisson(rate=3.0)
        assert d.support == Support.NATURAL

    def test_mean(self) -> None:
        """Mean equals rate."""
        d = dist.Poisson(rate=5.0)
        assert d.mean == 5.0

    def test_invalid_rate_raises(self) -> None:
        """Invalid rate raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.Poisson(rate=-1.0)
        with pytest.raises(ModelSpecError):
            dist.Poisson(rate=0.0)
