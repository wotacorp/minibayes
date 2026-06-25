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


class TestMultivariateNormal:
    """Tests for MultivariateNormal distribution."""

    def test_log_prob_matches_scipy(self) -> None:
        """log_prob matches scipy.stats reference for 2D."""
        mean = np.array([1.0, 2.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        d = dist.MultivariateNormal(mean=mean, cov=cov)
        scipy_d = stats.multivariate_normal(mean=mean, cov=cov)
        x: NDArray[np.float64] = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 3.0]])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_log_prob_single_observation(self) -> None:
        """log_prob with single observation returns float."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        d = dist.MultivariateNormal(mean=mean, cov=cov)
        result = d.log_prob(np.array([0.0, 0.0]))
        assert isinstance(result, float)

    def test_log_prob_batch(self) -> None:
        """log_prob with batch returns array."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        d = dist.MultivariateNormal(mean=mean, cov=cov)
        x: NDArray[np.float64] = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = d.log_prob(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_sample_shape_none(self) -> None:
        """sample with size=None returns shape (d,)."""
        mean = np.array([0.0, 0.0, 0.0])
        cov = np.eye(3)
        d = dist.MultivariateNormal(mean=mean, cov=cov)
        rng = np.random.default_rng(42)
        sample = d.sample(size=None, rng=rng)
        assert sample.shape == (3,)

    def test_sample_shape_int(self) -> None:
        """sample with size=n returns shape (n, d)."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        d = dist.MultivariateNormal(mean=mean, cov=cov)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert samples.shape == (100, 2)

    def test_invalid_mean_not_1d_raises(self) -> None:
        """Raises if mean is not 1D."""
        mean = np.array([[0.0, 0.0]])
        cov = np.eye(2)
        with pytest.raises(ModelSpecError, match="mean must be 1D"):
            dist.MultivariateNormal(mean=mean, cov=cov)

    def test_invalid_cov_dimension_raises(self) -> None:
        """Raises if cov dimension doesn't match mean."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(3)
        with pytest.raises(ModelSpecError, match="cov must be"):
            dist.MultivariateNormal(mean=mean, cov=cov)

    def test_invalid_cov_not_pd_raises(self) -> None:
        """Raises if cov is not positive definite."""
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not PD
        with pytest.raises(ModelSpecError, match="positive definite"):
            dist.MultivariateNormal(mean=mean, cov=cov)


class TestLKJCholesky:
    """Tests for LKJCholesky distribution."""

    def test_log_prob_identity(self) -> None:
        """Identity Cholesky has valid log_prob."""
        d = dist.LKJCholesky(dim=3, eta=2.0)
        L = np.eye(3)
        result = d.log_prob(L)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_sample_valid_cholesky(self) -> None:
        """Samples are lower triangular with positive diagonal."""
        d = dist.LKJCholesky(dim=3, eta=1.0)
        rng = np.random.default_rng(42)
        L: NDArray[np.float64] = d.sample(rng=rng)

        # Check lower triangular (upper triangle is zero)
        np.testing.assert_allclose(np.triu(L, k=1), 0.0)

        # Check positive diagonal
        assert np.all(np.diag(L) > 0)

    def test_sample_valid_correlation(self) -> None:
        """L @ L.T is valid correlation matrix (diag=1, |off|<1)."""
        d = dist.LKJCholesky(dim=3, eta=1.0)
        rng = np.random.default_rng(42)
        L: NDArray[np.float64] = d.sample(rng=rng)
        corr: NDArray[np.float64] = L @ L.T

        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

        # Off-diagonal should be in (-1, 1)
        off_diag = corr[np.triu_indices(3, k=1)]
        assert np.all(np.abs(off_diag) < 1)

    def test_transform_roundtrip(self) -> None:
        """inverse(forward(L)) == L for valid Cholesky."""
        d = dist.LKJCholesky(dim=3, eta=2.0)
        transform = d.default_transform()
        rng = np.random.default_rng(42)
        L: NDArray[np.float64] = d.sample(rng=rng)

        # Forward then inverse should recover L
        y = transform.forward(L)
        L_recovered = transform.inverse(y)
        np.testing.assert_allclose(L_recovered, L, atol=1e-6)

    def test_sample_shape(self) -> None:
        """Verify sample shapes."""
        d = dist.LKJCholesky(dim=2, eta=1.0)
        rng = np.random.default_rng(42)

        # Single sample
        single = d.sample(rng=rng)
        assert single.shape == (2, 2)

        # Multiple samples
        batch = d.sample(size=5, rng=rng)
        assert batch.shape == (5, 2, 2)

    def test_invalid_inputs_raise(self) -> None:
        """dim<2 or eta<=0 raises ModelSpecError."""
        with pytest.raises(ModelSpecError, match="dim must be >= 2"):
            dist.LKJCholesky(dim=1, eta=1.0)

        with pytest.raises(ModelSpecError, match="eta must be positive"):
            dist.LKJCholesky(dim=2, eta=0.0)

        with pytest.raises(ModelSpecError, match="eta must be positive"):
            dist.LKJCholesky(dim=2, eta=-1.0)

    def test_marginal_beta_distribution(self) -> None:
        """Off-diagonal correlations follow predicted Beta distribution."""
        from scipy import stats

        lkj = dist.LKJCholesky(dim=3, eta=2.0)
        rng = np.random.default_rng(42)

        # Sample many correlation matrices
        correlations: list[float] = []
        for _ in range(5000):
            L: NDArray[np.float64] = lkj.sample(rng=rng)
            R: NDArray[np.float64] = L @ L.T
            r_12: float = float(R[0, 1])  # type: ignore[misc]
            correlations.append(r_12)

        # Transform to (0, 1) and test against Beta
        transformed: list[float] = [(r + 1) / 2 for r in correlations]
        alpha: float = 2.0 + (3 - 2) / 2  # eta + (d-2)/2 = 2.5

        # Kolmogorov-Smirnov test
        result = stats.kstest(transformed, "beta", args=(alpha, alpha))
        assert result.pvalue > 0.01, f"KS test failed: p={result.pvalue}"

    def test_density_formula_matches_diagonal(self) -> None:
        """log_prob computed via method matches formula from diagonal."""
        lkj = dist.LKJCholesky(dim=3, eta=2.0)
        rng = np.random.default_rng(42)

        for _ in range(100):
            L: NDArray[np.float64] = lkj.sample(rng=rng)

            # Compute via method
            lp_method: float = float(lkj.log_prob(L))

            # Compute via formula: sum_{k=1}^{d-1} (d - (k+1) + 2*eta - 2) * log(L[k,k])
            lp_formula: float = 0.0
            d: int = 3
            eta: float = 2.0
            for k in range(1, d):
                exponent: float = d - (k + 1) + 2 * eta - 2
                lp_formula += exponent * float(np.log(L[k, k]))

            np.testing.assert_allclose(lp_method, lp_formula, rtol=1e-10)

    def test_eta_controls_concentration(self) -> None:
        """Higher eta -> correlations closer to zero."""
        rng = np.random.default_rng(42)

        std_devs: dict[float, float] = {}
        for eta in [0.5, 1.0, 2.0, 5.0]:
            lkj = dist.LKJCholesky(dim=3, eta=eta)
            correlations: list[float] = []
            for _ in range(1000):
                L: NDArray[np.float64] = lkj.sample(rng=rng)
                R: NDArray[np.float64] = L @ L.T
                correlations.append(float(R[0, 1]))  # type: ignore[misc]
            std_devs[eta] = float(np.std(correlations))

        # Higher eta should have smaller std (more concentrated around 0)
        assert std_devs[0.5] > std_devs[1.0] > std_devs[2.0] > std_devs[5.0]

    def test_jacobian_sign_correctness(self) -> None:
        """Jacobian has correct sign per minibayes convention."""
        from minibayes.transforms.corr_cholesky import CorrCholeskyTransform

        transform = CorrCholeskyTransform(dim=2)

        # Create a valid Cholesky factor
        L: NDArray[np.float64] = np.array([[1.0, 0.0], [0.5, np.sqrt(0.75)]])

        # Numerical differentiation
        eps: float = 1e-7
        y: NDArray[np.float64] = transform.forward(L)

        # For 2x2, there's 1 off-diagonal element
        # Perturb y and measure change in L
        y_plus: NDArray[np.float64] = y + eps
        y_minus: NDArray[np.float64] = y - eps
        L_plus: NDArray[np.float64] = transform.inverse(y_plus)
        L_minus: NDArray[np.float64] = transform.inverse(y_minus)

        # Numerical Jacobian dL/dy (for the single off-diagonal)
        numerical_deriv: float = float((L_plus[1, 0] - L_minus[1, 0]) / (2 * eps))

        # Analytical log|dL/dy|
        log_det: NDArray[np.float64] = transform.log_det_jacobian(L)
        analytical_deriv: float = float(np.exp(float(log_det)))

        # Should match (within numerical tolerance)
        np.testing.assert_allclose(abs(numerical_deriv), analytical_deriv, rtol=1e-4)
