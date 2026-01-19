"""Tests for TruncatedNormal distribution."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from minibayes import dist
from minibayes.distributions import Support
from minibayes.exceptions import ModelSpecError


class TestTruncatedNormal:
    """Tests for TruncatedNormal distribution."""

    def test_log_prob_matches_scipy_two_sided(self) -> None:
        """log_prob matches scipy.stats reference for two-sided truncation."""
        mu, sigma = 0.5, 1.0
        lower, upper = 0.2, 5.0
        d = dist.TruncatedNormal(mu=mu, sigma=sigma, lower=lower, upper=upper)

        # scipy truncnorm uses standardized bounds
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        scipy_d = stats.truncnorm(a, b, loc=mu, scale=sigma)

        x: NDArray[np.float64] = np.array([0.3, 0.5, 1.0, 2.0, 4.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_log_prob_matches_scipy_lower_bounded(self) -> None:
        """log_prob matches scipy.stats reference for lower-bounded only."""
        mu, sigma = 0.5, 1.0
        lower = 0.2
        d = dist.TruncatedNormal(mu=mu, sigma=sigma, lower=lower)

        # scipy truncnorm: use very large upper bound
        a = (lower - mu) / sigma
        b = np.inf
        scipy_d = stats.truncnorm(a, b, loc=mu, scale=sigma)

        x: NDArray[np.float64] = np.array([0.3, 0.5, 1.0, 2.0, 5.0])
        result: NDArray[np.float64] = np.asarray(d.log_prob(x))
        expected: NDArray[np.float64] = scipy_d.logpdf(x)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_log_prob_outside_bounds_is_neg_inf(self) -> None:
        """log_prob returns -inf outside truncation bounds."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2, upper=2.0)
        assert d.log_prob(0.1) == float("-inf")
        assert d.log_prob(2.5) == float("-inf")

    def test_log_prob_scalar(self) -> None:
        """log_prob works with scalar input."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2)
        result = d.log_prob(0.6)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_sample_within_bounds_two_sided(self) -> None:
        """Samples are within bounds for two-sided truncation."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2, upper=2.0)
        rng = np.random.default_rng(42)
        samples = d.sample(size=1000, rng=rng)
        assert np.all(samples >= 0.2)
        assert np.all(samples <= 2.0)

    def test_sample_within_bounds_lower_only(self) -> None:
        """Samples respect lower bound when only lower is specified."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2)
        rng = np.random.default_rng(42)
        samples = d.sample(size=1000, rng=rng)
        assert np.all(samples >= 0.2)

    def test_sample_shape_none(self) -> None:
        """sample returns scalar when size=None."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2)
        rng = np.random.default_rng(42)
        sample = d.sample(size=None, rng=rng)
        assert isinstance(sample, float)

    def test_sample_shape_int(self) -> None:
        """sample returns correct shape for integer size."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2)
        rng = np.random.default_rng(42)
        samples = d.sample(size=100, rng=rng)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (100,)

    def test_sample_shape_tuple(self) -> None:
        """sample returns correct shape for tuple size."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2)
        rng = np.random.default_rng(42)
        samples = d.sample(size=(10, 5), rng=rng)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (10, 5)

    def test_support(self) -> None:
        """Support is BOUNDED."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2, upper=2.0)
        assert d.support == Support.BOUNDED

    def test_invalid_sigma_raises(self) -> None:
        """Invalid sigma raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.TruncatedNormal(mu=0.0, sigma=-1.0, lower=0.0)
        with pytest.raises(ModelSpecError):
            dist.TruncatedNormal(mu=0.0, sigma=0.0, lower=0.0)

    def test_invalid_bounds_raises(self) -> None:
        """Invalid bounds (lower >= upper) raises ModelSpecError."""
        with pytest.raises(ModelSpecError):
            dist.TruncatedNormal(mu=0.0, sigma=1.0, lower=1.0, upper=0.0)
        with pytest.raises(ModelSpecError):
            dist.TruncatedNormal(mu=0.0, sigma=1.0, lower=1.0, upper=1.0)

    def test_properties(self) -> None:
        """Properties return expected values."""
        d = dist.TruncatedNormal(mu=0.5, sigma=2.0, lower=0.2, upper=5.0)
        assert d.mu == 0.5
        assert d.sigma == 2.0
        assert d.lower == 0.2
        assert d.upper == 5.0

    def test_default_bounds(self) -> None:
        """Default bounds are -inf and +inf."""
        d = dist.TruncatedNormal(mu=0.0, sigma=1.0)
        assert d.lower == float("-inf")
        assert d.upper == float("inf")

    def test_default_transform_two_sided(self) -> None:
        """default_transform returns AffineTransform for two-sided bounds."""
        from minibayes.transforms import AffineTransform

        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2, upper=2.0)
        transform = d.default_transform()
        assert isinstance(transform, AffineTransform)

    def test_default_transform_lower_bounded(self) -> None:
        """default_transform returns ShiftedLogTransform for lower-bounded only."""
        from minibayes.transforms import ShiftedLogTransform

        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2)
        transform = d.default_transform()
        assert isinstance(transform, ShiftedLogTransform)
        assert transform.lower == 0.2

    def test_transform_roundtrip(self) -> None:
        """Transform roundtrip preserves values."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2, upper=5.0)
        transform = d.default_transform()
        x: NDArray[np.float64] = np.array([0.5, 1.0, 2.0, 4.0])
        y = transform.forward(x)
        x_back = transform.inverse(y)
        np.testing.assert_allclose(x_back, x, rtol=1e-10)

    def test_shifted_log_transform_roundtrip(self) -> None:
        """ShiftedLogTransform roundtrip preserves values for lower-bounded."""
        d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2)
        transform = d.default_transform()
        x: NDArray[np.float64] = np.array([0.3, 0.5, 1.0, 5.0, 10.0])
        y = transform.forward(x)
        x_back = transform.inverse(y)
        np.testing.assert_allclose(x_back, x, rtol=1e-10)

    def test_sample_matches_scipy_distribution(self) -> None:
        """Sample distribution approximately matches scipy reference."""
        mu, sigma = 0.5, 1.0
        lower, upper = 0.2, 5.0
        d = dist.TruncatedNormal(mu=mu, sigma=sigma, lower=lower, upper=upper)

        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        scipy_d = stats.truncnorm(a, b, loc=mu, scale=sigma)

        rng = np.random.default_rng(42)
        samples = d.sample(size=10000, rng=rng)

        # Check sample mean and std are close to theoretical values
        expected_mean = scipy_d.mean()
        expected_std = scipy_d.std()

        np.testing.assert_allclose(np.mean(samples), expected_mean, rtol=0.05)
        np.testing.assert_allclose(np.std(samples), expected_std, rtol=0.1)


class TestShiftedLogTransform:
    """Tests for ShiftedLogTransform."""

    def test_forward(self) -> None:
        """forward transforms correctly."""
        from minibayes.transforms import ShiftedLogTransform

        t = ShiftedLogTransform(lower=0.2)
        x: NDArray[np.float64] = np.array([0.5, 1.0, 2.0])
        y = t.forward(x)
        expected: NDArray[np.float64] = np.log(x - 0.2)
        np.testing.assert_allclose(y, expected, rtol=1e-10)

    def test_inverse(self) -> None:
        """inverse transforms correctly."""
        from minibayes.transforms import ShiftedLogTransform

        t = ShiftedLogTransform(lower=0.2)
        y: NDArray[np.float64] = np.array([-1.0, 0.0, 1.0, 2.0])
        x = t.inverse(y)
        expected: NDArray[np.float64] = np.exp(y) + 0.2
        np.testing.assert_allclose(x, expected, rtol=1e-10)

    def test_roundtrip(self) -> None:
        """forward then inverse gives back original values."""
        from minibayes.transforms import ShiftedLogTransform

        t = ShiftedLogTransform(lower=0.5)
        x: NDArray[np.float64] = np.array([0.6, 1.0, 2.0, 10.0])
        y = t.forward(x)
        x_back = t.inverse(y)
        np.testing.assert_allclose(x_back, x, rtol=1e-10)

    def test_log_det_jacobian(self) -> None:
        """log_det_jacobian is log(x - lower)."""
        from minibayes.transforms import ShiftedLogTransform

        t = ShiftedLogTransform(lower=0.2)
        x: NDArray[np.float64] = np.array([0.5, 1.0, 2.0])
        ldj = t.log_det_jacobian(x)
        expected: NDArray[np.float64] = np.log(x - 0.2)
        np.testing.assert_allclose(ldj, expected, rtol=1e-10)

    def test_lower_property(self) -> None:
        """lower property returns the bound."""
        from minibayes.transforms import ShiftedLogTransform

        t = ShiftedLogTransform(lower=1.5)
        assert t.lower == 1.5
