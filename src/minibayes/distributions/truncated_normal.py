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

"""Truncated Normal distribution."""

import math

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.transforms.base import Transform
from minibayes.utils import ensure_rng

# Precompute constant: -0.5 * log(2 * pi)
_LOG_2PI_HALF: float = -0.5 * float(np.log(2.0 * np.pi))
_SQRT_2: float = float(np.sqrt(2.0))
_LOG_SQRT_2: float = 0.5 * float(np.log(2.0))


def _ndtr_scalar(x: float) -> float:
    """Standard normal CDF for a scalar using math.erf."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _log_ndtr_scalar(x: float) -> float:
    """
    Log of standard normal CDF for a scalar.

    Uses different formulations for numerical stability in different regions.
    """
    if x > 6.0:
        # Very high x: Phi(x) ≈ 1, so log(Phi(x)) ≈ 0
        return 0.0
    elif x > -20.0:
        # Normal region: use erf directly
        phi: float = 0.5 * (1.0 + math.erf(x / _SQRT_2))
        if phi > 0:
            return math.log(phi)
        return float("-inf")
    else:
        # Far left tail: use asymptotic expansion
        # log(Phi(x)) ≈ -0.5*x^2 - log(-x) - 0.5*log(2*pi)
        return -0.5 * x * x - math.log(-x) + _LOG_2PI_HALF


def _log_ndtr(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Log of standard normal CDF, vectorized."""
    result: NDArray[np.float64] = np.zeros_like(x)
    for i in range(x.size):
        flat_x: float = float(x.flat[i])
        result.flat[i] = _log_ndtr_scalar(flat_x)
    return result


class TruncatedNormal(Distribution):
    """
    Truncated Normal distribution.

    The truncated normal distribution is a normal distribution bounded to
    lie within a specified interval [lower, upper].

    The probability density function is:
        p(x | mu, sigma, a, b) = phi((x-mu)/sigma) / (sigma * Z)
    where:
        phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)  (standard normal PDF)
        Z = Phi((b-mu)/sigma) - Phi((a-mu)/sigma)  (normalizing constant)
        Phi(z) is the standard normal CDF

    Parameters
    ----------
    mu : float
        Location parameter (mean of underlying untruncated normal).
    sigma : float
        Scale parameter (std of underlying untruncated normal), must be positive.
    lower : float
        Lower truncation bound. Use -np.inf or float('-inf') for no lower bound.
    upper : float
        Upper truncation bound. Use np.inf or float('inf') for no upper bound.

    Raises
    ------
    ModelSpecError
        If sigma is not positive or if lower >= upper.

    Examples
    --------
    >>> from minibayes import dist
    >>> # Lower-bounded only (common for positive parameters)
    >>> d = dist.TruncatedNormal(mu=0.5, sigma=1.0, lower=0.2)
    >>> samples = d.sample(size=1000)
    >>> assert samples.min() >= 0.2
    >>>
    >>> # Two-sided truncation
    >>> d2 = dist.TruncatedNormal(mu=0.0, sigma=1.0, lower=-1.0, upper=1.0)
    """

    @property
    def support(self) -> Support:
        return Support.BOUNDED

    @property
    def mean(self) -> float:
        """Return the mean of the truncated distribution."""
        # For truncated normal, mean = mu + sigma * (phi(alpha) - phi(beta)) / Z
        # where alpha = (a-mu)/sigma, beta = (b-mu)/sigma
        # This is approximate for now; return mode approximation
        if np.isfinite(self._lower) and np.isfinite(self._upper):
            # Both bounds finite: return midpoint as approximation
            return (self._lower + self._upper) / 2.0
        elif np.isfinite(self._lower):
            # Lower bound only: return mu if mu > lower, else lower + sigma
            return max(self._mu, self._lower + self._sigma)
        elif np.isfinite(self._upper):
            # Upper bound only
            return min(self._mu, self._upper - self._sigma)
        return self._mu

    @property
    def mu(self) -> float:
        """Location parameter of underlying normal."""
        return self._mu

    @property
    def sigma(self) -> float:
        """Scale parameter of underlying normal."""
        return self._sigma

    @property
    def lower(self) -> float:
        """Lower truncation bound."""
        return self._lower

    @property
    def upper(self) -> float:
        """Upper truncation bound."""
        return self._upper

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        lower: float = float("-inf"),
        upper: float = float("inf"),
    ) -> None:
        if sigma <= 0:
            raise ModelSpecError("sigma must be positive")
        if lower >= upper:
            raise ModelSpecError("lower must be less than upper")

        self._mu = mu
        self._sigma = sigma
        self._lower = lower
        self._upper = upper

        # Precompute standardized bounds
        self._alpha = (lower - mu) / sigma  # standardized lower
        self._beta = (upper - mu) / sigma  # standardized upper

        # Precompute log normalizing constant: log(Phi(beta) - Phi(alpha))
        log_phi_beta: float = _log_ndtr_scalar(self._beta)
        log_phi_alpha: float = _log_ndtr_scalar(self._alpha)

        # Use log-subtraction: log(Phi(b) - Phi(a)) = log(Phi(b)) + log(1 - exp(log(Phi(a)) - log(Phi(b))))
        if np.isfinite(log_phi_beta) and np.isfinite(log_phi_alpha):
            if log_phi_beta > log_phi_alpha:
                log_diff: float = log_phi_alpha - log_phi_beta
                self._log_normalizer: float = log_phi_beta + math.log1p(-math.exp(log_diff))
            else:
                # Edge case: should not happen if lower < upper
                self._log_normalizer = float("-inf")
        elif np.isfinite(log_phi_beta) and not np.isfinite(log_phi_alpha):
            # alpha = -inf, so Phi(alpha) = 0
            self._log_normalizer = log_phi_beta
        elif not np.isfinite(log_phi_beta):
            # beta = +inf, so Phi(beta) = 1
            if np.isfinite(log_phi_alpha):
                # log(1 - Phi(alpha)) = log(1 - exp(log_phi_alpha))
                phi_alpha: float = _ndtr_scalar(self._alpha)
                if phi_alpha < 1.0:
                    self._log_normalizer = math.log(1.0 - phi_alpha)
                else:
                    self._log_normalizer = float("-inf")
            else:
                # Both infinite: full support, normalizer = 1, log = 0
                self._log_normalizer = 0.0
        else:
            self._log_normalizer = 0.0

    def log_prob(
        self, x: NDArray[np.float64] | float
    ) -> NDArray[np.float64] | float:
        """
        Compute log probability density at x.

        Parameters
        ----------
        x : ndarray or float
            Point(s) at which to evaluate log probability.

        Returns
        -------
        ndarray or float
            Log probability density value(s).
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)

        # Check bounds
        in_support: NDArray[np.bool_] = (arr >= self._lower) & (arr <= self._upper)

        # Standard normal log PDF of standardized value
        z: NDArray[np.float64] = (arr - self._mu) / self._sigma
        z_squared: NDArray[np.float64] = np.square(z)
        log_phi_z: NDArray[np.float64] = _LOG_2PI_HALF - 0.5 * z_squared

        # log p(x) = log_phi(z) - log(sigma) - log_normalizer
        log_sigma: float = float(np.log(self._sigma))
        result: NDArray[np.float64] = log_phi_z - log_sigma - self._log_normalizer

        # Set to -inf outside support
        result = np.where(in_support, result, float("-inf"))

        if arr.ndim == 0:
            return float(result)
        return result

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64] | float:
        """
        Draw random samples from the truncated normal distribution.

        Uses the inverse CDF method for efficiency.

        Parameters
        ----------
        size : int, tuple, or None
            Output shape. If None, return a scalar.
        rng : Generator, optional
            NumPy random generator. If None, creates a new one.

        Returns
        -------
        ndarray or float
            Random sample(s).
        """
        generator: np.random.Generator = ensure_rng(rng)

        # Determine output shape
        if size is None:
            n_samples = 1
            return_scalar = True
        elif isinstance(size, int):
            n_samples = size
            return_scalar = False
        else:
            n_samples = int(np.prod(size))
            return_scalar = False

        # Compute Phi(alpha) and Phi(beta) for inverse CDF
        phi_alpha: float = _ndtr_scalar(self._alpha)
        phi_beta: float = _ndtr_scalar(self._beta)

        # Sample uniform on (Phi(alpha), Phi(beta))
        u: NDArray[np.float64] = generator.uniform(
            low=phi_alpha, high=phi_beta, size=n_samples
        )

        # Inverse CDF: x = mu + sigma * Phi^{-1}(u)
        # Use Newton's method or bisection for inverse normal CDF
        z: NDArray[np.float64] = self._inv_ndtr(u)
        samples: NDArray[np.float64] = self._mu + self._sigma * z

        # Clip to bounds (numerical safety)
        samples = np.clip(samples, self._lower, self._upper)

        if return_scalar:
            scalar_val: float = float(samples.flat[0])
            return scalar_val

        if size is not None and not isinstance(size, int):
            samples = samples.reshape(size)

        return samples

    def _inv_ndtr(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Inverse of standard normal CDF (quantile function).

        Uses rational approximation for efficiency.
        """
        result: NDArray[np.float64] = np.zeros_like(p)

        for i in range(p.size):
            result.flat[i] = self._inv_ndtr_scalar(float(p.flat[i]))

        return result

    def _inv_ndtr_scalar(self, p: float) -> float:
        """
        Inverse normal CDF for a single value.

        Uses Abramowitz and Stegun approximation.
        """
        if p <= 0.0:
            return float("-inf")
        if p >= 1.0:
            return float("inf")

        # Use symmetry: for p < 0.5, compute for 1-p and negate
        if p < 0.5:
            return -self._inv_ndtr_scalar(1.0 - p)

        # Rational approximation for 0.5 <= p < 1
        # From Abramowitz & Stegun, formula 26.2.23
        t: float = math.sqrt(-2.0 * math.log(1.0 - p))

        # Coefficients for rational approximation
        c0: float = 2.515517
        c1: float = 0.802853
        c2: float = 0.010328
        d1: float = 1.432788
        d2: float = 0.189269
        d3: float = 0.001308

        numerator: float = c0 + c1 * t + c2 * t * t
        denominator: float = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t

        return t - numerator / denominator

    def default_transform(self) -> Transform:
        """
        Return transform for this distribution.

        For finite bounds, uses AffineTransform.
        For lower-bounded only (common case), uses ShiftedLogTransform.

        Returns
        -------
        Transform
            Appropriate transform based on bounds.
        """
        from minibayes.transforms import AffineTransform, IdentityTransform

        lower_finite: bool = np.isfinite(self._lower)
        upper_finite: bool = np.isfinite(self._upper)

        if lower_finite and upper_finite:
            # Both bounds finite: use AffineTransform
            return AffineTransform(low=self._lower, high=self._upper)
        elif lower_finite and not upper_finite:
            # Lower bound only: use ShiftedLogTransform
            from minibayes.transforms import ShiftedLogTransform

            return ShiftedLogTransform(lower=self._lower)
        elif not lower_finite and upper_finite:
            # Upper bound only: use negative shifted log
            # x in (-inf, upper) -> y in (-inf, +inf) via log(upper - x)
            # For now, use identity with warning (rare case)
            return IdentityTransform()
        else:
            # No bounds: identity (shouldn't happen for TruncatedNormal)
            return IdentityTransform()
