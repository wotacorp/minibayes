"""Cauchy distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng

# Precompute constant
_LOG_PI: float = float(np.log(np.pi))


class Cauchy(Distribution):
    """
    Cauchy distribution.

    The probability density function is:
        p(x) = 1 / (pi * scale * (1 + ((x - loc) / scale)^2))

    Parameters
    ----------
    loc : float
        Location parameter (median).
    scale : float
        Scale parameter (half-width at half-maximum), must be positive.

    Raises
    ------
    ModelSpecError
        If scale is not positive.

    Notes
    -----
    The Cauchy distribution has no defined mean or variance (they are infinite).
    It is equivalent to StudentT with df=1.
    """

    @property
    def support(self) -> Support:
        return Support.REAL

    @property
    def mean(self) -> float:
        # Cauchy has no defined mean
        return float("nan")

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ModelSpecError("scale must be positive")
        self._loc = loc
        self._scale = scale
        # Precompute log normalizer: -log(pi * scale)
        self._log_normalizer: float = -_LOG_PI - float(np.log(scale))

    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
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
        z: NDArray[np.float64] = (arr - self._loc) / self._scale
        z_squared: NDArray[np.float64] = np.square(z)
        log_denom: NDArray[np.float64] = np.log1p(z_squared)
        result: NDArray[np.float64] = self._log_normalizer - log_denom
        if arr.ndim == 0:
            return float(result)
        return result

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64] | float:
        """
        Draw random samples from the distribution.

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
        # Use inverse CDF: x = loc + scale * tan(pi * (U - 0.5))
        u: NDArray[np.float64] = generator.uniform(size=size)
        shifted_u: NDArray[np.float64] = u - 0.5
        tan_vals: NDArray[np.float64] = np.tan(np.pi * shifted_u)
        samples: NDArray[np.float64] = self._loc + self._scale * tan_vals
        if size is None:
            return float(samples)
        return samples
