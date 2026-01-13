"""Log-normal distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng

# Precompute constant: -0.5 * log(2 * pi)
_LOG_2PI_HALF: float = -0.5 * float(np.log(2.0 * np.pi))


class LogNormal(Distribution):
    """
    Log-normal distribution.

    If X ~ Normal(loc, scale), then exp(X) ~ LogNormal(loc, scale).

    The probability density function is:
        p(x) = (1 / (x * scale * sqrt(2*pi))) exp(-(log(x) - loc)^2 / (2*scale^2))

    Parameters
    ----------
    loc : float
        Mean of the underlying normal distribution (mu).
    scale : float
        Standard deviation of the underlying normal (sigma), must be positive.

    Raises
    ------
    ModelSpecError
        If scale is not positive.

    Notes
    -----
    The mean of the log-normal is exp(loc + scale^2/2), not loc.
    """

    @property
    def support(self) -> Support:
        return Support.POSITIVE

    @property
    def mean(self) -> float:
        return float(np.exp(self._loc + 0.5 * self._scale * self._scale))

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ModelSpecError("scale must be positive")
        self._loc = loc
        self._scale = scale

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

        # Compute log_prob, then mask non-positive values to -inf
        with np.errstate(invalid="ignore", divide="ignore"):
            log_x: NDArray[np.float64] = np.log(arr)
            z: NDArray[np.float64] = (log_x - self._loc) / self._scale
            z_squared: NDArray[np.float64] = np.square(z)
            log_scale: float = float(np.log(self._scale))
            # Normal log_prob on log(x), minus log(x) for Jacobian
            result: NDArray[np.float64] = _LOG_2PI_HALF - log_scale - 0.5 * z_squared - log_x

        # Return -inf for non-positive values
        result = np.where(arr > 0, result, -np.inf)

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
        normal_samples: NDArray[np.float64] = generator.normal(loc=self._loc, scale=self._scale, size=size)
        samples: NDArray[np.float64] = np.exp(normal_samples)
        if size is None:
            return float(samples)
        return samples
