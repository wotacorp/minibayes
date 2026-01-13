"""Poisson distribution."""

import math

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng


def _lgamma_scalar(val: float) -> float:
    """Compute lgamma for a scalar, handling poles at non-positive integers."""
    if val <= 0 and val == int(val):
        # lgamma has poles at non-positive integers; return -inf for log_prob
        return float("-inf")
    return math.lgamma(val)


def _lgamma_array(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute lgamma element-wise for arrays."""
    flat: list[float] = [_lgamma_scalar(float(v)) for v in x.flat]
    result: NDArray[np.float64] = np.array(flat, dtype=np.float64).reshape(x.shape)
    return result


class Poisson(Distribution):
    """
    Poisson distribution.

    The probability mass function is:
        P(X = k) = (rate^k * exp(-rate)) / k!

    Parameters
    ----------
    rate : float
        Expected number of events (lambda), must be positive.

    Raises
    ------
    ModelSpecError
        If rate is not positive.

    Notes
    -----
    Primarily used in likelihood functions for count data.
    """

    @property
    def support(self) -> Support:
        return Support.NATURAL

    @property
    def mean(self) -> float:
        return self._rate

    def __init__(self, rate: float = 1.0) -> None:
        if rate <= 0:
            raise ModelSpecError("rate must be positive")
        self._rate = rate
        self._log_rate: float = math.log(rate)

    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """
        Compute log probability mass at x.

        Parameters
        ----------
        x : ndarray or float
            Point(s) at which to evaluate log probability.
            Should be non-negative integers.

        Returns
        -------
        ndarray or float
            Log probability mass value(s).
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)

        # log P(X=k) = k * log(rate) - rate - lgamma(k+1)
        # lgamma(k+1) = log(k!)
        log_factorial: NDArray[np.float64] = _lgamma_array(arr + 1.0)
        result: NDArray[np.float64] = arr * self._log_rate - self._rate - log_factorial

        # Return -inf for non-natural numbers (negative or non-integer)
        is_non_negative: NDArray[np.bool_] = arr >= 0
        floored: NDArray[np.float64] = np.floor(arr)
        is_integer: NDArray[np.bool_] = arr == floored
        is_natural: NDArray[np.bool_] = is_non_negative & is_integer
        result = np.where(is_natural, result, -np.inf)

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
            Random sample(s) of non-negative integers.
        """
        generator: np.random.Generator = ensure_rng(rng)
        samples: NDArray[np.int64] = generator.poisson(lam=self._rate, size=size)
        samples_float: NDArray[np.float64] = samples.astype(np.float64)
        if size is None:
            return float(samples_float)
        return samples_float
