"""Numerical utilities for minibayes."""

import numpy as np
from numpy.typing import NDArray

from minibayes.exceptions import NumericalError


def ensure_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """
    Ensure we have a numpy random Generator.

    Parameters
    ----------
    seed : int, Generator, or None
        If int, create new Generator with this seed.
        If Generator, return as-is.
        If None, create Generator with random seed.

    Returns
    -------
    np.random.Generator
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def check_finite(value: float, name: str = "value") -> None:
    """
    Check that a value is finite (not NaN or Inf).

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name for error message.

    Raises
    ------
    NumericalError
        If value is not finite.
    """
    if not np.isfinite(value):
        raise NumericalError(f"{name} is not finite: {value}")


def log_sum_exp(x: NDArray[np.float64]) -> float:
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    float
        log(sum(exp(x)))
    """
    x = np.asarray(x)
    max_val: float = float(np.max(x))
    if np.isinf(max_val) and max_val < 0:
        return float("-inf")
    shifted: NDArray[np.float64] = x - max_val
    exp_shifted: NDArray[np.float64] = np.exp(shifted)
    sum_exp: float = float(np.sum(exp_shifted))
    return max_val + float(np.log(sum_exp))
