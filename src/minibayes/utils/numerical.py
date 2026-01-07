"""Numerical utilities for minibayes."""

import numpy as np
from numpy.typing import NDArray


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()
