"""Uniform distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support


class Uniform(Distribution):
    """
    Uniform distribution on [low, high].

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound (must be greater than low).
    """

    @property
    def support(self) -> Support:
        return Support.BOUNDED

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        raise NotImplementedError()

    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        raise NotImplementedError()

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64] | float:
        raise NotImplementedError()
