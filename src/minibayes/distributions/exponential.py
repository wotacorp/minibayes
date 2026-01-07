"""Exponential distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support


class Exponential(Distribution):
    """
    Exponential distribution.

    Parameters
    ----------
    rate : float
        Rate parameter (must be positive).
    """

    @property
    def support(self) -> Support:
        return Support.POSITIVE

    def __init__(self, rate: float = 1.0) -> None:
        raise NotImplementedError()

    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        raise NotImplementedError()

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64] | float:
        raise NotImplementedError()
