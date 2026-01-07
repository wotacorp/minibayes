"""Beta distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support


class Beta(Distribution):
    """
    Beta distribution on (0, 1).

    Parameters
    ----------
    alpha : float
        First shape parameter (must be positive).
    beta : float
        Second shape parameter (must be positive).
    """

    @property
    def support(self) -> Support:
        return Support.UNIT

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        raise NotImplementedError()

    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        raise NotImplementedError()

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64] | float:
        raise NotImplementedError()
