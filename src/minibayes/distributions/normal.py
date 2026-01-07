"""Normal (Gaussian) distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support


class Normal(Distribution):
    """
    Normal (Gaussian) distribution.

    Parameters
    ----------
    loc : float
        Mean of the distribution.
    scale : float
        Standard deviation (must be positive).
    """

    @property
    def support(self) -> Support:
        return Support.REAL

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        raise NotImplementedError()

    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        raise NotImplementedError()

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64] | float:
        raise NotImplementedError()
