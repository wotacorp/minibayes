"""Affine transform for bounded parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class AffineTransform(Transform):
    """
    Affine transform for bounded parameters.

    Maps (low, high) to (-inf, +inf) via scaled logit.

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound.
    """

    def __init__(self, low: float, high: float) -> None:
        raise NotImplementedError()

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()
