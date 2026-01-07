"""Logit transform for unit interval parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class LogitTransform(Transform):
    """
    Logit transform for (0, 1) parameters.

    forward: y = log(x / (1-x))
    inverse: x = 1 / (1 + exp(-y))
    """

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()
