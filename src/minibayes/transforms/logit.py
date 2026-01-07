"""Logit transform for unit interval parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class LogitTransform(Transform):
    """
    Logit transform for (0, 1) parameters.

    forward: φ = log(θ / (1-θ))     [constrained -> unconstrained]
    inverse: θ = 1 / (1 + exp(-φ))  [unconstrained -> constrained]
    log_det_jacobian: log|dθ/dφ| = log(θ) + log(1-θ)
    """

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        result: NDArray[np.float64] = np.log(arr / (1 - arr))
        return result

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        exp_neg: NDArray[np.float64] = np.exp(-arr)
        denom: NDArray[np.float64] = 1 + exp_neg
        result: NDArray[np.float64] = np.reciprocal(denom)
        return result

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        result: NDArray[np.float64] = np.log(arr) + np.log(1 - arr)
        return result
