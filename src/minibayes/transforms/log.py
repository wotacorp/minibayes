"""Log transform for positive parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class LogTransform(Transform):
    """
    Log transform for positive parameters.

    forward: φ = log(θ)      [constrained -> unconstrained]
    inverse: θ = exp(φ)      [unconstrained -> constrained]
    log_det_jacobian: log|dθ/dφ| = φ = log(θ)
    """

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        result: NDArray[np.float64] = np.log(arr)
        return result

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        result: NDArray[np.float64] = np.exp(arr)
        return result

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        result: NDArray[np.float64] = np.log(arr)
        return result
