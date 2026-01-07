"""Identity transform for unconstrained parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class IdentityTransform(Transform):
    """No transformation. For REAL support."""

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(x, dtype=np.float64)

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(y, dtype=np.float64)

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
