"""Identity transform for unconstrained parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class IdentityTransform(Transform):
    """No transformation. For REAL support."""

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()
