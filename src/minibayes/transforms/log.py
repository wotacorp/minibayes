"""Log transform for positive parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class LogTransform(Transform):
    """
    Log transform for positive parameters.

    forward: y = log(x)      [constrained -> unconstrained]
    inverse: x = exp(y)      [unconstrained -> constrained]
    log_det_jacobian: log(x) = y
    """

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError()
