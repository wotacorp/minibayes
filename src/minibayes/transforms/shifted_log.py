# Copyright 2026 WOTA CORP.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shifted log transform for lower-bounded parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class ShiftedLogTransform(Transform):
    """
    Shifted log transform for lower-bounded parameters.

    Maps (lower, +inf) to (-inf, +inf) via shifted logarithm.

    forward: phi = log(theta - lower)     [constrained -> unconstrained]
    inverse: theta = exp(phi) + lower     [unconstrained -> constrained]
    log_det_jacobian: log|d(theta)/d(phi)| = phi = log(theta - lower)

    This transform is useful for parameters that have a lower bound but
    no upper bound, such as TruncatedNormal with only a lower bound.

    Parameters
    ----------
    lower : float
        Lower bound of the parameter space.

    Examples
    --------
    >>> import numpy as np
    >>> from minibayes.transforms import ShiftedLogTransform
    >>> t = ShiftedLogTransform(lower=0.2)
    >>> x = np.array([0.5, 1.0, 2.0])  # Values > 0.2
    >>> y = t.forward(x)  # Transform to unconstrained
    >>> x_back = t.inverse(y)  # Transform back
    >>> np.allclose(x, x_back)
    True
    """

    def __init__(self, lower: float) -> None:
        self._lower = lower

    @property
    def lower(self) -> float:
        """Lower bound of the parameter space."""
        return self._lower

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform from constrained to unconstrained space.

        Parameters
        ----------
        x : ndarray
            Values in (lower, +inf).

        Returns
        -------
        ndarray
            Values in (-inf, +inf).
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        shifted: NDArray[np.float64] = arr - self._lower
        result: NDArray[np.float64] = np.log(shifted)
        return result

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform from unconstrained to constrained space.

        Parameters
        ----------
        y : ndarray
            Values in (-inf, +inf).

        Returns
        -------
        ndarray
            Values in (lower, +inf).
        """
        arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        exp_y: NDArray[np.float64] = np.exp(arr)
        result: NDArray[np.float64] = exp_y + self._lower
        return result

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute log absolute determinant of the Jacobian.

        For the inverse transform theta = exp(phi) + lower:
            d(theta)/d(phi) = exp(phi) = theta - lower
            log|d(theta)/d(phi)| = log(theta - lower) = phi

        Parameters
        ----------
        x : ndarray
            Values in constrained space (lower, +inf).

        Returns
        -------
        ndarray
            Log determinant of Jacobian at each point.
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        shifted: NDArray[np.float64] = arr - self._lower
        result: NDArray[np.float64] = np.log(shifted)
        return result
