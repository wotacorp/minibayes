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

"""Affine transform for bounded parameters."""

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class AffineTransform(Transform):
    """
    Affine transform for bounded parameters.

    Maps (low, high) to (-inf, +inf) via scaled logit.

    forward: φ = logit((θ-a)/(b-a))     [constrained -> unconstrained]
    inverse: θ = a + (b-a)·σ(φ)         [unconstrained -> constrained]
    log_det_jacobian: log|dθ/dφ| = log(θ-a) + log(b-θ) - log(b-a)

    Parameters
    ----------
    low : float
        Lower bound (a).
    high : float
        Upper bound (b).
    """

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high
        self._width = high - low

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        z: NDArray[np.float64] = (arr - self.low) / self._width
        result: NDArray[np.float64] = np.log(z / (1 - z))
        return result

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        exp_neg: NDArray[np.float64] = np.exp(-arr)
        denom: NDArray[np.float64] = 1 + exp_neg
        z: NDArray[np.float64] = np.reciprocal(denom)
        result: NDArray[np.float64] = self.low + z * self._width
        return result

    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        log_low: NDArray[np.float64] = np.log(arr - self.low)
        log_high: NDArray[np.float64] = np.log(self.high - arr)
        log_width: float = float(np.log(self._width))
        result: NDArray[np.float64] = log_low + log_high - log_width
        return result
