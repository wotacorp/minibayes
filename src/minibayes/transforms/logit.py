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
