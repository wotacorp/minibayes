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
