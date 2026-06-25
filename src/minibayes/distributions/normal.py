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

"""Normal (Gaussian) distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng

# Precompute constant: -0.5 * log(2 * pi)
_LOG_2PI_HALF: float = -0.5 * float(np.log(2.0 * np.pi))


class Normal(Distribution):
    """
    Normal (Gaussian) distribution.

    The probability density function is:
        p(x) = (1 / (σ√(2π))) exp(-(x-μ)² / (2σ²))

    Parameters
    ----------
    loc : float
        Mean of the distribution (μ).
    scale : float
        Standard deviation (σ), must be positive.

    Raises
    ------
    ModelSpecError
        If scale is not positive.
    """

    @property
    def support(self) -> Support:
        return Support.REAL

    @property
    def mean(self) -> float:
        return self._loc

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ModelSpecError("scale must be positive")
        self._loc = loc
        self._scale = scale

    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """
        Compute log probability density at x.

        Parameters
        ----------
        x : ndarray or float
            Point(s) at which to evaluate log probability.

        Returns
        -------
        ndarray or float
            Log probability density value(s).
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        z: NDArray[np.float64] = (arr - self._loc) / self._scale
        z_squared: NDArray[np.float64] = np.square(z)
        log_scale: float = float(np.log(self._scale))
        result: NDArray[np.float64] = _LOG_2PI_HALF - log_scale - 0.5 * z_squared
        if arr.ndim == 0:
            return float(result)
        return result

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64] | float:
        """
        Draw random samples from the distribution.

        Parameters
        ----------
        size : int, tuple, or None
            Output shape. If None, return a scalar.
        rng : Generator, optional
            NumPy random generator. If None, creates a new one.

        Returns
        -------
        ndarray or float
            Random sample(s).
        """
        generator: np.random.Generator = ensure_rng(rng)
        samples: NDArray[np.float64] = generator.normal(loc=self._loc, scale=self._scale, size=size)
        if size is None:
            return float(samples)
        return samples
