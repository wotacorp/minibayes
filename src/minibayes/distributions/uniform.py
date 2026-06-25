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

"""Uniform distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.transforms import AffineTransform, Transform
from minibayes.utils import ensure_rng


class Uniform(Distribution):
    """
    Uniform distribution on [low, high].

    The probability density function is:
        p(x) = 1 / (high - low)  for x ∈ [low, high]

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound, must be greater than low.

    Raises
    ------
    ModelSpecError
        If high is not greater than low.
    """

    @property
    def support(self) -> Support:
        return Support.BOUNDED

    @property
    def mean(self) -> float:
        return (self._low + self._high) / 2.0

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        if high <= low:
            raise ModelSpecError("high must be greater than low")
        self._low = low
        self._high = high
        self._width = high - low
        # Precompute log probability (constant for uniform)
        self._log_prob_value: float = -float(np.log(self._width))

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
            Log probability density value(s). Returns -inf for x outside [low, high].
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        # Check if in support
        in_support: NDArray[np.bool_] = (arr >= self._low) & (arr <= self._high)
        result: NDArray[np.float64] = np.where(in_support, self._log_prob_value, -np.inf)
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
            Random sample(s), in [low, high].
        """
        generator: np.random.Generator = ensure_rng(rng)
        samples: NDArray[np.float64] = generator.uniform(low=self._low, high=self._high, size=size)
        if size is None:
            return float(samples)
        return samples

    def default_transform(self) -> Transform:
        """
        Return AffineTransform for this bounded distribution.

        Returns
        -------
        Transform
            AffineTransform with this distribution's bounds.
        """
        return AffineTransform(low=self._low, high=self._high)
