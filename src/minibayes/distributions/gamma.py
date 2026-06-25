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

"""Gamma distribution."""

from math import lgamma

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng


class Gamma(Distribution):
    """
    Gamma distribution.

    The probability density function is:
        p(x) = (β^α / Γ(α)) x^(α-1) exp(-βx)  for x > 0

    Note: This uses the shape/rate parameterization, not shape/scale.
    NumPy uses shape/scale, so we convert internally (scale = 1/rate).

    Parameters
    ----------
    shape : float
        Shape parameter (α), must be positive.
    rate : float
        Rate parameter (β), must be positive.

    Raises
    ------
    ModelSpecError
        If shape or rate is not positive.
    """

    @property
    def support(self) -> Support:
        return Support.POSITIVE

    @property
    def mean(self) -> float:
        return self._shape / self._rate

    def __init__(self, shape: float = 1.0, rate: float = 1.0) -> None:
        if shape <= 0:
            raise ModelSpecError("shape must be positive")
        if rate <= 0:
            raise ModelSpecError("rate must be positive")
        self._shape = shape
        self._rate = rate
        # Precompute constant term: α·log(β) - lgamma(α)
        self._log_normalizer: float = shape * float(np.log(rate)) - lgamma(shape)

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
            Log probability density value(s). Returns -inf for x <= 0.
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        # Suppress warnings for log of non-positive values (handled via np.where)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_x: NDArray[np.float64] = np.log(arr)
            # log_prob = α·log(β) + (α-1)·log(x) - β·x - lgamma(α)
            #          = log_normalizer + (α-1)·log(x) - β·x
            result: NDArray[np.float64] = self._log_normalizer + (self._shape - 1) * log_x - self._rate * arr
            # Set log_prob to -inf where x <= 0
            result = np.where(arr > 0, result, -np.inf)
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
            Random sample(s), always positive.
        """
        generator: np.random.Generator = ensure_rng(rng)
        # NumPy uses scale = 1/rate
        scale: float = 1.0 / self._rate
        samples: NDArray[np.float64] = generator.gamma(shape=self._shape, scale=scale, size=size)
        if size is None:
            return float(samples)
        return samples
