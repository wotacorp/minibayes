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

"""Laplace (double exponential) distribution."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng


class Laplace(Distribution):
    """
    Laplace (double exponential) distribution.

    The probability density function is:
        p(x) = (1 / (2 * scale)) * exp(-|x - loc| / scale)

    Parameters
    ----------
    loc : float
        Location parameter (mean and median).
    scale : float
        Scale parameter (diversity), must be positive.

    Raises
    ------
    ModelSpecError
        If scale is not positive.

    Notes
    -----
    The Laplace distribution is useful for sparse priors, as it corresponds
    to L1 regularization (Lasso) in regression.
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
        # Precompute log normalizer: -log(2 * scale)
        self._log_normalizer: float = -float(np.log(2.0)) - float(np.log(scale))

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
        abs_diff: NDArray[np.float64] = np.abs(arr - self._loc)
        result: NDArray[np.float64] = self._log_normalizer - abs_diff / self._scale
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
        samples: NDArray[np.float64] = generator.laplace(loc=self._loc, scale=self._scale, size=size)
        if size is None:
            return float(samples)
        return samples
