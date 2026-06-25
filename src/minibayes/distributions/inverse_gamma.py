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

"""Inverse gamma distribution."""

import math

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng


class InverseGamma(Distribution):
    """
    Inverse gamma distribution.

    If X ~ Gamma(shape, rate), then 1/X ~ InverseGamma(shape, scale=1/rate).

    The probability density function is:
        p(x) = (scale^shape / Gamma(shape)) * x^(-shape-1) * exp(-scale/x)

    Parameters
    ----------
    shape : float
        Shape parameter (alpha), must be positive.
    scale : float
        Scale parameter (beta), must be positive.

    Raises
    ------
    ModelSpecError
        If shape or scale is not positive.

    Notes
    -----
    Common prior for variance parameters. If X ~ InverseGamma(shape, scale),
    then X is the conjugate prior for the variance of a Normal distribution.
    """

    @property
    def support(self) -> Support:
        return Support.POSITIVE

    @property
    def mean(self) -> float:
        if self._shape <= 1:
            return float("inf")
        return self._scale / (self._shape - 1)

    def __init__(self, shape: float = 1.0, scale: float = 1.0) -> None:
        if shape <= 0:
            raise ModelSpecError("shape must be positive")
        if scale <= 0:
            raise ModelSpecError("scale must be positive")
        self._shape = shape
        self._scale = scale

        # Precompute log normalizer: shape * log(scale) - lgamma(shape)
        self._log_normalizer: float = self._shape * math.log(self._scale) - math.lgamma(self._shape)
        self._neg_shape_plus_1: float = -(self._shape + 1)

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

        # Compute log_prob, then mask non-positive values to -inf
        with np.errstate(invalid="ignore", divide="ignore"):
            log_x: NDArray[np.float64] = np.log(arr)
            inv_x: NDArray[np.float64] = np.reciprocal(arr)
            # log p(x) = log_normalizer + (-shape-1)*log(x) - scale/x
            result: NDArray[np.float64] = self._log_normalizer + self._neg_shape_plus_1 * log_x - self._scale * inv_x

        # Return -inf for non-positive values
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

        Uses the relationship: if X ~ Gamma(shape, 1/scale), then 1/X ~ InverseGamma.

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
        # Gamma uses scale parameterization, so scale = 1/self._scale
        gamma_samples: NDArray[np.float64] = generator.gamma(shape=self._shape, scale=1.0 / self._scale, size=size)
        samples: NDArray[np.float64] = np.reciprocal(gamma_samples)
        if size is None:
            return float(samples)
        return samples
