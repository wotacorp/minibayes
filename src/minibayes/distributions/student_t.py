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

"""Student's t distribution."""

import math

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng

# Precompute constant
_LOG_PI: float = float(np.log(np.pi))


class StudentT(Distribution):
    """
    Student's t distribution.

    The probability density function is:
        p(x) = Gamma((df+1)/2) / (sqrt(df*pi) * scale * Gamma(df/2))
               * (1 + ((x-loc)/scale)^2 / df)^(-(df+1)/2)

    Parameters
    ----------
    df : float
        Degrees of freedom, must be positive.
    loc : float
        Location parameter (mean when df > 1).
    scale : float
        Scale parameter, must be positive.

    Raises
    ------
    ModelSpecError
        If df or scale is not positive.

    Notes
    -----
    - When df = 1, this is the Cauchy distribution.
    - When df -> inf, this approaches the Normal distribution.
    - The mean exists only for df > 1, variance only for df > 2.
    """

    @property
    def support(self) -> Support:
        return Support.REAL

    @property
    def mean(self) -> float:
        if self._df <= 1:
            return float("nan")
        return self._loc

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0) -> None:
        if df <= 0:
            raise ModelSpecError("df must be positive")
        if scale <= 0:
            raise ModelSpecError("scale must be positive")
        self._df = df
        self._loc = loc
        self._scale = scale

        # Precompute log normalizer
        df_half: float = 0.5 * df
        df_plus_1_half: float = 0.5 * (df + 1.0)
        self._log_normalizer: float = (
            math.lgamma(df_plus_1_half) - math.lgamma(df_half) - 0.5 * float(np.log(df)) - 0.5 * _LOG_PI - float(np.log(scale))
        )
        self._neg_df_plus_1_half: float = -df_plus_1_half

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
        log_term: NDArray[np.float64] = np.log1p(z_squared / self._df)
        result: NDArray[np.float64] = self._log_normalizer + self._neg_df_plus_1_half * log_term
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
        samples: NDArray[np.float64] = generator.standard_t(df=self._df, size=size)
        scaled_samples: NDArray[np.float64] = self._loc + self._scale * samples
        if size is None:
            return float(scaled_samples)
        return scaled_samples
