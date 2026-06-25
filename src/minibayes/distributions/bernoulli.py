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

"""Bernoulli distribution."""

import math

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng


class Bernoulli(Distribution):
    """
    Bernoulli distribution.

    The probability mass function is:
        P(X = 1) = prob
        P(X = 0) = 1 - prob

    Parameters
    ----------
    prob : float
        Probability of success, must be in [0, 1].

    Raises
    ------
    ModelSpecError
        If prob is not in [0, 1].

    Notes
    -----
    Primarily used in likelihood functions for binary outcome data.
    """

    @property
    def support(self) -> Support:
        return Support.BINARY

    @property
    def mean(self) -> float:
        return self._prob

    def __init__(self, prob: float = 0.5) -> None:
        if prob < 0 or prob > 1:
            raise ModelSpecError("prob must be in [0, 1]")
        self._prob = prob

        # Precompute log probabilities (handle edge cases)
        if prob == 0:
            self._log_prob_1: float = float("-inf")
            self._log_prob_0: float = 0.0
        elif prob == 1:
            self._log_prob_1 = 0.0
            self._log_prob_0 = float("-inf")
        else:
            self._log_prob_1 = math.log(prob)
            self._log_prob_0 = math.log(1.0 - prob)

    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """
        Compute log probability mass at x.

        Parameters
        ----------
        x : ndarray or float
            Point(s) at which to evaluate log probability.
            Should be 0 or 1.

        Returns
        -------
        ndarray or float
            Log probability mass value(s).
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)

        # log P(X=x) = x * log(p) + (1-x) * log(1-p)
        result: NDArray[np.float64] = arr * self._log_prob_1 + (1.0 - arr) * self._log_prob_0

        # Return -inf for values not in {0, 1}
        is_zero: NDArray[np.bool_] = arr == 0
        is_one: NDArray[np.bool_] = arr == 1
        valid: NDArray[np.bool_] = is_zero | is_one
        result = np.where(valid, result, -np.inf)

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
            Random sample(s) of 0 or 1.
        """
        generator: np.random.Generator = ensure_rng(rng)
        samples: NDArray[np.int64] = generator.binomial(n=1, p=self._prob, size=size)
        samples_float: NDArray[np.float64] = samples.astype(np.float64)
        if size is None:
            return float(samples_float)
        return samples_float
