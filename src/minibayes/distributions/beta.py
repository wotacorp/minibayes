"""Beta distribution."""

from math import lgamma

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng


class Beta(Distribution):
    """
    Beta distribution on (0, 1).

    The probability density function is:
        p(x) = (1/B(α,β)) x^(α-1) (1-x)^(β-1)  for x ∈ (0, 1)

    where B(α,β) = Γ(α)Γ(β)/Γ(α+β) is the beta function.

    Parameters
    ----------
    alpha : float
        First shape parameter (α), must be positive.
    beta : float
        Second shape parameter (β), must be positive.

    Raises
    ------
    ModelSpecError
        If alpha or beta is not positive.
    """

    @property
    def support(self) -> Support:
        return Support.UNIT

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha <= 0:
            raise ModelSpecError("alpha must be positive")
        if beta <= 0:
            raise ModelSpecError("beta must be positive")
        self._alpha = alpha
        self._beta = beta
        # Precompute log of beta function: log(B(α,β)) = lgamma(α) + lgamma(β) - lgamma(α+β)
        self._log_beta: float = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)

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
            Log probability density value(s). Returns -inf for x outside (0, 1).
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
        # Suppress warnings for log of boundary values (handled via np.where)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_x: NDArray[np.float64] = np.log(arr)
            log_1_minus_x: NDArray[np.float64] = np.log(1 - arr)
            # log_prob = (α-1)·log(x) + (β-1)·log(1-x) - log(B(α,β))
            result: NDArray[np.float64] = (self._alpha - 1) * log_x + (self._beta - 1) * log_1_minus_x - self._log_beta
            # Set log_prob to -inf where x is outside (0, 1)
            in_support: NDArray[np.bool_] = (arr > 0) & (arr < 1)
            result = np.where(in_support, result, -np.inf)
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
            Random sample(s), in (0, 1).
        """
        generator: np.random.Generator = ensure_rng(rng)
        samples: NDArray[np.float64] = generator.beta(a=self._alpha, b=self._beta, size=size)
        if size is None:
            return float(samples)
        return samples
