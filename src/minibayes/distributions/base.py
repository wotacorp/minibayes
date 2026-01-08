"""Base classes for probability distributions."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from minibayes.transforms import Transform


class Support(Enum):
    """Support of a probability distribution."""

    REAL = "real"  # (-inf, +inf) -> IdentityTransform
    POSITIVE = "positive"  # (0, +inf) -> LogTransform
    UNIT = "unit"  # (0, 1) -> LogitTransform
    BOUNDED = "bounded"  # (a, b) -> AffineTransform


class Distribution(ABC):
    """
    Abstract base class for probability distributions.

    All distributions must implement:
    - support: The domain of the distribution
    - log_prob: Log probability density/mass function
    - sample: Random sampling
    """

    @property
    @abstractmethod
    def support(self) -> Support:
        """
        Return support of distribution.

        Used to determine automatic transform:
        - REAL -> IdentityTransform
        - POSITIVE -> LogTransform
        - UNIT -> LogitTransform
        - BOUNDED -> AffineTransform

        Returns
        -------
        Support
            The support enum value.
        """

    @property
    @abstractmethod
    def mean(self) -> float:
        """
        Return the mean (expected value) of the distribution.

        Returns
        -------
        float
            The mean of the distribution.
        """

    @abstractmethod
    def log_prob(self, x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """
        Compute log probability density/mass at x.

        Parameters
        ----------
        x : ndarray or float
            Point(s) at which to evaluate log probability.

        Returns
        -------
        ndarray or float
            Log probability value(s).
        """

    @abstractmethod
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
            NumPy random generator. If None, use default.

        Returns
        -------
        ndarray or float
            Random sample(s).
        """

    def obs_logp(self, data: NDArray[np.float64] | float) -> float:
        """
        Compute total log probability for observed data.

        Equivalent to float(np.sum(self.log_prob(data))).

        Parameters
        ----------
        data : ndarray or float
            Observed data points.

        Returns
        -------
        float
            Sum of log probabilities.
        """
        lp: NDArray[np.float64] | float = self.log_prob(data)
        if isinstance(lp, float):
            return lp
        lp_sum: float = float(np.sum(lp))
        return lp_sum

    def default_transform(self) -> "Transform":
        """
        Return appropriate transform for this distribution's support.

        Override in subclass if non-standard transform is preferred.

        Returns
        -------
        Transform
            Transform instance for this distribution.
        """
        from minibayes.transforms import (
            IdentityTransform,
            LogitTransform,
            LogTransform,
        )

        match self.support:
            case Support.REAL:
                return IdentityTransform()
            case Support.POSITIVE:
                return LogTransform()
            case Support.UNIT:
                return LogitTransform()
            case Support.BOUNDED:
                raise NotImplementedError("BOUNDED distributions must override default_transform() to provide bounds for AffineTransform")
