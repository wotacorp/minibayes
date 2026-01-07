"""Base class for parameter transforms."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Transform(ABC):
    """Bijective transform for constrained parameters."""

    @abstractmethod
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform from constrained to unconstrained space.

        Parameters
        ----------
        x : ndarray
            Values in constrained space.

        Returns
        -------
        ndarray
            Values in unconstrained space.
        """

    @abstractmethod
    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform from unconstrained to constrained space.

        Parameters
        ----------
        y : ndarray
            Values in unconstrained space.

        Returns
        -------
        ndarray
            Values in constrained space.
        """

    @abstractmethod
    def log_det_jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Log absolute determinant of Jacobian (for constrained x).

        Parameters
        ----------
        x : ndarray
            Values in constrained space.

        Returns
        -------
        ndarray
            Log absolute determinant of Jacobian.
        """
