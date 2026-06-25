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

"""Transform for correlation matrix Cholesky factors."""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from minibayes.transforms.base import Transform


class CorrCholeskyTransform(Transform):
    """
    Transform for correlation matrix Cholesky factors.

    Maps between:
    - Constrained: Lower triangular Cholesky factor L where L @ L.T is a
      correlation matrix (diagonal = 1, off-diagonal in (-1, 1))
    - Unconstrained: Flat vector of arctanh-transformed off-diagonal elements

    The diagonal of L is determined by the unit row norm constraint:
    L[i,i] = sqrt(1 - sum(L[i,:i]^2))

    Parameters
    ----------
    dim : int
        Dimension of the correlation matrix.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._n_offdiag = dim * (dim - 1) // 2

    def forward(self, chol: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform from constrained Cholesky to unconstrained vector.

        Parameters
        ----------
        chol : ndarray, shape (dim, dim)
            Lower triangular Cholesky factor of correlation matrix.

        Returns
        -------
        ndarray, shape (n_offdiag,)
            Unconstrained vector (arctanh of normalized off-diagonals).
        """
        arr: NDArray[np.float64] = np.asarray(chol, dtype=np.float64)
        y: NDArray[np.float64] = np.zeros(self._n_offdiag, dtype=np.float64)

        idx = 0
        for i in range(1, self._dim):
            for j in range(i):
                # Compute the "remaining variance" for this position
                row_sum_sq: float = cast("float", np.sum(arr[i, :j] ** 2))
                remaining: float = 1.0 - row_sum_sq
                if remaining <= 0:
                    remaining = 1e-10  # Numerical safeguard

                # Normalize the off-diagonal element to (-1, 1)
                elem_ij: float = float(arr[i, j])  # type: ignore[misc]
                z: float = elem_ij / float(np.sqrt(remaining))
                # Clamp to avoid arctanh overflow
                z = max(-0.9999, min(0.9999, z))
                y[idx] = float(np.arctanh(z))
                idx += 1

        return y

    def inverse(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform from unconstrained vector to constrained Cholesky.

        Parameters
        ----------
        y : ndarray, shape (n_offdiag,)
            Unconstrained vector.

        Returns
        -------
        ndarray, shape (dim, dim)
            Lower triangular Cholesky factor of correlation matrix.
        """
        arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        chol: NDArray[np.float64] = np.zeros((self._dim, self._dim), dtype=np.float64)

        # First row: chol[0,0] = 1 (unit diagonal for correlation)
        chol[0, 0] = 1.0

        idx = 0
        for i in range(1, self._dim):
            row_sum_sq: float = 0.0
            for j in range(i):
                # Transform back to (-1, 1)
                elem_idx: float = float(arr[idx])  # type: ignore[misc]
                z: float = float(np.tanh(elem_idx))
                remaining: float = 1.0 - row_sum_sq
                if remaining <= 0:
                    remaining = 1e-10

                chol_ij: float = z * float(np.sqrt(remaining))
                chol[i, j] = chol_ij
                row_sum_sq += chol_ij * chol_ij
                idx += 1

            # Diagonal element from unit row norm constraint
            diag_sq: float = 1.0 - row_sum_sq
            chol[i, i] = float(np.sqrt(max(diag_sq, 1e-10)))

        return chol

    def log_det_jacobian(self, chol: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Log absolute determinant of Jacobian of inverse transform.

        Returns log|dL/dy| where L is constrained and y is unconstrained.
        This follows the minibayes convention: log_det_jacobian returns
        the Jacobian needed for density correction when sampling in
        unconstrained space.

        Parameters
        ----------
        chol : ndarray, shape (dim, dim)
            Lower triangular Cholesky factor in constrained space.

        Returns
        -------
        ndarray
            Log |dL/dy| as 0-d array.
        """
        arr: NDArray[np.float64] = np.asarray(chol, dtype=np.float64)
        log_det: float = 0.0

        for i in range(1, self._dim):
            for j in range(i):
                row_sum_sq: float = cast("float", np.sum(arr[i, :j] ** 2))
                remaining: float = 1.0 - row_sum_sq
                if remaining <= 0:
                    remaining = 1e-10

                elem_ij: float = float(arr[i, j])  # type: ignore[misc]
                z: float = elem_ij / float(np.sqrt(remaining))
                z = max(-0.9999, min(0.9999, z))

                # Jacobian of inverse: dL/dy = sqrt(r) * (1 - z^2)
                # log |dL/dy| = 0.5 * log(r) + log(1 - z^2)
                log_det += 0.5 * float(np.log(remaining)) + float(np.log(1.0 - z * z))

        result: NDArray[np.float64] = np.array(log_det, dtype=np.float64)
        return result
