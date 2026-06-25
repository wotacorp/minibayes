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

"""LKJ Cholesky distribution for correlation matrices."""

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.transforms.base import Transform
from minibayes.transforms.corr_cholesky import CorrCholeskyTransform
from minibayes.utils import ensure_rng


class LKJCholesky(Distribution):
    """
    LKJ prior on correlation matrices via Cholesky factor.

    The LKJ distribution is a prior over correlation matrices. This
    implementation works with the Cholesky factor L where L @ L.T gives
    the correlation matrix.

    The density is:
        p(L | eta) ∝ prod_{k=2}^{d} L[k,k]^(d - k + 2*eta - 2)

    Parameters
    ----------
    dim : int
        Dimension of the correlation matrix (d x d). Must be >= 2.
    eta : float
        Concentration parameter. Default is 1.0.
        - eta = 1: Uniform distribution over correlation matrices
        - eta > 1: Favors correlation matrices closer to identity (less correlation)
        - eta < 1: Favors extreme correlations (not commonly used)

    Attributes
    ----------
    dim : int
        Dimension of the correlation matrix.
    eta : float
        Concentration parameter.

    Raises
    ------
    ModelSpecError
        If dim < 2 or eta <= 0.

    Examples
    --------
    >>> from minibayes import dist
    >>> import numpy as np
    >>> lkj = dist.LKJCholesky(dim=2, eta=2.0)
    >>> L = lkj.sample()  # (2, 2) lower triangular
    >>> corr = L @ L.T    # Correlation matrix
    >>> corr[0, 0], corr[1, 1]  # Diagonal is 1
    (1.0, 1.0)
    """

    @property
    def support(self) -> Support:
        return Support.REAL  # We override default_transform()

    @property
    def mean(self) -> float:
        """Return eta (not meaningful as 'mean' for matrix distribution)."""
        return self._eta

    @property
    def dim(self) -> int:
        """Return dimension of the correlation matrix."""
        return self._dim

    @property
    def eta(self) -> float:
        """Return concentration parameter."""
        return self._eta

    def __init__(self, dim: int, eta: float = 1.0) -> None:
        if dim < 2:
            raise ModelSpecError("dim must be >= 2")
        if eta <= 0:
            raise ModelSpecError("eta must be positive")

        self._dim = dim
        self._eta = eta

    def log_prob(
        self, x: NDArray[np.float64] | float
    ) -> NDArray[np.float64] | float:
        """
        Compute log probability density of Cholesky factor L.

        Parameters
        ----------
        x : ndarray, shape (dim, dim)
            Lower triangular Cholesky factor of a correlation matrix.

        Returns
        -------
        float
            Log probability density (unnormalized).

        Notes
        -----
        The log density is:
            log p(L | eta) = sum_{k=2}^{d} (d - k + 2*eta - 2) * log(L[k,k])

        This is the Cholesky factor parameterization from Stan.
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)

        # Validate shape
        if arr.shape != (self._dim, self._dim):
            raise ModelSpecError(
                f"L must have shape ({self._dim}, {self._dim}), got {arr.shape}"
            )

        # Compute log density from diagonal elements
        # log p(L) = sum_{k=2}^{d} (d - k + 2*eta - 2) * log(L[k,k])
        # In 0-indexed: for k in 1..d-1, exponent = d - (k+1) + 2*eta - 2
        log_p: float = 0.0
        for k in range(1, self._dim):
            exponent: float = self._dim - (k + 1) + 2.0 * self._eta - 2.0
            diag_k: float = float(arr[k, k])  # type: ignore[misc]
            if diag_k <= 0:
                return float("-inf")
            log_p += exponent * float(np.log(diag_k))

        return log_p

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """
        Draw random Cholesky factor(s) from the LKJ distribution.

        Uses the vine method for generating random correlation matrices,
        then returns the Cholesky factor.

        Parameters
        ----------
        size : int, tuple, or None
            Number of samples. If None, return single (dim, dim) matrix.
            If int n, return (n, dim, dim) array.
        rng : Generator, optional
            NumPy random generator.

        Returns
        -------
        ndarray
            Cholesky factor(s). Shape is (dim, dim) if size is None,
            otherwise (size, dim, dim) or (*size, dim, dim).
        """
        generator: np.random.Generator = ensure_rng(rng)

        if size is None:
            return self._sample_one(generator)

        if isinstance(size, int):
            size = (size,)

        n_samples: int = int(np.prod(size))
        samples_list: list[NDArray[np.float64]] = [
            self._sample_one(generator) for _ in range(n_samples)
        ]
        samples: NDArray[np.float64] = np.array(samples_list, dtype=np.float64)
        final_shape: tuple[int, ...] = size + (self._dim, self._dim)
        return samples.reshape(final_shape)

    def _sample_one(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """
        Sample a single Cholesky factor using the vine method.

        This implements the algorithm from Lewandowski, Kurowicka, Joe (2009).
        """
        d = self._dim
        chol: NDArray[np.float64] = np.zeros((d, d), dtype=np.float64)

        # First element is always 1
        chol[0, 0] = 1.0

        for i in range(1, d):
            # Sample partial correlations from Beta distribution
            # Beta(eta + (d-i-1)/2, eta + (d-i-1)/2) gives values in (0, 1)
            # Transform to (-1, 1) using 2*beta - 1
            beta_param: float = self._eta + (d - i - 1) / 2.0

            remaining: float = 1.0
            for j in range(i):
                # Sample partial correlation
                if beta_param > 0:
                    beta_sample: float = float(rng.beta(beta_param, beta_param))
                    # Transform from (0, 1) to (-1, 1)
                    partial_corr: float = 2.0 * beta_sample - 1.0
                else:
                    # For very small beta, sample uniformly
                    partial_corr = 2.0 * float(rng.random()) - 1.0

                # Convert partial correlation to Cholesky element
                chol_ij: float = partial_corr * float(np.sqrt(remaining))
                chol[i, j] = chol_ij
                remaining -= chol_ij * chol_ij
                remaining = max(remaining, 1e-10)

            # Diagonal element
            chol[i, i] = float(np.sqrt(remaining))

        return chol

    def default_transform(self) -> Transform:
        """
        Return transform for this distribution.

        Returns the CorrCholeskyTransform that maps between the constrained
        Cholesky factor space and unconstrained space.

        Returns
        -------
        CorrCholeskyTransform
            Transform for correlation Cholesky factors.
        """
        return CorrCholeskyTransform(self._dim)
