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

"""Multivariate Normal (Gaussian) distribution."""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution, Support
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng

_LOG_2PI: float = float(np.log(2.0 * np.pi))


class MultivariateNormal(Distribution):
    """
    Multivariate Normal (Gaussian) distribution.

    This distribution is intended for use as a likelihood (observation model),
    not as a prior for sampled parameters.

    The probability density function is:
        p(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 (x-μ)ᵀ Σ⁻¹ (x-μ))

    Parameters
    ----------
    mean : ndarray, shape (d,)
        Mean vector of the distribution (μ).
    cov : ndarray, shape (d, d)
        Covariance matrix (Σ). Must be symmetric positive definite.

    Attributes
    ----------
    dim : int
        Dimensionality of the distribution.

    Raises
    ------
    ModelSpecError
        If mean is not 1D.
        If cov is not square or doesn't match mean dimension.
        If cov is not positive definite.
    """

    @property
    def support(self) -> Support:
        return Support.REAL

    @property
    def mean(self) -> NDArray[np.float64]:  # type: ignore[override]
        """Return the mean vector."""
        return self._mean.copy()

    @property
    def dim(self) -> int:
        """Return dimensionality of the distribution."""
        return self._dim

    @property
    def cov(self) -> NDArray[np.float64]:
        """Return the covariance matrix."""
        return self._cov.copy()

    def __init__(
        self,
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> None:
        mean_arr: NDArray[np.float64] = np.asarray(mean, dtype=np.float64)
        if mean_arr.ndim != 1:
            raise ModelSpecError(f"mean must be 1D, got shape {mean_arr.shape}")
        self._mean = mean_arr
        self._dim = len(mean_arr)

        cov_arr: NDArray[np.float64] = np.asarray(cov, dtype=np.float64)
        if cov_arr.shape != (self._dim, self._dim):
            raise ModelSpecError(f"cov must be ({self._dim}, {self._dim}), got {cov_arr.shape}")
        self._cov = cov_arr

        try:
            chol_raw = np.linalg.cholesky(cov_arr)  # type: ignore[misc]
            self._chol: NDArray[np.float64] = np.asarray(chol_raw, dtype=np.float64)  # type: ignore[misc]
        except np.linalg.LinAlgError as e:
            raise ModelSpecError("cov must be symmetric positive definite") from e

        # Precompute log normalizer: -0.5 * (d * log(2π) + log|Σ|)
        # log|Σ| = 2 * sum(log(diag(L)))
        diag_chol: NDArray[np.float64] = np.diag(self._chol)
        log_diag: NDArray[np.float64] = np.log(diag_chol)
        log_det: float = 2.0 * cast("float", np.sum(log_diag))
        self._log_normalizer: float = -0.5 * (self._dim * _LOG_2PI + log_det)

    def log_prob(  # type: ignore[override]
        self, x: NDArray[np.float64]
    ) -> NDArray[np.float64] | float:
        """
        Compute log probability density at x.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate log probability.
            - Single observation: shape (d,)
            - Batch of observations: shape (n, d)

        Returns
        -------
        ndarray or float
            Log probability density value(s).
            - Single observation: returns float
            - Batch: returns ndarray of shape (n,)
        """
        arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)

        if arr.ndim == 1:
            dim0: int = cast("int", arr.shape[0])
            if dim0 != self._dim:
                raise ModelSpecError(f"x must have dimension {self._dim}, got {dim0}")
            diff: NDArray[np.float64] = arr - self._mean
            z_raw = np.linalg.solve(self._chol, diff)  # type: ignore[misc]
            z: NDArray[np.float64] = np.asarray(z_raw, dtype=np.float64)  # type: ignore[misc]
            maha_sq: float = cast("float", np.dot(z, z))
            return self._log_normalizer - 0.5 * maha_sq

        if arr.ndim == 2:
            dim1: int = cast("int", arr.shape[1])
            if dim1 != self._dim:
                raise ModelSpecError(f"x must have dimension {self._dim} in last axis, got {dim1}")
            diff_batch: NDArray[np.float64] = arr - self._mean  # (n, d)
            z_raw_batch = np.linalg.solve(self._chol, diff_batch.T).T  # type: ignore[misc]
            z_batch: NDArray[np.float64] = np.asarray(z_raw_batch, dtype=np.float64)  # type: ignore[misc]
            z_sq: NDArray[np.float64] = z_batch * z_batch
            maha_sq_arr: NDArray[np.float64] = np.sum(z_sq, axis=1)  # (n,)
            result: NDArray[np.float64] = self._log_normalizer - 0.5 * maha_sq_arr
            return result

        raise ModelSpecError(f"x must be 1D or 2D, got {arr.ndim}D")

    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """
        Draw random samples from the distribution.

        Parameters
        ----------
        size : int, tuple, or None
            Number of samples to draw.
            - None: return single sample, shape (d,)
            - int n: return n samples, shape (n, d)
            - tuple (n, m): return shape (n, m, d)
        rng : Generator, optional
            NumPy random generator.

        Returns
        -------
        ndarray
            Random samples. Last dimension is always d.
        """
        generator: np.random.Generator = ensure_rng(rng)

        if size is None:
            z_raw = generator.standard_normal(self._dim)
            z: NDArray[np.float64] = np.asarray(z_raw, dtype=np.float64)
            chol_z: NDArray[np.float64] = self._chol @ z
            sample: NDArray[np.float64] = self._mean + chol_z
            return sample

        if isinstance(size, int):
            size = (size,)

        z_raw_batch = generator.standard_normal(size + (self._dim,))
        z_batch: NDArray[np.float64] = np.asarray(z_raw_batch, dtype=np.float64)
        einsum_result = np.einsum("ij,...j->...i", self._chol, z_batch)  # type: ignore[misc]
        transformed: NDArray[np.float64] = np.asarray(einsum_result, dtype=np.float64)  # type: ignore[misc]
        samples: NDArray[np.float64] = self._mean + transformed
        return samples
