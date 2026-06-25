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

"""Internal utilities for visualization functions."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from minibayes.results import InferenceResult


def extract_samples(
    data: InferenceResult | dict[str, NDArray[np.float64]] | NDArray[np.float64],
    params: list[str] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Extract samples dictionary from various input types.

    Parameters
    ----------
    data : InferenceResult, dict, or ndarray
        Input data. If InferenceResult, uses .samples attribute.
        If dict, uses directly. If ndarray, wraps as {"param": data}.
    params : list[str], optional
        Parameters to include. None means all.

    Returns
    -------
    dict[str, ndarray]
        Samples dictionary with shape (num_chains, num_samples) per param.
    """
    # Import here to avoid circular imports
    from minibayes.results import InferenceResult

    # Extract samples dict (merge samples + derived for InferenceResult)
    samples: dict[str, NDArray[np.float64]]
    if isinstance(data, InferenceResult):
        samples = {**data.samples, **data.derived}
    elif isinstance(data, dict):
        samples = data
    else:
        # Must be ndarray
        arr: NDArray[np.float64] = np.asarray(data, dtype=np.float64)
        samples = {"param": arr}

    # Filter by params if specified
    if params is not None:
        samples = {k: v for k, v in samples.items() if k in params}

    # Ensure all are 2D (chains, samples), skip higher dimensions with warning
    result: dict[str, NDArray[np.float64]] = {}
    for name, arr in samples.items():
        if arr.ndim > 2:
            warnings.warn(
                f"Skipping '{name}': {arr.ndim}D arrays not supported in plots "
                "(use add_derived() to extract scalars)",
                stacklevel=3,
            )
            continue
        result[name] = ensure_2d(arr)

    return result


def ensure_2d(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Ensure array is 2D with shape (num_chains, num_samples).

    Parameters
    ----------
    arr : ndarray
        Input array, 1D or 2D.

    Returns
    -------
    ndarray
        2D array with shape (num_chains, num_samples).
    """
    if arr.ndim == 1:
        reshaped: NDArray[np.float64] = arr.reshape(1, -1)
        return reshaped
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def flatten_samples(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Flatten 2D samples array to 1D.

    Parameters
    ----------
    arr : ndarray
        Array with shape (num_chains, num_samples).

    Returns
    -------
    ndarray
        1D array with all samples.
    """
    flat: NDArray[np.float64] = arr.flatten()
    return flat
