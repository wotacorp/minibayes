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

"""Export and import utilities for inference results."""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from minibayes.results import InferenceResult


def save_npz(result: InferenceResult, path: str) -> None:
    """
    Save inference result to NumPy compressed archive.

    Parameters
    ----------
    result : InferenceResult
        Results to save.
    path : str
        Output file path.
    """
    # Build arrays for metadata
    num_samples_arr: NDArray[np.int64] = np.array(result.num_samples, dtype=np.int64)
    num_warmup_arr: NDArray[np.int64] = np.array(result.num_warmup, dtype=np.int64)
    num_chains_arr: NDArray[np.int64] = np.array(result.num_chains, dtype=np.int64)
    elapsed_time_arr: NDArray[np.float64] = np.array(result.elapsed_time, dtype=np.float64)

    # Sampler name as unicode string array
    sampler_arr: NDArray[np.str_] = np.array(result.sampler, dtype=np.str_)

    # Acceptance rate - always convert to array
    acc_rate_arr: NDArray[np.float64] = np.asarray(result.acceptance_rate, dtype=np.float64)

    # Build sample arrays with prefixes
    sample_arrays: dict[str, NDArray[np.float64]] = {}
    for name, arr in result.samples.items():
        sample_arrays[f"samples_{name}"] = arr
    for name, arr in result.samples_unconstrained.items():
        sample_arrays[f"unconstrained_{name}"] = arr
    for name, arr in result.derived.items():
        sample_arrays[f"derived_{name}"] = arr

    # np.savez_compressed kwargs type is imprecise in numpy stubs
    np.savez_compressed(
        path,
        num_samples=num_samples_arr,
        num_warmup=num_warmup_arr,
        num_chains=num_chains_arr,
        sampler=sampler_arr,
        elapsed_time=elapsed_time_arr,
        acceptance_rate=acc_rate_arr,
        **sample_arrays,  # type: ignore[arg-type]
    )


def load_npz(path: str) -> InferenceResult:
    """
    Load inference result from NumPy compressed archive.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    InferenceResult
        Loaded results.
    """
    # NpzFile indexing returns Any in numpy stubs
    npz_data = np.load(path, allow_pickle=True)  # type: ignore[misc]
    try:
        samples: dict[str, NDArray[np.float64]] = {}
        samples_unconstrained: dict[str, NDArray[np.float64]] = {}
        derived: dict[str, NDArray[np.float64]] = {}

        file_keys: list[str] = list(npz_data.files)  # type: ignore[misc]
        for key in file_keys:
            if key.startswith("samples_"):
                name: str = key[len("samples_") :]
                samples[name] = cast("NDArray[np.float64]", npz_data[key])  # type: ignore[misc]
            elif key.startswith("unconstrained_"):
                name = key[len("unconstrained_") :]
                samples_unconstrained[name] = cast("NDArray[np.float64]", npz_data[key])  # type: ignore[misc]
            elif key.startswith("derived_"):
                name = key[len("derived_") :]
                derived[name] = cast("NDArray[np.float64]", npz_data[key])  # type: ignore[misc]

        acceptance_rate: NDArray[np.float64] = np.atleast_1d(
            cast("NDArray[np.float64]", npz_data["acceptance_rate"])  # type: ignore[misc]
        )

        return InferenceResult(
            samples=samples,
            samples_unconstrained=samples_unconstrained,
            acceptance_rate=acceptance_rate,
            num_samples=int(cast("np.int64", npz_data["num_samples"])),  # type: ignore[misc]
            num_warmup=int(cast("np.int64", npz_data["num_warmup"])),  # type: ignore[misc]
            num_chains=int(cast("np.int64", npz_data["num_chains"])),  # type: ignore[misc]
            sampler=str(cast("np.str_", npz_data["sampler"])),  # type: ignore[misc]
            elapsed_time=float(cast("np.float64", npz_data["elapsed_time"])),  # type: ignore[misc]
            derived=derived,
        )
    finally:
        npz_data.close()  # type: ignore[misc]


def to_json(result: InferenceResult) -> dict[str, object]:
    """
    Convert inference result to JSON-serializable dict.

    Parameters
    ----------
    result : InferenceResult
        Results to convert.

    Returns
    -------
    dict
        JSON-serializable dictionary.
    """
    # Convert acceptance_rate to JSON-serializable list
    acc_rate: list[float] = cast("list[float]", result.acceptance_rate.tolist())

    # Build samples dicts
    samples_dict: dict[str, list[float]] = {}
    for k, v in result.samples.items():
        samples_dict[k] = cast("list[float]", v.tolist())

    samples_unc_dict: dict[str, list[float]] = {}
    for k, v in result.samples_unconstrained.items():
        samples_unc_dict[k] = cast("list[float]", v.tolist())

    derived_dict: dict[str, list[float]] = {}
    for k, v in result.derived.items():
        derived_dict[k] = cast("list[float]", v.tolist())

    return {
        "samples": samples_dict,
        "samples_unconstrained": samples_unc_dict,
        "derived": derived_dict,
        "acceptance_rate": acc_rate,
        "num_samples": result.num_samples,
        "num_warmup": result.num_warmup,
        "num_chains": result.num_chains,
        "sampler": result.sampler,
        "elapsed_time": result.elapsed_time,
    }
