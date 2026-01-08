"""Posterior and prior predictive sampling."""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from minibayes.utils import ensure_rng

if TYPE_CHECKING:
    from minibayes.model import Model
    from minibayes.results import InferenceResult


def _get_total_samples(samples: dict[str, NDArray[np.float64]]) -> int:
    """Get total number of samples (flattening chains if multi-chain)."""
    first_key: str = next(iter(samples.keys()))
    sample_arr: NDArray[np.float64] = samples[first_key]
    if sample_arr.ndim == 1:
        return int(sample_arr.shape[0])
    # Multi-chain: (n_chains, n_samples)
    n_chains: int = int(sample_arr.shape[0])
    n_samples: int = int(sample_arr.shape[1])
    return n_chains * n_samples


def _get_param_dict(
    samples: dict[str, NDArray[np.float64]],
    flat_idx: int,
) -> dict[str, float]:
    """
    Extract single parameter dict from samples at given flattened index.

    Handles both single-chain (n_samples,) and multi-chain
    (n_chains, n_samples) shapes by flattening.

    Parameters
    ----------
    samples : dict[str, NDArray[np.float64]]
        Posterior samples.
    flat_idx : int
        Index into flattened samples.

    Returns
    -------
    dict[str, float]
        Parameter values at the given index.
    """
    result: dict[str, float] = {}
    for name, arr in samples.items():
        if arr.ndim == 1:
            result[name] = cast("float", arr[flat_idx])
        else:
            # Multi-chain: flatten and index
            flat_arr: NDArray[np.float64] = arr.ravel()
            result[name] = cast("float", flat_arr[flat_idx])
    return result


def _stack_predictions(
    all_predictions: list[dict[str, NDArray[np.float64]]],
) -> dict[str, NDArray[np.float64]]:
    """Stack list of prediction dicts into single dict with stacked arrays."""
    output: dict[str, NDArray[np.float64]] = {}
    pred_keys: list[str] = list(all_predictions[0].keys())
    for key in pred_keys:
        stacked: NDArray[np.float64] = np.stack(
            [p[key] for p in all_predictions], axis=0
        )
        output[key] = stacked
    return output


def sample_posterior_predictive(
    result: "InferenceResult",
    predictive_fn: Callable[
        [dict[str, float], np.random.Generator], dict[str, ArrayLike]
    ],
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Generate posterior predictive samples.

    For each posterior sample, calls predictive_fn to generate predictions.
    This allows model checking and prediction on new data.

    Parameters
    ----------
    result : InferenceResult
        Posterior samples from MCMC.
    predictive_fn : Callable[[dict[str, float], Generator], dict[str, ArrayLike]]
        Function (params, rng) -> predictions.
        - params: dict of parameter values (e.g., {"mu": 2.5, "sigma": 1.0})
        - rng: numpy random generator for stochastic predictions
        - Returns: dict[str, array] of predictions (must return dict)
    num_samples : int, optional
        Number of posterior samples to use. If None, uses all samples.
        If less than total, samples are thinned uniformly.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Predictions with shape (num_samples, *prediction_shape).

    Examples
    --------
    >>> from minibayes import dist
    >>> def predictive(params, rng):
    ...     mu, sigma = params["mu"], params["sigma"]
    ...     return {"y_pred": dist.Normal(mu, sigma).sample(size=10, rng=rng)}
    >>> ppc = sample_posterior_predictive(result, predictive, seed=42)
    >>> ppc["y_pred"].shape  # (num_samples, 10)
    """
    if not result.samples:
        raise ValueError("No samples in result")

    rng: np.random.Generator = ensure_rng(seed)
    total_samples: int = _get_total_samples(result.samples)

    # Determine indices to use
    if num_samples is None or num_samples >= total_samples:
        indices: NDArray[np.int64] = np.arange(total_samples, dtype=np.int64)
    else:
        # Uniform thinning
        step: int = total_samples // num_samples
        indices = np.arange(0, total_samples, step, dtype=np.int64)[:num_samples]

    # Generate predictions
    all_predictions: list[dict[str, NDArray[np.float64]]] = []
    n_indices: int = len(indices)

    for i in range(n_indices):
        idx: int = cast("int", indices[i])
        params: dict[str, float] = _get_param_dict(result.samples, idx)
        raw_output: dict[str, ArrayLike] = predictive_fn(params, rng)
        pred: dict[str, NDArray[np.float64]] = {
            k: np.asarray(v, dtype=np.float64) for k, v in raw_output.items()
        }
        all_predictions.append(pred)

    return _stack_predictions(all_predictions)


def sample_prior_predictive(
    model: "Model",
    predictive_fn: Callable[
        [dict[str, float], np.random.Generator], dict[str, ArrayLike]
    ],
    num_samples: int = 500,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Generate prior predictive samples.

    Draws parameters from prior, then generates predictions.
    Useful for prior predictive checks before seeing data.

    Parameters
    ----------
    model : Model
        Model with priors defined.
    predictive_fn : Callable[[dict[str, float], Generator], dict[str, ArrayLike]]
        Function (params, rng) -> predictions (must return dict).
    num_samples : int
        Number of prior samples to draw.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Prior predictive samples with shape (num_samples, *prediction_shape).

    Examples
    --------
    >>> from minibayes import dist
    >>> def predictive(params, rng):
    ...     return {"y": dist.Normal(params["mu"], params["sigma"]).sample(size=5, rng=rng)}
    >>> ppc = sample_prior_predictive(model, predictive, num_samples=1000, seed=42)
    """
    rng: np.random.Generator = ensure_rng(seed)

    all_predictions: list[dict[str, NDArray[np.float64]]] = []
    for _ in range(num_samples):
        params: dict[str, float] = model.sample_prior(rng)
        raw_output: dict[str, ArrayLike] = predictive_fn(params, rng)
        pred: dict[str, NDArray[np.float64]] = {
            k: np.asarray(v, dtype=np.float64) for k, v in raw_output.items()
        }
        all_predictions.append(pred)

    return _stack_predictions(all_predictions)
