"""MCMC inference engine."""

from typing import Any, Callable

from minibayes.model import Model
from minibayes.results import InferenceResult


def sample(
    model: Model | Callable[[dict[str, float], Any], float],
    data: Any = None,
    initial: dict[str, float] | None = None,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    sampler: str = "adaptive_mh",
    sampler_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> InferenceResult:
    """
    Run MCMC sampling.

    Parameters
    ----------
    model : Model or Callable
        Either a Model instance, or a function with signature:
        (params: dict[str, float], data: Any) -> float
    data : Any
        Observed data passed to likelihood.
    initial : dict, optional
        Initial parameter values. If None, sampled from prior (Model)
        or must be provided (Callable).
    num_samples : int
        Number of samples to draw (post-warmup).
    num_warmup : int
        Number of warmup/tuning samples (discarded).
    num_chains : int
        Number of independent chains.
    sampler : str
        One of: "mh", "adaptive_mh".
    sampler_kwargs : dict, optional
        Additional arguments passed to sampler.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    InferenceResult
        Container with samples and diagnostics.
    """
    raise NotImplementedError()
