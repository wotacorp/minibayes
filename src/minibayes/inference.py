"""MCMC inference engine."""

import sys
import time
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray

from minibayes.exceptions import ModelSpecError, SamplingError, SamplingTimeoutError
from minibayes.model import Model, StructuredParams
from minibayes.params import ParamInfo
from minibayes.results import InferenceResult
from minibayes.samplers import AdaptiveMetropolis, EnsembleSampler, MetropolisHastings
from minibayes.samplers.base import Sampler
from minibayes.utils import ensure_rng
from minibayes.utils.progress import ProgressBar

# Sampler name to class mapping
_SAMPLERS: dict[str, type[Sampler]] = {
    "mh": MetropolisHastings,
    "adaptive_mh": AdaptiveMetropolis,
    "ensemble": EnsembleSampler,
}

# Default memory warning threshold in MB
_DEFAULT_MEMORY_WARNING_MB: float = 1000.0

# Default maximum samples limit
_DEFAULT_MAX_SAMPLES: int = 100_000


def _estimate_memory_mb(num_params: int, num_samples: int, num_chains: int) -> float:
    """
    Estimate memory usage in MB for sampling results.

    Parameters
    ----------
    num_params : int
        Number of flat (scalar) parameters.
    num_samples : int
        Number of samples per chain.
    num_chains : int
        Number of chains.

    Returns
    -------
    float
        Estimated memory usage in megabytes.
    """
    # Each scalar param stores: (num_chains, num_samples) float64 array
    # We store both constrained and unconstrained versions (2x)
    bytes_per_param: int = num_chains * num_samples * 8 * 2
    return (num_params * bytes_per_param) / (1024 * 1024)


def _get_initial_state(
    model: Model,
    initial: StructuredParams | None,
    data: object,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Get initial state in flat unconstrained space.

    Uses prior means if initial is None (deterministic, robust).
    Falls back to sampling from prior if means give invalid log_prob.
    """
    if initial is None:
        # Use prior means as default (deterministic, robust)
        constrained: StructuredParams = model.prior_means()
        unconstrained: dict[str, float] = model.to_flat_unconstrained(constrained)
        lp: float = model.log_prob_unconstrained(unconstrained, data)
        if np.isfinite(lp):
            return unconstrained
        # Fallback: sample from prior if means give -inf log_prob
        max_attempts: int = 10
        for _ in range(max_attempts):
            constrained = model.sample_prior(rng)
            unconstrained = model.to_flat_unconstrained(constrained)
            lp = model.log_prob_unconstrained(unconstrained, data)
            if np.isfinite(lp):
                return unconstrained
        raise SamplingError(f"Could not find valid initial state after {max_attempts} attempts")
    else:
        # Validate and transform provided initial values
        model.validate_params(initial)
        return model.to_flat_unconstrained(initial)


def _run_sampler(
    sampler: Sampler,
    log_prob_fn: Callable[[dict[str, float]], float],
    num_warmup: int,
    num_samples: int,
    rng: np.random.Generator,
    flat_param_names: list[str],
    progress: bool,
    progress_prefix: str,
    timeout: float | None,
    start_time: float,
) -> tuple[dict[str, NDArray[np.float64]], float]:
    """
    Run sampler using stateful interface.

    Works for both single-chain and ensemble samplers.

    Returns
    -------
    tuple[dict[str, NDArray[np.float64]], float]
        Samples with shape (num_chains, num_samples) per param, and avg acceptance rate.
    """
    check_interval: int = 100

    def check_timeout() -> None:
        if timeout is not None:
            elapsed: float = time.perf_counter() - start_time
            if elapsed > timeout:
                raise SamplingTimeoutError(
                    f"Sampling timed out after {elapsed:.1f}s (timeout={timeout}s)"
                )

    # Warmup phase
    with ProgressBar(num_warmup, desc=f"{progress_prefix}[Warmup]  ", disable=not progress) as pbar:
        for step_num in range(num_warmup):
            sampler.advance(log_prob_fn, rng, warmup=True, step_num=step_num)
            pbar.update()
            if step_num % check_interval == 0:
                check_timeout()

    # Post-warmup finalization
    sampler.post_warmup()

    # Sampling phase - pre-allocate arrays for memory efficiency
    num_chains: int = sampler.num_chains
    samples: dict[str, NDArray[np.float64]] = {
        name: np.empty((num_chains, num_samples), dtype=np.float64)
        for name in flat_param_names
    }
    total_accept_rate: float = 0.0

    with ProgressBar(num_samples, desc=f"{progress_prefix}[Sampling]", disable=not progress) as pbar:
        for step_num in range(num_samples):
            accept_rate: float = sampler.advance(log_prob_fn, rng)
            total_accept_rate += accept_rate
            states: list[dict[str, float]] = sampler.get_states()
            for k, state in enumerate(states):
                for name in flat_param_names:
                    samples[name][k, step_num] = state[name]
            pbar.update()
            if step_num % check_interval == 0:
                check_timeout()

    avg_acceptance: float = total_accept_rate / num_samples if num_samples > 0 else 0.0
    return samples, avg_acceptance


def _run_chain_worker(
    args: tuple[
        "Model",
        object,
        StructuredParams | None,
        str,
        dict[str, object] | None,
        int,
        int,
        np.random.Generator,
        list[str],
        int,
        int,
        float | None,
        float,
    ],
) -> tuple[dict[str, NDArray[np.float64]], float]:
    """
    Worker function for parallel chain execution.

    This is a module-level function (not a closure) so it can be pickled
    for multiprocessing.
    """
    (
        model,
        data,
        initial,
        sampler_name,
        sampler_kwargs,
        num_warmup,
        num_samples,
        rng,
        flat_param_names,
        chain_idx,
        num_chains,
        timeout,
        start_time,
    ) = args

    # Build log_prob_fn locally (not a closure from outer scope)
    def log_prob_fn(p: dict[str, float]) -> float:
        return model.log_prob_unconstrained(p, data)

    # Get initial state
    chain_initial: dict[str, float] = _get_initial_state(model, initial, data, rng)

    # Create sampler
    kwargs: dict[str, object] = sampler_kwargs if sampler_kwargs is not None else {}
    sampler_instance: Sampler = _SAMPLERS[sampler_name](**kwargs)
    sampler_instance.initialize([chain_initial], log_prob_fn)

    # Run chain
    chain_samples, accept_rate = _run_sampler(
        sampler_instance,
        log_prob_fn,
        num_warmup,
        num_samples,
        rng,
        flat_param_names,
        progress=False,  # Disable progress in workers
        progress_prefix="",
        timeout=timeout,
        start_time=start_time,
    )

    # Extract single chain from (1, num_samples) to (num_samples,)
    flat_samples: dict[str, NDArray[np.float64]] = {
        name: chain_samples[name][0, :] for name in flat_param_names
    }
    return flat_samples, accept_rate


def sample(
    model: Model,
    data: object = None,
    initial: StructuredParams | None = None,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    parallel: bool = False,
    sampler: str = "adaptive_mh",
    sampler_kwargs: dict[str, object] | None = None,
    seed: int | None = None,
    progress: bool = False,
    timeout: float | None = None,
    max_samples: int | None = _DEFAULT_MAX_SAMPLES,
    max_memory_mb: float | None = _DEFAULT_MEMORY_WARNING_MB,
) -> InferenceResult:
    """
    Run MCMC sampling.

    Parameters
    ----------
    model : Model
        A Model instance with priors and likelihood.
    data : Any
        Observed data passed to likelihood.
    initial : dict, optional
        Initial parameter values (constrained space). If None, sampled
        from prior.
    num_samples : int
        Number of samples to draw (post-warmup).
    num_warmup : int
        Number of warmup/tuning samples (discarded).
    num_chains : int
        Number of independent chains. For ensemble sampler, this is the
        number of walkers (must be even and >= 2*ndim).
    parallel : bool
        If True and num_chains > 1, run chains in parallel using processes.
        Requires log_likelihood to be a module-level function (not a lambda).
        Ignored for ensemble sampler (walkers run in single process).
        Default: False.
    sampler : str
        One of: "mh", "adaptive_mh", "ensemble".
    sampler_kwargs : dict, optional
        Additional arguments passed to sampler.
    seed : int, optional
        Random seed for reproducibility.
    progress : bool
        If True, display progress bars for warmup and sampling phases.
        Default: False.
    timeout : float, optional
        Maximum time in seconds for sampling. If exceeded, raises
        SamplingTimeoutError. Default: None (no timeout).
    max_samples : int, optional
        Maximum allowed num_samples. Raises ValueError if exceeded.
        Default: 100,000. Set to None to disable this safety limit.
    max_memory_mb : float, optional
        Maximum estimated memory usage in MB. Raises MemoryError if exceeded.
        Default: 1000 MB. Set to None to disable this safety limit.

    Returns
    -------
    InferenceResult
        Container with samples and diagnostics. Vector parameters
        have shape (num_chains, num_samples, size).

    Raises
    ------
    ValueError
        If num_samples exceeds max_samples.
    MemoryError
        If estimated memory usage exceeds max_memory_mb.
    SamplingTimeoutError
        If timeout is specified and exceeded.
    ModelSpecError
        If sampler name is invalid.
    """
    start_time: float = time.perf_counter()

    # Validate max_samples limit
    if max_samples is not None and num_samples > max_samples:
        raise ValueError(
            f"num_samples ({num_samples}) exceeds max_samples ({max_samples}). "
            f"Set max_samples=None to disable this limit."
        )

    # Validate memory usage limit
    num_flat_params: int = len(model.flat_param_names)
    estimated_mb: float = _estimate_memory_mb(num_flat_params, num_samples, num_chains)
    if max_memory_mb is not None and estimated_mb > max_memory_mb:
        raise MemoryError(
            f"Estimated memory usage ({estimated_mb:.0f} MB) exceeds "
            f"max_memory_mb ({max_memory_mb:.0f} MB). "
            f"Reduce num_samples/num_chains, or set max_memory_mb=None to disable."
        )

    # Validate sampler name
    if sampler not in _SAMPLERS:
        valid_samplers: str = ", ".join(_SAMPLERS.keys())
        raise ModelSpecError(f"Unknown sampler '{sampler}'. Valid options: {valid_samplers}")

    # Set up RNG
    rng: np.random.Generator = ensure_rng(seed)

    # Get parameter names from model
    flat_param_names: list[str] = model.flat_param_names
    structured_param_names: list[str] = model.param_names
    param_info = model.param_info

    # Build log_prob function
    def log_prob_fn(p: dict[str, float]) -> float:
        return model.log_prob_unconstrained(p, data)

    # Create sampler
    kwargs: dict[str, object] = sampler_kwargs if sampler_kwargs is not None else {}
    sampler_instance: Sampler = _SAMPLERS[sampler](**kwargs)

    # Determine number of chains/walkers and initialize
    if sampler == "ensemble":
        ndim: int = len(flat_param_names)
        # Default to 2*ndim walkers if num_chains == 1
        num_walkers: int = num_chains if num_chains > 1 else max(2 * ndim, 4)
        # Validate
        if num_walkers % 2 != 0:
            raise ModelSpecError("num_chains must be even for ensemble sampler")
        if num_walkers < 2 * ndim:
            raise ModelSpecError(
                f"num_chains must be >= 2*ndim ({2 * ndim}) for ensemble sampler"
            )
        # Initialize from prior samples
        initial_states: list[dict[str, float]] = []
        for _ in range(num_walkers):
            constrained: StructuredParams = model.sample_prior(rng)
            initial_states.append(model.to_flat_unconstrained(constrained))
        sampler_instance.initialize(initial_states, log_prob_fn)
        num_chains = num_walkers  # Update for result
        progress_prefix: str = ""
    else:
        # Single-chain samplers: run each chain independently
        if num_chains > 1:
            all_samples_flat: dict[str, list[NDArray[np.float64]]] = {
                name: [] for name in flat_param_names
            }
            acceptance_rates: list[float] = []
            child_rngs: list[np.random.Generator] = list(rng.spawn(num_chains))

            if parallel:
                # Parallel execution using processes
                worker_args = [
                    (
                        model,
                        data,
                        initial,
                        sampler,
                        sampler_kwargs,
                        num_warmup,
                        num_samples,
                        child_rngs[i],
                        flat_param_names,
                        i,
                        num_chains,
                        timeout,
                        start_time,
                    )
                    for i in range(num_chains)
                ]

                if progress:
                    sys.stderr.write(f"Running {num_chains} chains in parallel...\n")

                try:
                    with ProcessPoolExecutor(max_workers=num_chains) as executor:
                        results = list(executor.map(_run_chain_worker, worker_args))
                except BrokenExecutor as e:
                    raise SamplingError(
                        "Parallel sampling failed: log_likelihood must be a module-level "
                        "function defined in an importable .py file (not a lambda, closure, "
                        "or function defined in a notebook/__main__). "
                        "Use parallel=False for notebooks."
                    ) from e

                for chain_samples, accept_rate in results:
                    for name in flat_param_names:
                        all_samples_flat[name].append(chain_samples[name])
                    acceptance_rates.append(accept_rate)
            else:
                # Sequential execution
                for chain_idx in range(num_chains):
                    chain_rng = child_rngs[chain_idx]
                    chain_initial = _get_initial_state(model, initial, data, chain_rng)
                    chain_sampler: Sampler = _SAMPLERS[sampler](**kwargs)
                    chain_sampler.initialize([chain_initial], log_prob_fn)

                    chain_result: tuple[dict[str, NDArray[np.float64]], float] = _run_sampler(
                        chain_sampler,
                        log_prob_fn,
                        num_warmup,
                        num_samples,
                        chain_rng,
                        flat_param_names,
                        progress,
                        f"Chain {chain_idx + 1}/{num_chains} ",
                        timeout,
                        start_time,
                    )
                    chain_samples_arr: dict[str, NDArray[np.float64]] = chain_result[0]
                    chain_accept_rate: float = chain_result[1]

                    for name in flat_param_names:
                        # Extract single chain from (1, num_samples) to (num_samples,)
                        all_samples_flat[name].append(chain_samples_arr[name][0, :])
                    acceptance_rates.append(chain_accept_rate)

            # Return early since we already ran all chains
            elapsed_time: float = time.perf_counter() - start_time
            if progress:
                sys.stderr.write(f"Sampling complete in {elapsed_time:.2f}s\n")

            return _build_result(
                all_samples_flat,
                acceptance_rates,
                model,
                structured_param_names,
                flat_param_names,
                param_info,
                num_samples,
                num_warmup,
                num_chains,
                sampler,
                elapsed_time,
            )

        # Single chain: initialize and run
        chain_initial = _get_initial_state(model, initial, data, rng)
        sampler_instance.initialize([chain_initial], log_prob_fn)
        progress_prefix = ""

    # Run sampler (unified path for ensemble and single chain with num_chains=1)
    sampler_samples, avg_acceptance = _run_sampler(
        sampler_instance,
        log_prob_fn,
        num_warmup,
        num_samples,
        rng,
        flat_param_names,
        progress,
        progress_prefix,
        timeout,
        start_time,
    )

    # Build acceptance rates array
    acceptance_rates_arr: list[float] = [avg_acceptance] * num_chains

    elapsed_time = time.perf_counter() - start_time
    if progress:
        sys.stderr.write(f"Sampling complete in {elapsed_time:.2f}s\n")

    return _build_result(
        sampler_samples,
        acceptance_rates_arr,
        model,
        structured_param_names,
        flat_param_names,
        param_info,
        num_samples,
        num_warmup,
        num_chains,
        sampler,
        elapsed_time,
    )


def _build_result(
    all_samples_flat: dict[str, list[NDArray[np.float64]]] | dict[str, NDArray[np.float64]],
    acceptance_rates: list[float],
    model: Model,
    structured_param_names: list[str],
    flat_param_names: list[str],
    param_info: dict[str, ParamInfo],
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    sampler: str,
    elapsed_time: float,
) -> InferenceResult:
    """Build InferenceResult from flat samples."""
    # Convert flat samples to arrays
    flat_samples_unc: dict[str, NDArray[np.float64]] = {}
    for name in flat_param_names:
        sample_data: list[NDArray[np.float64]] | NDArray[np.float64] = all_samples_flat[name]
        if isinstance(sample_data, list):
            # List of arrays (from multi-chain path) - stack into 2D
            arr: NDArray[np.float64] = np.stack(sample_data, axis=0)
        else:
            # Already an array (from single-chain/ensemble path)
            arr = sample_data
        flat_samples_unc[name] = arr  # shape: (num_chains, num_samples)

    # Reconstruct structured samples
    samples_unconstrained: dict[str, NDArray[np.float64]] = {}
    samples_constrained: dict[str, NDArray[np.float64]] = {}

    # Get unconstrained sizes from model
    unconstrained_sizes = model._unconstrained_sizes

    for name in structured_param_names:
        info = param_info[name]
        transform = model.transforms[name]

        if info.is_vector:
            # Vector/matrix param: gather theta[0], theta[1], ...
            unc_size = unconstrained_sizes[name]
            unc_list: list[NDArray[np.float64]] = []
            for i in range(unc_size):
                flat_name = f"{name}[{i}]"
                unc_list.append(flat_samples_unc[flat_name])
            unc_arr: NDArray[np.float64] = np.stack(unc_list, axis=-1)
            samples_unconstrained[name] = unc_arr
            # Transform to constrained
            constrained_samples: list[NDArray[np.float64]] = []
            for chain_idx in range(num_chains):
                chain_constrained: list[NDArray[np.float64]] = []
                for sample_idx in range(num_samples):
                    unc_sample = unc_arr[chain_idx, sample_idx, :]
                    const_sample: NDArray[np.float64] = transform.inverse(unc_sample)
                    chain_constrained.append(const_sample)
                chain_arr: NDArray[np.float64] = np.array(chain_constrained, dtype=np.float64)
                constrained_samples.append(chain_arr)
            samples_constrained[name] = np.array(constrained_samples, dtype=np.float64)
        else:
            # Scalar param: shape (chains, samples)
            unc_arr = flat_samples_unc[name]
            samples_unconstrained[name] = unc_arr
            samples_constrained[name] = transform.inverse(unc_arr)

    # Build acceptance rate array
    acceptance_rate: NDArray[np.float64] = np.array(acceptance_rates, dtype=np.float64)

    return InferenceResult(
        samples=samples_constrained,
        samples_unconstrained=samples_unconstrained,
        acceptance_rate=acceptance_rate,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        sampler=sampler,
        elapsed_time=elapsed_time,
    )
