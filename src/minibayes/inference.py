"""MCMC inference engine."""

import sys
import time
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray

from minibayes.exceptions import ModelSpecError, SamplingError, SamplingTimeoutError
from minibayes.model import Model, StructuredParams
from minibayes.results import InferenceResult
from minibayes.samplers import AdaptiveMetropolis, MetropolisHastings
from minibayes.samplers.base import Sampler
from minibayes.utils import ensure_rng
from minibayes.utils.progress import ProgressBar

# Sampler name to class mapping
_SAMPLERS: dict[str, type[Sampler]] = {
    "mh": MetropolisHastings,
    "adaptive_mh": AdaptiveMetropolis,
}


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


def _run_chain(
    sampler: Sampler,
    initial_state: dict[str, float],
    log_prob_fn: Callable[[dict[str, float]], float],
    num_warmup: int,
    num_samples: int,
    rng: np.random.Generator,
    flat_param_names: list[str],
    chain_idx: int,
    num_chains: int,
    progress: bool,
    timeout: float | None,
    start_time: float,
) -> tuple[dict[str, list[float]], int]:
    """
    Run a single MCMC chain.

    Returns samples (flat unconstrained) and acceptance count.

    Raises
    ------
    SamplingTimeoutError
        If timeout is exceeded.
    """
    state: dict[str, float] = initial_state.copy()

    # Timeout check interval (every 100 steps)
    check_interval: int = 100

    def check_timeout() -> None:
        if timeout is not None:
            elapsed: float = time.perf_counter() - start_time
            if elapsed > timeout:
                raise SamplingTimeoutError(f"Sampling timed out after {elapsed:.1f}s (timeout={timeout}s)")

    # Warmup phase
    chain_label: str = f"Chain {chain_idx + 1}/{num_chains}"
    with ProgressBar(
        num_warmup,
        desc=f"{chain_label} [Warmup]  ",
        disable=not progress,
    ) as pbar:
        for step_num in range(num_warmup):
            state, _ = sampler.warmup_step(state, log_prob_fn, rng, step_num)
            pbar.update()
            if step_num % check_interval == 0:
                check_timeout()

    # Post-warmup finalization (e.g., freeze adaptive samplers)
    sampler.post_warmup()

    # Sampling phase
    samples: dict[str, list[float]] = {name: [] for name in flat_param_names}
    accepts: int = 0

    with ProgressBar(
        num_samples,
        desc=f"{chain_label} [Sampling]",
        disable=not progress,
    ) as pbar:
        for step_num in range(num_samples):
            state, accepted = sampler.step(state, log_prob_fn, rng)
            accepts += int(accepted)
            for name in flat_param_names:
                samples[name].append(state[name])
            pbar.update()
            if step_num % check_interval == 0:
                check_timeout()

    return samples, accepts


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
) -> tuple[dict[str, list[float]], int]:
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

    # Run chain (progress=False in workers to avoid stderr conflicts)
    return _run_chain(
        sampler_instance,
        chain_initial,
        log_prob_fn,
        num_warmup,
        num_samples,
        rng,
        flat_param_names,
        chain_idx,
        num_chains,
        progress=False,
        timeout=timeout,
        start_time=start_time,
    )


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
        Number of independent chains.
    parallel : bool
        If True and num_chains > 1, run chains in parallel using processes.
        Requires log_likelihood to be a module-level function (not a lambda).
        Default: False.
    sampler : str
        One of: "mh", "adaptive_mh".
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

    Returns
    -------
    InferenceResult
        Container with samples and diagnostics. Vector parameters
        have shape (num_chains, num_samples, size).

    Raises
    ------
    SamplingTimeoutError
        If timeout is specified and exceeded.
    ModelSpecError
        If sampler name is invalid.
    """
    start_time: float = time.perf_counter()

    # Validate sampler name
    if sampler not in _SAMPLERS:
        valid_samplers: str = ", ".join(_SAMPLERS.keys())
        raise ModelSpecError(f"Unknown sampler '{sampler}'. Valid options: {valid_samplers}")

    # Set up RNG
    rng: np.random.Generator = ensure_rng(seed)

    # Get parameter names from model
    # Samplers work with flat params, results are structured
    flat_param_names: list[str] = model.flat_param_names
    structured_param_names: list[str] = model.param_names
    param_info = model.param_info

    # Build log_prob function
    def log_prob_fn(p: dict[str, float]) -> float:
        return model.log_prob_unconstrained(p, data)

    # Create child RNGs for each chain
    child_rngs: list[np.random.Generator] = list(rng.spawn(num_chains))

    # Run chains (parallel or sequential)
    if parallel and num_chains > 1:
        # Parallel execution using processes (true parallelism)
        # Note: log_likelihood must be a module-level function for pickling
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
            # Workers crashed during unpickling - likely due to unpicklable log_likelihood
            raise SamplingError(
                "Parallel sampling failed: log_likelihood must be a module-level "
                "function defined in an importable .py file (not a lambda, closure, "
                "or function defined in a notebook/__main__). "
                "Use parallel=False for notebooks."
            ) from e
    else:
        # Sequential execution (supports closures/lambdas)
        kwargs: dict[str, object] = sampler_kwargs if sampler_kwargs is not None else {}

        def run_one_chain(chain_idx: int) -> tuple[dict[str, list[float]], int]:
            chain_rng = child_rngs[chain_idx]
            chain_initial = _get_initial_state(model, initial, data, chain_rng)
            sampler_instance: Sampler = _SAMPLERS[sampler](**kwargs)
            return _run_chain(
                sampler_instance,
                chain_initial,
                log_prob_fn,
                num_warmup,
                num_samples,
                chain_rng,
                flat_param_names,
                chain_idx,
                num_chains,
                progress,
                timeout,
                start_time,
            )

        results = [run_one_chain(i) for i in range(num_chains)]

    # Collect flat samples
    all_samples_flat: dict[str, list[list[float]]] = {name: [] for name in flat_param_names}
    acceptance_rates: list[float] = []
    for chain_samples, accepts in results:
        for name in flat_param_names:
            all_samples_flat[name].append(chain_samples[name])
        acceptance_rates.append(accepts / num_samples)

    # Convert flat samples to arrays
    flat_samples_unc: dict[str, NDArray[np.float64]] = {}
    for name in flat_param_names:
        arr: NDArray[np.float64] = np.array(all_samples_flat[name], dtype=np.float64)
        flat_samples_unc[name] = arr  # shape: (num_chains, num_samples)

    # Reconstruct structured samples
    samples_unconstrained: dict[str, NDArray[np.float64]] = {}
    samples_constrained: dict[str, NDArray[np.float64]] = {}

    # Get unconstrained sizes from model (may differ from info.size for matrix params)
    unconstrained_sizes = model._unconstrained_sizes

    for name in structured_param_names:
        info = param_info[name]
        transform = model.transforms[name]

        if info.is_vector:
            # Vector/matrix param: gather theta[0], theta[1], ...
            # Use unconstrained size (may differ for matrix params with transforms)
            unc_size = unconstrained_sizes[name]
            unc_list: list[NDArray[np.float64]] = []
            for i in range(unc_size):
                flat_name = f"{name}[{i}]"
                unc_list.append(flat_samples_unc[flat_name])
            # Stack: each is (chains, samples), stack to (chains, samples, unc_size)
            unc_arr: NDArray[np.float64] = np.stack(unc_list, axis=-1)
            samples_unconstrained[name] = unc_arr
            # Transform: inverse maps unconstrained back to constrained
            # For matrix params, this may reshape from (chains, samples, unc_size) to (chains, samples, d, d)
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

    # Build acceptance rate - always array for consistency
    acceptance_rate: NDArray[np.float64] = np.array(acceptance_rates, dtype=np.float64)

    elapsed_time: float = time.perf_counter() - start_time

    if progress:
        sys.stderr.write(f"Sampling complete in {elapsed_time:.2f}s\n")

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
