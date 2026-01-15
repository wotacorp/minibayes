"""Convergence diagnostics for MCMC."""

import warnings
from typing import cast

import numpy as np
from numpy.typing import NDArray

from minibayes.exceptions import ConvergenceWarning


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    return 1 << (n - 1).bit_length()


def effective_sample_size(samples: NDArray[np.float64]) -> float:
    """
    Compute effective sample size accounting for autocorrelation.

    Uses FFT-based autocorrelation estimation with Sokal's
    automatic windowing criterion.

    Parameters
    ----------
    samples : ndarray
        1D array of MCMC samples.

    Returns
    -------
    float
        Effective sample size. Always >= 1 and <= len(samples).

    References
    ----------
    - Sokal (1997) "Monte Carlo Methods in Statistical Mechanics"
    - emcee: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    """
    n: int = len(samples)
    if n < 4:
        return float(n)

    # Center the samples
    mean: float = cast("float", np.mean(samples))
    x: NDArray[np.float64] = samples - mean

    # FFT-based autocorrelation
    n_fft: int = _next_power_of_two(2 * n)
    f: NDArray[np.complex128] = np.fft.fft(x, n=n_fft)
    f_conj: NDArray[np.complex128] = np.conjugate(f)
    product: NDArray[np.complex128] = f * f_conj
    acf_full: NDArray[np.complex128] = np.fft.ifft(product)
    acf: NDArray[np.float64] = acf_full[:n].real

    # Normalize by variance (acf[0])
    acf_zero: float = cast("float", acf[0])
    if acf_zero <= 0:
        return float(n)
    acf_normalized: NDArray[np.float64] = acf / acf_zero

    # Integrated autocorrelation time: tau = 1 + 2 * sum(rho_k)
    # Cumulative sum gives tau(M) = 1 + 2 * sum_{k=1}^{M} rho_k
    cumsum: NDArray[np.float64] = np.cumsum(acf_normalized)
    taus: NDArray[np.float64] = 2.0 * cumsum - 1.0

    # Sokal's automatic windowing: find smallest M where M >= c * tau(M)
    # c ~ 5 is recommended
    c: float = 5.0
    window: int = 0
    for m in range(1, n):
        tau_m: float = cast("float", taus[m])
        if m >= c * tau_m:
            window = m
            break
    else:
        # No window found, use last value
        window = n - 1

    tau: float = cast("float", taus[window])

    # ESS = n / tau, clipped to [1, n]
    if tau <= 0:
        return float(n)
    ess: float = n / tau
    return max(1.0, min(ess, float(n)))


def r_hat(chains: NDArray[np.float64]) -> float:
    """
    Compute Gelman-Rubin R-hat diagnostic.

    Values > 1.01 indicate non-convergence. Values close to 1.0
    suggest chains have converged to the same distribution.

    Parameters
    ----------
    chains : ndarray
        Array of shape (num_chains, num_samples).

    Returns
    -------
    float
        R-hat statistic. Returns NaN if num_chains < 2 or
        within-chain variance is zero.

    References
    ----------
    - Gelman & Rubin (1992) "Inference from Iterative Simulation
      Using Multiple Sequences"
    """
    if chains.ndim != 2:
        raise ValueError(f"chains must be 2D, got shape {chains.shape}")

    num_chains: int = chains.shape[0]
    n: int = chains.shape[1]

    if num_chains < 2:
        return float("nan")
    if n < 2:
        return float("nan")

    # Chain means
    chain_means: NDArray[np.float64] = np.mean(chains, axis=1)

    # Between-chain variance B
    # B = n * Var(chain_means) where Var uses ddof=1
    b_var: float = cast("float", np.var(chain_means, ddof=1))
    b: float = n * b_var

    # Within-chain variance W = mean of chain variances
    chain_vars: NDArray[np.float64] = np.var(chains, axis=1, ddof=1)
    w: float = cast("float", np.mean(chain_vars))

    if w <= 0:
        return float("nan")

    # Marginal posterior variance estimate
    # var_hat = (n-1)/n * W + (1/n) * B
    var_hat: float = ((n - 1) / n) * w + (1 / n) * b

    # R-hat = sqrt(var_hat / W)
    ratio: float = var_hat / w
    r_hat_val: float = float(np.sqrt(ratio))
    return r_hat_val


def _summarize_2d(
    name: str,
    arr: NDArray[np.float64],
    percentiles: list[int],
) -> dict[str, float]:
    """Summarize a 2D array (num_chains, num_samples)."""
    flat: NDArray[np.float64] = arr.flatten()

    stats: dict[str, float] = {
        "mean": cast("float", np.mean(flat)),
        "std": cast("float", np.std(flat, ddof=1)),
    }

    # Compute percentiles
    for p in percentiles:
        key: str = f"{p}%"
        stats[key] = cast("float", np.percentile(flat, p))

    # ESS: compute per chain, average
    num_chains: int = arr.shape[0]
    ess_values: list[float] = []
    for i in range(num_chains):
        chain_samples: NDArray[np.float64] = arr[i, :]
        ess_values.append(effective_sample_size(chain_samples))
    ess_arr: NDArray[np.float64] = np.asarray(ess_values, dtype=np.float64)
    stats["ess"] = cast("float", np.mean(ess_arr))

    # R-hat: always include (NaN for single chain)
    stats["r_hat"] = r_hat(arr)

    return stats


def _warn_if_non_converged(name: str, stats: dict[str, float]) -> None:
    """Emit ConvergenceWarning if diagnostics suggest non-convergence."""
    r_hat_val: float = stats["r_hat"]
    if r_hat_val > 1.01 and not np.isnan(r_hat_val):
        warnings.warn(
            f"R-hat for {name} is {r_hat_val:.3f} (>1.01), suggesting non-convergence",
            ConvergenceWarning,
            stacklevel=4,
        )
    ess_val: float = stats["ess"]
    if ess_val < 100:
        warnings.warn(
            f"ESS for {name} is {ess_val:.1f} (<100), samples may be unreliable",
            ConvergenceWarning,
            stacklevel=4,
        )


def summary(
    samples: dict[str, NDArray[np.float64]],
    percentiles: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute summary statistics for all parameters.

    Parameters
    ----------
    samples : dict[str, ndarray]
        Samples for each parameter.
        Scalars: shape (num_chains, num_samples).
        Vectors: shape (num_chains, num_samples, size).
    percentiles : list[int], optional
        Percentiles to compute. Default: [5, 50, 95].

    Returns
    -------
    dict[str, dict[str, float]]
        Summary with keys: mean, std, percentiles, ess, r_hat per parameter.
        Vector parameters are expanded to name[0], name[1], etc.
        r_hat is NaN for single chain.
    """
    if percentiles is None:
        percentiles = [5, 50, 95]

    result: dict[str, dict[str, float]] = {}

    for name, arr in samples.items():
        # Handle different dimensions
        if arr.ndim == 1:
            # 1D: single chain scalar -> (1, num_samples)
            arr = arr.reshape(1, -1)
            result[name] = _summarize_2d(name, arr, percentiles)
            _warn_if_non_converged(name, result[name])

        elif arr.ndim == 2:
            # 2D: multi-chain scalar -> (num_chains, num_samples)
            result[name] = _summarize_2d(name, arr, percentiles)
            _warn_if_non_converged(name, result[name])

        elif arr.ndim == 3:
            # 3D: vector parameter -> (num_chains, num_samples, size)
            size: int = arr.shape[2]
            for i in range(size):
                elem_name: str = f"{name}[{i}]"
                elem_arr: NDArray[np.float64] = arr[:, :, i]
                result[elem_name] = _summarize_2d(elem_name, elem_arr, percentiles)
                _warn_if_non_converged(elem_name, result[elem_name])

        else:
            raise ValueError(f"Unsupported array dimension {arr.ndim} for {name}")

    return result
