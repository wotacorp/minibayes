"""Convergence diagnostics for MCMC."""

import numpy as np
from numpy.typing import NDArray


def effective_sample_size(samples: NDArray[np.float64]) -> float:
    """
    Compute effective sample size accounting for autocorrelation.

    Uses FFT-based autocorrelation estimation.

    Parameters
    ----------
    samples : ndarray
        1D array of MCMC samples.

    Returns
    -------
    float
        Effective sample size. Always <= len(samples).
    """
    raise NotImplementedError()


def r_hat(chains: NDArray[np.float64]) -> float:
    """
    Compute Gelman-Rubin R-hat diagnostic.

    Values > 1.01 indicate non-convergence.

    Parameters
    ----------
    chains : ndarray
        Array of shape (num_chains, num_samples).

    Returns
    -------
    float
        R-hat statistic. Should be close to 1.0 for converged chains.
    """
    raise NotImplementedError()


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
    percentiles : list[int], optional
        Percentiles to compute. Default: [5, 50, 95].

    Returns
    -------
    dict
        Summary with keys: mean, std, percentiles, ess per parameter.
    """
    raise NotImplementedError()
