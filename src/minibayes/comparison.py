"""Model comparison metrics for Bayesian inference.

This module provides WAIC (Widely Applicable Information Criterion)
for comparing Bayesian models.

References
----------
Watanabe, S. (2010). Asymptotic Equivalence of Bayes Cross Validation and
    Widely Applicable Information Criterion in Singular Learning Theory.
    Journal of Machine Learning Research, 11, 3571-3594.

Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., & Rubin, D.B.
    (2014). Bayesian Data Analysis, 3rd ed. CRC Press.

Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation
    using leave-one-out cross-validation and WAIC. Statistics and Computing,
    27(5), 1413-1432.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from minibayes.utils.numerical import log_sum_exp

if TYPE_CHECKING:
    from minibayes.model import Model
    from minibayes.results import InferenceResult


@dataclass
class WAICResult:
    """
    Result of WAIC computation.

    Attributes
    ----------
    waic : float
        WAIC value (lower is better). Computed as -2 * (lppd - p_waic).
    p_waic : float
        Effective number of parameters (pWAIC2, variance-based).
    lppd : float
        Log pointwise predictive density.
    se : float
        Standard error of WAIC estimate.
    pointwise : NDArray[np.float64]
        Per-observation WAIC contributions, shape (n_obs,).
        Can be used to identify influential observations.
    """

    waic: float
    p_waic: float
    lppd: float
    se: float
    pointwise: NDArray[np.float64]

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"WAICResult(waic={self.waic:.2f}, p_waic={self.p_waic:.2f}, "
            f"lppd={self.lppd:.2f}, se={self.se:.2f})"
        )


def waic(
    result: InferenceResult,
    model: Model,
    data: object,
) -> WAICResult:
    """
    Compute WAIC (Widely Applicable Information Criterion).

    WAIC is a fully Bayesian approach for estimating out-of-sample
    prediction error using the computed log pointwise predictive density
    and correcting for overfitting with the effective number of parameters.

    Parameters
    ----------
    result : InferenceResult
        Posterior samples from MCMC.
    model : Model
        Model used for sampling. Must have log_likelihood that returns
        pointwise log-likelihood array of shape (n_obs,).
    data : object
        Observed data passed to log_likelihood.

    Returns
    -------
    WAICResult
        WAIC, p_waic, lppd, standard error, and pointwise values.

    Notes
    -----
    WAIC formula (Gelman et al. 2014, p. 174):

        WAIC = -2 * (lppd - p_waic)

    where:
        lppd = sum_i log(mean_s p(y_i | theta_s))
        p_waic = sum_i var_s(log p(y_i | theta_s))

    Lower WAIC indicates better predictive accuracy.

    Examples
    --------
    >>> result = mb.sample(model, data, num_samples=2000)
    >>> waic_result = mb.waic(result, model, data)
    >>> print(f"WAIC: {waic_result.waic:.1f}")
    """
    # Flatten samples across chains: shape (n_chains * n_samples,) per param
    n_chains = result.num_chains
    n_samples = result.num_samples
    total_samples = n_chains * n_samples

    # Compute log-likelihood for each posterior sample
    # First sample to determine n_obs
    first_params = _extract_params(result.samples, 0, 0)
    first_ll: NDArray[np.float64] = model.log_likelihood(first_params, data)
    n_obs: int = int(first_ll.shape[0])

    # Allocate matrix: (total_samples, n_obs)
    log_lik: NDArray[np.float64] = np.zeros((total_samples, n_obs), dtype=np.float64)

    # Fill in log-likelihoods
    idx = 0
    for chain_idx in range(n_chains):
        for sample_idx in range(n_samples):
            params = _extract_params(result.samples, chain_idx, sample_idx)
            ll: NDArray[np.float64] = model.log_likelihood(params, data)
            log_lik[idx, :] = ll
            idx += 1

    # Compute WAIC components
    # lppd_i = log(mean_s exp(log_lik[s, i])) = log_sum_exp(log_lik[:, i]) - log(S)
    log_s: float = float(np.log(total_samples))
    lppd_list: list[float] = [
        log_sum_exp(log_lik[:, i]) - log_s for i in range(n_obs)
    ]
    lppd_i: NDArray[np.float64] = np.array(lppd_list, dtype=np.float64)
    lppd: float = float(np.sum(lppd_i))

    # p_waic = sum_i var_s(log_lik[s, i])  (pWAIC2, variance-based)
    var_result: NDArray[np.float64] = cast(
        "NDArray[np.float64]", np.var(log_lik, axis=0, ddof=1)
    )
    p_waic_i: NDArray[np.float64] = np.asarray(var_result, dtype=np.float64)
    p_waic: float = float(np.sum(p_waic_i))

    # WAIC = -2 * (lppd - p_waic)
    diff: NDArray[np.float64] = lppd_i - p_waic_i
    pointwise_waic: NDArray[np.float64] = -2.0 * diff
    waic_value: float = float(np.sum(pointwise_waic))

    # Standard error: se = sqrt(n * var(pointwise_waic))
    # Handle edge case of n=1 observation where variance is undefined
    se: float
    if n_obs > 1:
        # Use cast to handle numpy scalar return types
        var_pw: float = cast("float", np.var(pointwise_waic, ddof=1))
        se = cast("float", np.sqrt(n_obs * var_pw))
    else:
        se = 0.0

    return WAICResult(
        waic=waic_value,
        p_waic=p_waic,
        lppd=lppd,
        se=se,
        pointwise=pointwise_waic,
    )


def _extract_params(
    samples: dict[str, NDArray[np.float64]],
    chain_idx: int,
    sample_idx: int,
) -> dict[str, float | NDArray[np.float64]]:
    """
    Extract parameters for a single sample.

    Parameters
    ----------
    samples : dict[str, NDArray]
        Samples dictionary with shape (n_chains, n_samples, ...).
    chain_idx : int
        Chain index.
    sample_idx : int
        Sample index within chain.

    Returns
    -------
    dict[str, float | NDArray]
        Parameters for this sample. Scalars as float, vectors/matrices as arrays.
    """
    params: dict[str, float | NDArray[np.float64]] = {}
    for name, arr in samples.items():
        if arr.ndim == 2:
            # Scalar parameter: (n_chains, n_samples)
            params[name] = float(arr[chain_idx, sample_idx])  # type: ignore[misc]
        else:
            # Vector/matrix parameter: (n_chains, n_samples, ...)
            sliced: NDArray[np.float64] = arr[chain_idx, sample_idx]
            params[name] = np.asarray(sliced, dtype=np.float64)
    return params
