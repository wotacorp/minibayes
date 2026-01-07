"""Inference results container."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class InferenceResult:
    """
    Container for MCMC results.

    Attributes
    ----------
    samples : dict[str, ndarray]
        Samples for each parameter in CONSTRAINED space.
        Shape: (num_chains, num_samples) or (num_samples,) if num_chains=1.
    samples_unconstrained : dict[str, ndarray]
        Samples in unconstrained space (what sampler actually produced).
    acceptance_rate : float or ndarray
        Acceptance rate(s) per chain.
    num_samples : int
        Number of samples per chain.
    num_warmup : int
        Number of warmup samples (discarded).
    num_chains : int
        Number of independent chains.
    sampler : str
        Name of sampler used.
    elapsed_time : float
        Total sampling time in seconds.
    """

    samples: dict[str, NDArray[np.float64]]
    samples_unconstrained: dict[str, NDArray[np.float64]]
    acceptance_rate: float | NDArray[np.float64]
    num_samples: int
    num_warmup: int
    num_chains: int
    sampler: str
    elapsed_time: float

    def summary(
        self,
        percentiles: list[int] | None = None,
        params: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compute summary statistics.

        Parameters
        ----------
        percentiles : list[int], optional
            Percentiles to compute. Default: [5, 50, 95].
        params : list[str], optional
            Parameters to include. Default: all.

        Returns
        -------
        dict
            Summary with keys: mean, std, percentiles, ess, r_hat.
        """
        raise NotImplementedError()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to plain dict (for JSON serialization).

        Returns
        -------
        dict
            Dictionary representation of results.
        """
        raise NotImplementedError()

    def save(self, path: str, format: str = "npz") -> None:
        """
        Save results to file.

        Parameters
        ----------
        path : str
            Output file path.
        format : str
            Format: "npz" (NumPy compressed) or "json".
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str) -> "InferenceResult":
        """
        Load results from file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        InferenceResult
            Loaded results.
        """
        raise NotImplementedError()
