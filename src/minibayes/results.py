"""Inference results container."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike


@dataclass
class InferenceResult:
    """
    Container for MCMC results.

    Attributes
    ----------
    samples : dict[str, ndarray]
        Samples for each parameter in CONSTRAINED space.
        Shape: (num_chains, num_samples).
    samples_unconstrained : dict[str, ndarray]
        Samples in unconstrained space (what sampler actually produced).
        Shape: (num_chains, num_samples).
    acceptance_rate : ndarray
        Acceptance rate per chain. Shape: (num_chains,).
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
    derived : dict[str, ndarray]
        User-added derived parameters (e.g., correlations from Cholesky factors).
        Shape: (num_chains, num_samples) or (num_chains, num_samples, ...).
    """

    samples: dict[str, NDArray[np.float64]]
    samples_unconstrained: dict[str, NDArray[np.float64]]
    acceptance_rate: NDArray[np.float64]
    num_samples: int
    num_warmup: int
    num_chains: int
    sampler: str
    elapsed_time: float
    derived: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def summary(
        self,
        percentiles: list[int] | None = None,
        params: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
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
        dict[str, dict[str, float]]
            Summary with keys: mean, std, percentiles, ess, r_hat.
        """
        from minibayes.diagnostics import summary as compute_summary

        # Merge samples and derived
        all_samples: dict[str, NDArray[np.float64]] = {**self.samples, **self.derived}

        if params is not None:
            filtered: dict[str, NDArray[np.float64]] = {
                k: v for k, v in all_samples.items() if k in params
            }
        else:
            filtered = all_samples

        return compute_summary(filtered, percentiles)

    def add_derived(self, name: str, samples: NDArray[np.float64]) -> None:
        """
        Add a derived parameter computed from existing samples.

        Parameters
        ----------
        name : str
            Name for the derived parameter.
        samples : ndarray
            Samples array with shape (num_chains, num_samples) or
            (num_chains, num_samples, ...) for vector/matrix quantities.

        Raises
        ------
        ValueError
            If name conflicts with existing sample or shape is invalid.

        Examples
        --------
        >>> # Extract correlation from Cholesky factor
        >>> L_samples = result.samples["L_corr"]
        >>> rho_samples = L_samples[:, :, 1, 0]
        >>> result.add_derived("rho", rho_samples)
        >>> viz.plot_density(result, params=["rho"])
        """
        if name in self.samples:
            raise ValueError(f"'{name}' already exists in samples")
        if name in self.derived:
            raise ValueError(f"'{name}' already exists in derived")
        expected_prefix: tuple[int, int] = (self.num_chains, self.num_samples)
        if samples.shape[:2] != expected_prefix:
            raise ValueError(
                f"Shape must be ({self.num_chains}, {self.num_samples}, ...), "
                f"got {samples.shape}"
            )
        self.derived[name] = samples

    def predict(
        self,
        predictive_fn: Callable[
            [dict[str, float | NDArray[np.float64]], np.random.Generator],
            dict[str, ArrayLike],
        ],
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """
        Generate posterior predictive samples (convenience wrapper).

        Parameters
        ----------
        predictive_fn : Callable[[StructuredParams, Generator], dict[str, ArrayLike]]
            Function (params, rng) -> predictions. params contains scalars as float,
            vectors as 1D arrays.
        num_samples : int, optional
            Number of posterior samples to use.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            Predictions with shape (num_samples, *prediction_shape).

        See Also
        --------
        sample_posterior_predictive : Full documentation.
        """
        from minibayes.predictive import sample_posterior_predictive

        return sample_posterior_predictive(self, predictive_fn, num_samples, seed)

    def to_dict(self) -> dict[str, object]:
        """
        Convert to plain dict (for JSON serialization).

        Returns
        -------
        dict
            Dictionary representation of results.
        """
        from minibayes.utils.export import to_json

        return to_json(self)

    def save(self, path: str, format: str = "npz") -> None:
        """
        Save results to file.

        Parameters
        ----------
        path : str
            Output file path.
        format : str
            Format: "npz" (NumPy compressed) or "json".

        Raises
        ------
        ValueError
            If format is not "npz" or "json".
        """
        from minibayes.utils.export import save_npz, to_json

        if format == "npz":
            save_npz(self, path)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(to_json(self), f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'npz' or 'json'.")

    @classmethod
    def load(cls, path: str) -> InferenceResult:
        """
        Load results from file.

        Parameters
        ----------
        path : str
            Input file path. Format detected from extension (.npz or .json).

        Returns
        -------
        InferenceResult
            Loaded results.
        """
        from minibayes.utils.export import load_npz

        if path.endswith(".npz"):
            return load_npz(path)
        elif path.endswith(".json"):
            return cls._load_json(path)
        else:
            # Try npz first, fall back to json
            try:
                return load_npz(path)
            except Exception:
                return cls._load_json(path)

    @classmethod
    def _load_json(cls, path: str) -> InferenceResult:
        """Load from JSON file."""
        with open(path) as f:
            raw_data: object = json.load(f)

        # We know json.load returns a dict for our format
        if not isinstance(raw_data, dict):
            raise ValueError("JSON must contain a dictionary")

        data: dict[str, object] = raw_data

        # Extract samples
        samples_raw: object = data.get("samples", {})
        if not isinstance(samples_raw, dict):
            raise ValueError("samples must be a dictionary")

        samples: dict[str, NDArray[np.float64]] = {}
        for k, v in samples_raw.items():
            if isinstance(k, str):
                arr: NDArray[np.float64] = np.asarray(v, dtype=np.float64)
                samples[k] = arr

        # Extract unconstrained samples
        samples_unc_raw: object = data.get("samples_unconstrained", {})
        if not isinstance(samples_unc_raw, dict):
            raise ValueError("samples_unconstrained must be a dictionary")

        samples_unconstrained: dict[str, NDArray[np.float64]] = {}
        for k, v in samples_unc_raw.items():
            if isinstance(k, str):
                arr = np.asarray(v, dtype=np.float64)
                samples_unconstrained[k] = arr

        # Extract derived samples (optional, may not exist in older files)
        derived_raw: object = data.get("derived", {})
        derived: dict[str, NDArray[np.float64]] = {}
        if isinstance(derived_raw, dict):
            for k, v in derived_raw.items():
                if isinstance(k, str):
                    arr = np.asarray(v, dtype=np.float64)
                    derived[k] = arr

        # Extract acceptance rate as array
        acc_rate_raw: object = data.get("acceptance_rate", [0.0])
        acceptance_rate: NDArray[np.float64] = np.atleast_1d(np.asarray(acc_rate_raw, dtype=np.float64))

        # Extract scalar values
        num_samples_raw: object = data.get("num_samples", 0)
        num_samples: int = int(num_samples_raw) if isinstance(num_samples_raw, (int, float)) else 0

        num_warmup_raw: object = data.get("num_warmup", 0)
        num_warmup: int = int(num_warmup_raw) if isinstance(num_warmup_raw, (int, float)) else 0

        num_chains_raw: object = data.get("num_chains", 1)
        num_chains: int = int(num_chains_raw) if isinstance(num_chains_raw, (int, float)) else 1

        sampler_raw: object = data.get("sampler", "")
        sampler: str = str(sampler_raw) if isinstance(sampler_raw, str) else ""

        elapsed_time_raw: object = data.get("elapsed_time", 0.0)
        elapsed_time: float = float(elapsed_time_raw) if isinstance(elapsed_time_raw, (int, float)) else 0.0

        return cls(
            samples=samples,
            samples_unconstrained=samples_unconstrained,
            acceptance_rate=acceptance_rate,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            sampler=sampler,
            elapsed_time=elapsed_time,
            derived=derived,
        )
