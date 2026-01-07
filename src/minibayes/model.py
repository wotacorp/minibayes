"""Model class for structured Bayesian inference."""

from typing import Callable

import numpy as np

from minibayes.distributions.base import Distribution
from minibayes.transforms import Transform


class Model:
    """
    A Bayesian model with explicit priors and likelihood.

    Priors are assumed independent (no hierarchical structure).
    Transforms are derived automatically from distribution support.

    Parameters
    ----------
    priors : dict[str, Distribution]
        Prior distributions for each parameter.
    likelihood : Callable[[dict, object], float]
        Function (params, data) -> log_likelihood.
    """

    def __init__(
        self,
        priors: dict[str, Distribution],
        likelihood: Callable[[dict[str, float], object], float],
    ) -> None:
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    # Explicit methods - no context-dependent magic
    # -------------------------------------------------------------------------

    def sample_prior(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """
        Draw one sample from the joint prior.

        Parameters
        ----------
        rng : Generator, optional
            NumPy random generator.

        Returns
        -------
        dict[str, float]
            Parameter values in constrained space.
        """
        raise NotImplementedError()

    def log_prior(self, params: dict[str, float]) -> float:
        """
        Compute log prior probability.

        Parameters
        ----------
        params : dict
            Parameter values in constrained space.

        Returns
        -------
        float
            Sum of log_prob for each prior.
        """
        raise NotImplementedError()

    def log_likelihood(self, params: dict[str, float], data: object) -> float:
        """
        Compute log likelihood.

        Parameters
        ----------
        params : dict
            Parameter values in constrained space.
        data : object
            Observed data.

        Returns
        -------
        float
            Log likelihood value.
        """
        raise NotImplementedError()

    def log_prob(self, params: dict[str, float], data: object) -> float:
        """
        Compute unnormalized log posterior = log_prior + log_likelihood.

        Parameters
        ----------
        params : dict
            Parameter values in constrained space.
        data : object
            Observed data.

        Returns
        -------
        float
            Log posterior (unnormalized).
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    # Transform handling - automatic but inspectable
    # -------------------------------------------------------------------------

    @property
    def transforms(self) -> dict[str, Transform]:
        """
        Get transforms for each parameter.

        Transforms are derived from distribution support:
        - REAL -> IdentityTransform
        - POSITIVE -> LogTransform
        - UNIT -> LogitTransform
        - BOUNDED -> AffineTransform

        Returns
        -------
        dict[str, Transform]
            Transform for each parameter.
        """
        raise NotImplementedError()

    def to_unconstrained(self, params: dict[str, float]) -> dict[str, float]:
        """
        Transform constrained params to unconstrained space.

        Example: {"sigma": 2.0} -> {"sigma": 0.693}  (log transform)

        Parameters
        ----------
        params : dict
            Parameter values in constrained space.

        Returns
        -------
        dict
            Parameter values in unconstrained space.
        """
        raise NotImplementedError()

    def to_constrained(self, unconstrained: dict[str, float]) -> dict[str, float]:
        """
        Transform unconstrained params back to constrained space.

        Example: {"sigma": 0.693} -> {"sigma": 2.0}  (exp transform)

        Parameters
        ----------
        unconstrained : dict
            Parameter values in unconstrained space.

        Returns
        -------
        dict
            Parameter values in constrained space.
        """
        raise NotImplementedError()

    def log_prob_unconstrained(
        self,
        unconstrained: dict[str, float],
        data: object,
    ) -> float:
        """
        Compute log_prob in unconstrained space.

        This is what the sampler actually calls. Includes Jacobian
        correction for the change of variables.

        Parameters
        ----------
        unconstrained : dict
            Parameter values in unconstrained space.
        data : object
            Observed data.

        Returns
        -------
        float
            Log posterior with Jacobian correction.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    @property
    def param_names(self) -> list[str]:
        """
        List of parameter names.

        Returns
        -------
        list[str]
            Parameter names.
        """
        raise NotImplementedError()

    def validate_params(self, params: dict[str, float]) -> bool:
        """
        Check if params are valid (correct names, within support).

        Parameters
        ----------
        params : dict
            Parameter values to validate.

        Returns
        -------
        bool
            True if valid.

        Raises
        ------
        MinibayesError
            If params are invalid, with details.
        """
        raise NotImplementedError()
