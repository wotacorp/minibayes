"""Model class for structured Bayesian inference."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution
from minibayes.exceptions import ModelSpecError
from minibayes.transforms import Transform
from minibayes.utils import ensure_rng


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
        if not priors:
            raise ModelSpecError("priors must be non-empty")

        self._priors = priors
        self._likelihood = likelihood
        self._param_names = list(priors.keys())

        # Build transforms from distribution support
        self._transforms: dict[str, Transform] = {}
        for name, prior in priors.items():
            self._transforms[name] = prior.default_transform()

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
        generator: np.random.Generator = ensure_rng(rng)
        result: dict[str, float] = {}
        for name, prior in self._priors.items():
            sample_val = prior.sample(size=None, rng=generator)
            result[name] = float(sample_val)
        return result

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
        total: float = 0.0
        for name, prior in self._priors.items():
            lp = prior.log_prob(params[name])
            total += float(lp)
        return total

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
        return self._likelihood(params, data)

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
        lp: float = self.log_prior(params)
        ll: float = self.log_likelihood(params, data)
        return lp + ll

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
        return self._transforms

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
        result: dict[str, float] = {}
        for name in self._param_names:
            transform = self._transforms[name]
            scalar: NDArray[np.float64] = np.atleast_1d(np.float64(params[name]))
            transformed: NDArray[np.float64] = transform.forward(scalar)
            result[name] = float(transformed.flat[0])
        return result

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
        result: dict[str, float] = {}
        for name in self._param_names:
            transform = self._transforms[name]
            scalar: NDArray[np.float64] = np.atleast_1d(np.float64(unconstrained[name]))
            constrained: NDArray[np.float64] = transform.inverse(scalar)
            result[name] = float(constrained.flat[0])
        return result

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
        # Transform to constrained space
        constrained: dict[str, float] = self.to_constrained(unconstrained)

        # Compute log_prob in constrained space
        lp: float = self.log_prob(constrained, data)

        # Add Jacobian correction: sum of log|d(theta)/d(phi)|
        jacobian_correction: float = 0.0
        for name in self._param_names:
            transform = self._transforms[name]
            theta: float = constrained[name]
            scalar: NDArray[np.float64] = np.atleast_1d(np.float64(theta))
            log_det: NDArray[np.float64] = transform.log_det_jacobian(scalar)
            jacobian_correction += float(log_det.flat[0])

        return lp + jacobian_correction

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
        return self._param_names

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
        ModelSpecError
            If params are invalid, with details.
        """
        # Check parameter names
        param_keys = set(params.keys())
        expected_keys = set(self._param_names)

        missing = expected_keys - param_keys
        if missing:
            raise ModelSpecError(f"Missing parameters: {missing}")

        extra = param_keys - expected_keys
        if extra:
            raise ModelSpecError(f"Unknown parameters: {extra}")

        # Check values are within support
        for name, prior in self._priors.items():
            value = params[name]
            lp = prior.log_prob(value)
            if not np.isfinite(float(lp)):
                raise ModelSpecError(f"Parameter '{name}' value {value} is outside support")

        return True
