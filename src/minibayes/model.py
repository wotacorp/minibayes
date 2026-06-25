# Copyright 2026 WOTA CORP.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model class for structured Bayesian inference."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from minibayes.exceptions import ModelSpecError
from minibayes.params import ParamContext, ParamInfo, ParamMode
from minibayes.transforms import IdentityTransform, Transform

# Type alias for structured parameter values
StructuredParams = dict[str, float | NDArray[np.float64]]
FlatParams = dict[str, float]


class Model:
    """
    A Bayesian model with explicit priors and log-likelihood.

    Supports hierarchical models through the p(...) API where dependencies
    are defined by execution order.

    Parameters
    ----------
    priors : Callable[[ParamContext], None]
        Function that registers parameters using p(name, dist, size=None).
    log_likelihood : Callable[[dict, object], float]
        Function (params, data) -> log_likelihood value.

    Examples
    --------
    >>> from minibayes import Model, dist
    >>>
    >>> def priors(p):
    ...     mu = p("mu", dist.Normal(0, 5))
    ...     tau = p("tau", dist.HalfCauchy(5))
    ...     theta = p("theta", dist.Normal(mu, tau), size=8)
    >>>
    >>> def log_likelihood(params, data):
    ...     return dist.Normal(params["theta"], data["sigma"]).log_prob(data["y"])
    >>>
    >>> model = Model(priors=priors, log_likelihood=log_likelihood)
    """

    def __init__(
        self,
        priors: Callable[[ParamContext], None],
        log_likelihood: Callable[[StructuredParams, object], NDArray[np.float64]],
    ) -> None:
        self._priors_fn = priors
        self._log_likelihood_fn = log_likelihood

        # Run priors once in sample mode to discover structure
        ctx = ParamContext(mode=ParamMode.SAMPLE)
        self._priors_fn(ctx)

        if not ctx.param_info:
            raise ModelSpecError("priors function must register at least one parameter")

        self._param_info: dict[str, ParamInfo] = ctx.param_info
        self._param_order: list[str] = ctx.param_order
        self._param_names: list[str] = list(ctx.param_info.keys())

        # Build transforms from distribution support
        self._transforms: dict[str, Transform] = {}
        for name, info in self._param_info.items():
            self._transforms[name] = info.distribution.default_transform()

        # Compute unconstrained sizes (may differ for matrix params with transforms)
        self._unconstrained_sizes: dict[str, int] = {}
        for name, info in self._param_info.items():
            if info.shape is not None:
                # Matrix param: unconstrained size = length of transform output
                transform = self._transforms[name]
                sample_val: NDArray[np.float64] = np.asarray(ctx.values[name], dtype=np.float64)
                unc: NDArray[np.float64] = transform.forward(sample_val)
                self._unconstrained_sizes[name] = int(unc.size)
            else:
                # Scalar/vector: unconstrained size = constrained size
                self._unconstrained_sizes[name] = info.size

        # Precompute params that need Jacobian correction (skip IdentityTransform)
        self._params_with_jacobian: list[str] = [
            name for name in self._param_order if not isinstance(self._transforms[name], IdentityTransform)
        ]

        # Build flat parameter names for sampler interface
        self._flat_names: list[str] = self._build_flat_names()

    # -------------------------------------------------------------------------
    # Explicit methods - no context-dependent magic
    # -------------------------------------------------------------------------

    def sample_prior(self, rng: np.random.Generator | None = None) -> StructuredParams:
        """
        Draw one sample from the joint prior.

        Parameters
        ----------
        rng : Generator, optional
            NumPy random generator.

        Returns
        -------
        dict[str, float | NDArray]
            Parameter values in constrained space.
            Scalar parameters are floats, vector parameters are 1D arrays.
        """
        ctx = ParamContext(mode=ParamMode.SAMPLE, rng=rng)
        self._priors_fn(ctx)
        return ctx.values

    def prior_means(self) -> dict[str, float | NDArray[np.float64]]:
        """
        Return the mean of each prior distribution.

        For hierarchical models, this samples parents first to get
        means for child parameters.

        Returns
        -------
        dict[str, float | NDArray]
            Mean value for each parameter.
        """
        result: dict[str, float | NDArray[np.float64]] = {}
        for name, info in self._param_info.items():
            if info.shape is not None:
                # Matrix parameter: use identity matrix as default (valid for correlation Cholesky)
                d = info.shape[0]
                result[name] = np.eye(d, dtype=np.float64)
            elif info.is_vector:
                mean_val: float = info.distribution.mean
                result[name] = np.full(info.size, mean_val, dtype=np.float64)
            else:
                result[name] = info.distribution.mean
        return result

    def log_prior(self, params: StructuredParams) -> float:
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
        ctx = ParamContext(mode=ParamMode.EVALUATE, values=params)
        self._priors_fn(ctx)
        return ctx.log_prob

    def log_likelihood(self, params: StructuredParams, data: object) -> NDArray[np.float64]:
        """
        Compute pointwise log likelihood.

        Parameters
        ----------
        params : dict
            Parameter values in constrained space.
        data : object
            Observed data.

        Returns
        -------
        NDArray[np.float64]
            Pointwise log likelihood values, shape (n_obs,).
        """
        return self._log_likelihood_fn(params, data)

    def log_prob(self, params: StructuredParams, data: object) -> float:
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
        ll_pointwise: NDArray[np.float64] = self.log_likelihood(params, data)
        ll: float = float(np.sum(ll_pointwise))
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

    def to_unconstrained(self, params: StructuredParams) -> StructuredParams:
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
        result: StructuredParams = {}
        for name in self._param_order:
            info = self._param_info[name]
            transform = self._transforms[name]
            value = params[name]

            if info.is_vector:
                arr: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
                result[name] = transform.forward(arr)
            else:
                scalar: NDArray[np.float64] = np.atleast_1d(np.float64(value))
                transformed: NDArray[np.float64] = transform.forward(scalar)
                result[name] = float(transformed.flat[0])
        return result

    def to_constrained(self, unconstrained: StructuredParams) -> StructuredParams:
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
        result: StructuredParams = {}
        for name in self._param_order:
            info = self._param_info[name]
            transform = self._transforms[name]
            value = unconstrained[name]

            if info.is_vector:
                arr: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
                result[name] = transform.inverse(arr)
            else:
                scalar: NDArray[np.float64] = np.atleast_1d(np.float64(value))
                constrained: NDArray[np.float64] = transform.inverse(scalar)
                result[name] = float(constrained.flat[0])
        return result

    # -------------------------------------------------------------------------
    # Flat/structured conversion for sampler interface
    # -------------------------------------------------------------------------

    def _build_flat_names(self) -> list[str]:
        """Build list of flattened parameter names."""
        flat_names: list[str] = []
        for name in self._param_order:
            info = self._param_info[name]
            if info.is_vector:
                # Use unconstrained size (may differ for matrix params)
                unc_size = self._unconstrained_sizes[name]
                flat_names.extend(f"{name}[{i}]" for i in range(unc_size))
            else:
                flat_names.append(name)
        return flat_names

    def to_flat_unconstrained(self, params: StructuredParams) -> FlatParams:
        """
        Convert structured constrained params to flat unconstrained.

        This is used to prepare parameters for the sampler, which operates
        on a flat dict[str, float].

        Parameters
        ----------
        params : dict
            Structured params: scalars as float, vectors as 1D arrays, matrices as 2D.

        Returns
        -------
        dict[str, float]
            Flattened params with keys like "theta[0]", "theta[1]", etc.
        """
        result: FlatParams = {}
        for name in self._param_order:
            info = self._param_info[name]
            transform = self._transforms[name]
            value = params[name]

            if info.is_vector:
                arr: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
                unc: NDArray[np.float64] = transform.forward(arr)
                # Use unconstrained size (may differ for matrix params)
                unc_size = self._unconstrained_sizes[name]
                for i in range(unc_size):
                    val: float = float(unc.flat[i])
                    result[f"{name}[{i}]"] = val
            else:
                scalar: NDArray[np.float64] = np.atleast_1d(np.float64(value))
                unc = transform.forward(scalar)
                result[name] = float(unc.flat[0])
        return result

    def from_flat_unconstrained(self, flat: FlatParams) -> StructuredParams:
        """
        Convert flat unconstrained params to structured constrained.

        This is used to convert sampler output back to structured form.

        Parameters
        ----------
        flat : dict[str, float]
            Flattened unconstrained params.

        Returns
        -------
        dict
            Structured params: scalars as float, vectors as 1D arrays, matrices as 2D.
        """
        result: StructuredParams = {}
        for name in self._param_order:
            info = self._param_info[name]
            transform = self._transforms[name]

            if info.is_vector:
                # Use unconstrained size (may differ for matrix params)
                unc_size = self._unconstrained_sizes[name]
                unc_list: list[float] = [flat[f"{name}[{i}]"] for i in range(unc_size)]
                unc_arr: NDArray[np.float64] = np.array(unc_list, dtype=np.float64)
                result[name] = transform.inverse(unc_arr)
            else:
                scalar: NDArray[np.float64] = np.atleast_1d(np.float64(flat[name]))
                result[name] = float(transform.inverse(scalar).flat[0])
        return result

    def log_prob_unconstrained(
        self,
        unconstrained: FlatParams,
        data: object,
    ) -> float:
        """
        Compute log_prob in unconstrained space.

        This is what the sampler actually calls. Includes Jacobian
        correction for the change of variables.

        Parameters
        ----------
        unconstrained : dict[str, float]
            Flat parameter values in unconstrained space.
        data : object
            Observed data.

        Returns
        -------
        float
            Log posterior with Jacobian correction.
        """
        # Convert flat unconstrained to structured constrained
        constrained: StructuredParams = self.from_flat_unconstrained(unconstrained)

        # Compute log_prob in constrained space
        lp: float = self.log_prob(constrained, data)

        # Add Jacobian correction (skip IdentityTransform params - they contribute 0)
        jacobian_correction: float = 0.0
        for name in self._params_with_jacobian:
            info = self._param_info[name]
            transform = self._transforms[name]
            value = constrained[name]

            if info.is_vector:
                arr: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
                log_det: NDArray[np.float64] = transform.log_det_jacobian(arr)
                jacobian_correction += float(np.sum(log_det))
            else:
                scalar: NDArray[np.float64] = np.atleast_1d(np.float64(value))
                log_det = transform.log_det_jacobian(scalar)
                jacobian_correction += float(log_det.flat[0])

        return lp + jacobian_correction

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    @property
    def param_names(self) -> list[str]:
        """
        List of parameter names (structured).

        Returns
        -------
        list[str]
            Parameter names.
        """
        return list(self._param_names)

    @property
    def flat_param_names(self) -> list[str]:
        """
        List of flattened parameter names.

        Vector parameters are expanded: ["theta[0]", "theta[1]", ...].

        Returns
        -------
        list[str]
            Flattened parameter names.
        """
        return list(self._flat_names)

    @property
    def param_info(self) -> dict[str, ParamInfo]:
        """
        Get parameter metadata.

        Returns
        -------
        dict[str, ParamInfo]
            Metadata for each parameter.
        """
        return dict(self._param_info)

    def validate_params(self, params: StructuredParams) -> bool:
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
        for name, info in self._param_info.items():
            value = params[name]
            lp: NDArray[np.float64] | float = info.distribution.log_prob(value)
            # Type narrowing for mypy
            if not isinstance(lp, float):
                lp_arr: NDArray[np.float64] = lp
                if not bool(np.all(np.isfinite(lp_arr))):
                    raise ModelSpecError(f"Parameter '{name}' has values outside support")
            else:
                if not np.isfinite(lp):
                    raise ModelSpecError(f"Parameter '{name}' value {value} is outside support")

        return True
