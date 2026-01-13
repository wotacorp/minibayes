"""Parameter context for hierarchical models."""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from minibayes.distributions.base import Distribution
from minibayes.exceptions import ModelSpecError
from minibayes.utils import ensure_rng


class ParamMode(Enum):
    """Mode of operation for ParamContext."""

    SAMPLE = auto()  # Draw from distributions
    EVALUATE = auto()  # Accumulate log_prob


@dataclass(frozen=True)
class ParamInfo:
    """
    Metadata about a registered parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    distribution : Distribution
        Prior distribution for this parameter.
    is_vector : bool
        True if this is a vector parameter (size > 1).
    size : int
        Number of elements (1 for scalar, >1 for vector).
    """

    name: str
    distribution: Distribution
    is_vector: bool
    size: int


class ParamContext:
    """
    Context for registering and evaluating parameters.

    Used as the `p` argument in priors functions. Operates in two modes:

    - SAMPLE mode: Draw from distributions, store and return values
    - EVALUATE mode: Look up current values, accumulate log_prob, return values

    Parameters
    ----------
    mode : ParamMode
        Operating mode (SAMPLE or EVALUATE).
    values : dict, optional
        Pre-existing parameter values (required for EVALUATE mode).
    rng : Generator, optional
        NumPy random generator (used in SAMPLE mode).

    Examples
    --------
    >>> from minibayes import dist
    >>> from minibayes.params import ParamContext, ParamMode
    >>>
    >>> # Sampling mode
    >>> ctx = ParamContext(mode=ParamMode.SAMPLE)
    >>> def priors(p):
    ...     mu = p("mu", dist.Normal(0, 5))
    ...     sigma = p("sigma", dist.HalfNormal(1))
    ...     return mu, sigma
    >>> priors(ctx)
    >>> ctx.values  # {'mu': ..., 'sigma': ...}
    >>>
    >>> # Evaluation mode
    >>> ctx = ParamContext(mode=ParamMode.EVALUATE, values={'mu': 1.0, 'sigma': 0.5})
    >>> priors(ctx)
    >>> ctx.log_prob  # sum of log_prob for all parameters
    """

    def __init__(
        self,
        mode: ParamMode,
        values: dict[str, float | NDArray[np.float64]] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._mode = mode
        self._log_prob: float = 0.0
        self._param_info: dict[str, ParamInfo] = {}
        self._order: list[str] = []

        if mode == ParamMode.EVALUATE:
            # Fast path: don't copy values (we only read), skip RNG
            self._values: dict[str, float | NDArray[np.float64]] = (
                values if values is not None else {}
            )
            self._rng: np.random.Generator | None = None
        else:
            # SAMPLE mode: need RNG
            self._values = {}
            self._rng = ensure_rng(rng)

    def __call__(
        self,
        name: str,
        dist: Distribution,
        size: int | None = None,
    ) -> float | NDArray[np.float64]:
        """
        Register a parameter and return its value.

        Parameters
        ----------
        name : str
            Parameter name. Must be unique within the model.
        dist : Distribution
            Prior distribution for this parameter.
        size : int, optional
            If provided, create a vector parameter with `size` IID draws.

        Returns
        -------
        float or NDArray
            Current parameter value. Scalar if size is None, array otherwise.

        Raises
        ------
        ModelSpecError
            If parameter name is already registered.
        KeyError
            If in EVALUATE mode and parameter value is missing.
        """
        # EVALUATE mode: fast path - skip metadata tracking
        if self._mode == ParamMode.EVALUATE:
            value: float | NDArray[np.float64] = self._values[name]
            lp: NDArray[np.float64] | float = dist.log_prob(value)
            if not isinstance(lp, float):
                self._log_prob += float(np.sum(lp))
            else:
                self._log_prob += lp
            return value

        # SAMPLE mode: full tracking
        if name in self._param_info:
            raise ModelSpecError(f"Parameter '{name}' already registered")

        is_vector: bool = size is not None
        actual_size: int = size if size is not None else 1

        self._param_info[name] = ParamInfo(
            name=name,
            distribution=dist,
            is_vector=is_vector,
            size=actual_size,
        )
        self._order.append(name)
        return self._sample_param(name, dist, size)

    def _sample_param(
        self,
        name: str,
        dist: Distribution,
        size: int | None,
    ) -> float | NDArray[np.float64]:
        """Draw from distribution and store value."""
        if size is not None:
            # Vector parameter: sample size IID draws
            samples: list[float] = [
                float(dist.sample(rng=self._rng)) for _ in range(size)
            ]
            value: NDArray[np.float64] = np.array(samples, dtype=np.float64)
            self._values[name] = value
            return value
        else:
            # Scalar parameter
            scalar_value: float = float(dist.sample(rng=self._rng))
            self._values[name] = scalar_value
            return scalar_value

    @property
    def mode(self) -> ParamMode:
        """Return the current operating mode."""
        return self._mode

    @property
    def log_prob(self) -> float:
        """Return accumulated log probability (EVALUATE mode)."""
        return self._log_prob

    @property
    def values(self) -> dict[str, float | NDArray[np.float64]]:
        """Return copy of parameter values."""
        return dict(self._values)

    @property
    def param_info(self) -> dict[str, ParamInfo]:
        """Return copy of parameter metadata."""
        return dict(self._param_info)

    @property
    def param_order(self) -> list[str]:
        """Return parameter names in registration order."""
        return list(self._order)

    @property
    def param_names(self) -> list[str]:
        """Return parameter names (alias for param_order)."""
        return list(self._order)
