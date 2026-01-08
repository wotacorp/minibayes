"""Base class for MCMC samplers."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class Sampler(ABC):
    """Abstract base class for MCMC samplers."""

    @abstractmethod
    def step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], bool]:
        """
        Take one MCMC step.

        Parameters
        ----------
        current : dict
            Current parameter values (unconstrained space).
        log_prob_fn : Callable
            Function params -> log_prob (in unconstrained space).
        rng : Generator
            NumPy random generator.

        Returns
        -------
        new_state : dict
            New parameter values.
        accepted : bool
            Whether proposal was accepted.
        """

    @abstractmethod
    def warmup_step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        step_num: int,
    ) -> tuple[dict[str, float], bool]:
        """
        Take one warmup step (may adapt internal state).

        Parameters
        ----------
        current : dict
            Current parameter values (unconstrained space).
        log_prob_fn : Callable
            Function params -> log_prob (in unconstrained space).
        rng : Generator
            NumPy random generator.
        step_num : int
            Current warmup step number (for adaptation scheduling).

        Returns
        -------
        new_state : dict
            New parameter values.
        accepted : bool
            Whether proposal was accepted.
        """

    def post_warmup(self) -> None:  # noqa: B027
        """
        Called after warmup completes.

        Override in subclasses to perform cleanup or finalization,
        such as freezing adaptation or releasing memory.
        """
