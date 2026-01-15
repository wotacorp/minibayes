"""Base class for MCMC samplers."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class Sampler(ABC):
    """
    Abstract base class for MCMC samplers.

    Samplers support two interfaces:

    1. **Stateless interface** (step/warmup_step): Caller manages state.
       Used for direct sampler usage and backwards compatibility.

    2. **Stateful interface** (initialize/advance/get_states): Sampler
       manages state internally. Used by inference.py for unified handling
       of single-chain and multi-chain samplers.

    Subclasses must implement the stateless interface. The stateful interface
    has default implementations that wrap the stateless methods.
    """

    def __init__(self) -> None:
        self._states: list[dict[str, float]] = []
        self._log_prob_fn: Callable[[dict[str, float]], float] | None = None

    # -------------------------------------------------------------------------
    # Stateless interface (caller manages state)
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Stateful interface (sampler manages state internally)
    # -------------------------------------------------------------------------

    def initialize(
        self,
        initial_states: list[dict[str, float]],
        log_prob_fn: Callable[[dict[str, float]], float],
    ) -> None:
        """
        Initialize sampler with one or more initial states.

        Parameters
        ----------
        initial_states : list[dict[str, float]]
            List of initial states. For single-chain samplers, this should
            have exactly one element. For ensemble samplers, K elements.
        log_prob_fn : Callable
            Function to compute log probability.
        """
        self._states = [s.copy() for s in initial_states]
        self._log_prob_fn = log_prob_fn

    def advance(
        self,
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        warmup: bool = False,
        step_num: int = 0,
    ) -> float:
        """
        Advance all chains by one step.

        Parameters
        ----------
        log_prob_fn : Callable
            Function to compute log probability.
        rng : Generator
            NumPy random generator.
        warmup : bool
            If True, use warmup_step (with adaptation). Default: False.
        step_num : int
            Current step number (for warmup adaptation scheduling).

        Returns
        -------
        float
            Average acceptance rate across all chains.
        """
        if not self._states:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")

        accepts: int = 0
        for i in range(len(self._states)):
            if warmup:
                new_state, accepted = self.warmup_step(
                    self._states[i], log_prob_fn, rng, step_num
                )
            else:
                new_state, accepted = self.step(self._states[i], log_prob_fn, rng)
            self._states[i] = new_state
            accepts += int(accepted)

        return accepts / len(self._states)

    def get_states(self) -> list[dict[str, float]]:
        """
        Get current states of all chains.

        Returns
        -------
        list[dict[str, float]]
            List of current states for each chain.
        """
        return [s.copy() for s in self._states]

    @property
    def num_chains(self) -> int:
        """Number of chains being managed by this sampler."""
        return len(self._states)
