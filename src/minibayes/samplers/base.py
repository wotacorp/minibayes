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

"""Base class for MCMC samplers."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class Sampler(ABC):
    """
    Abstract base class for MCMC samplers.

    Subclasses must implement:
    - `advance()`: Take one step for all chains

    Optional hooks:
    - `post_warmup()`: Called after warmup completes (for cleanup/finalization)
    """

    def __init__(self) -> None:
        self._states: list[dict[str, float]] = []
        self._log_prob_fn: Callable[[dict[str, float]], float] | None = None
        # Cached log_probs for each chain (avoids recomputing on rejected proposals)
        self._log_probs: list[float] = []
        # Positions array for efficient extraction (num_chains, num_params)
        self._positions: NDArray[np.float64] | None = None
        self._param_names: list[str] | None = None

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
        # Initialize log_prob cache
        self._log_probs = [log_prob_fn(s) for s in self._states]
        # Initialize positions array for efficient extraction
        if initial_states:
            self._param_names = list(initial_states[0].keys())
            n_chains: int = len(initial_states)
            n_params: int = len(self._param_names)
            self._positions = np.empty((n_chains, n_params), dtype=np.float64)
            for i, state in enumerate(initial_states):
                for j, name in enumerate(self._param_names):
                    self._positions[i, j] = state[name]

    @abstractmethod
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
            If True, perform warmup step (may adapt). Default: False.
        step_num : int
            Current step number (for warmup adaptation scheduling).

        Returns
        -------
        float
            Average acceptance rate across all chains.
        """

    def post_warmup(self) -> None:  # noqa: B027
        """
        Called after warmup completes.

        Override in subclasses to perform cleanup or finalization,
        such as freezing adaptation or releasing memory.
        """

    def _update_positions(self, chain_idx: int, state: dict[str, float]) -> None:
        """Update positions array for a single chain."""
        if self._positions is not None and self._param_names is not None:
            for j, name in enumerate(self._param_names):
                self._positions[chain_idx, j] = state[name]

    def get_states(self) -> list[dict[str, float]]:
        """
        Get current states of all chains.

        Returns
        -------
        list[dict[str, float]]
            List of current states for each chain.
        """
        return [s.copy() for s in self._states]

    def get_positions(self) -> NDArray[np.float64]:
        """
        Get current positions as a numpy array.

        Returns
        -------
        NDArray[np.float64]
            Positions array with shape (num_chains, num_params).
            This is a view, not a copy.
        """
        if self._positions is None:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")
        return self._positions

    def get_param_names(self) -> list[str]:
        """
        Get parameter names in order matching get_positions() columns.

        Returns
        -------
        list[str]
            Parameter names.
        """
        if self._param_names is None:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")
        return self._param_names

    @property
    def num_chains(self) -> int:
        """Number of chains being managed by this sampler."""
        return len(self._states)
