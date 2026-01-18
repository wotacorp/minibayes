"""Affine-invariant ensemble sampler (Goodman & Weare 2010)."""

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from minibayes.exceptions import ModelSpecError
from minibayes.samplers.base import Sampler


class EnsembleSampler(Sampler):
    """
    Affine-invariant ensemble sampler (Goodman & Weare 2010).

    Uses K walkers moving in parallel via stretch moves. Each walker
    is treated as an implicit chain, producing samples at each iteration.

    Parameters
    ----------
    stretch_scale : float
        Stretch move scale parameter 'a'. Must be > 1.0.
        Default: 2.0 (recommended by Goodman & Weare).

    References
    ----------
    Goodman & Weare (2010) "Ensemble samplers with affine invariance"
    Foreman-Mackey et al. (2013) "emcee: The MCMC Hammer"
    """

    def __init__(self, stretch_scale: float = 2.0) -> None:
        super().__init__()
        if stretch_scale <= 1.0:
            raise ModelSpecError("stretch_scale must be > 1.0")

        self._a: float = stretch_scale
        # Ensemble uses NDArray for efficiency (overrides base class list type)
        self._log_probs: NDArray[np.float64] | None = None  # type: ignore[assignment]
        self._positions: NDArray[np.float64] | None = None  # Cached walker positions
        self._param_names: list[str] | None = None
        self._accept_count: int = 0
        self._step_count: int = 0

    # -------------------------------------------------------------------------
    # Stateful interface (overrides base class)
    # -------------------------------------------------------------------------

    def initialize(
        self,
        initial_states: list[dict[str, float]],
        log_prob_fn: Callable[[dict[str, float]], float],
    ) -> None:
        """
        Initialize walkers from list of K initial states.

        Parameters
        ----------
        initial_states : list[dict[str, float]]
            K initial states. Must have K >= 2 and K even.
        log_prob_fn : Callable
            Function to compute log probability.
        """
        num_walkers: int = len(initial_states)
        if num_walkers < 2 or num_walkers % 2 != 0:
            raise ModelSpecError("num_walkers must be even and >= 2")

        # Call parent to store states
        super().initialize(initial_states, log_prob_fn)

        self._param_names = list(initial_states[0].keys())

        # Build cached positions array for efficient access
        positions_list: list[list[float]] = [
            [state[name] for name in self._param_names] for state in initial_states
        ]
        self._positions = np.array(positions_list, dtype=np.float64)

        # Compute initial log probabilities
        self._log_probs = np.zeros(num_walkers, dtype=np.float64)
        for k, state in enumerate(initial_states):
            lp: float = log_prob_fn(state)
            self._log_probs[k] = lp if np.isfinite(lp) else float("-inf")

        # Reset acceptance tracking
        self._accept_count = 0
        self._step_count = 0

    def advance(
        self,
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        warmup: bool = False,
        step_num: int = 0,
    ) -> float:
        """
        Advance all walkers by one stretch move iteration.

        Uses the red-blue splitting pattern: update first half using
        second half as complement, then update second half using first half.

        Parameters
        ----------
        log_prob_fn : Callable
            Function to compute log probability.
        rng : np.random.Generator
            NumPy random generator.
        warmup : bool
            Ignored (ensemble sampler doesn't adapt during warmup).
        step_num : int
            Ignored.

        Returns
        -------
        float
            Acceptance rate for this iteration.
        """
        if not self._states or self._log_probs is None:
            raise RuntimeError("EnsembleSampler not initialized. Call initialize() first.")

        num_walkers: int = len(self._states)
        half: int = num_walkers // 2

        # Update first half using second half as complement
        accept1: int = self._update_half(0, half, half, num_walkers, log_prob_fn, rng)

        # Update second half using first half as complement
        accept2: int = self._update_half(half, num_walkers, 0, half, log_prob_fn, rng)

        self._accept_count += accept1 + accept2
        self._step_count += num_walkers
        return float(accept1 + accept2) / num_walkers

    def _update_half(
        self,
        active_start: int,
        active_end: int,
        comp_start: int,
        comp_end: int,
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
    ) -> int:
        """Update walkers in [active_start:active_end] using complement."""
        if self._param_names is None or self._log_probs is None or self._positions is None:
            raise RuntimeError("Sampler not initialized")

        n_active: int = active_end - active_start
        n_comp: int = comp_end - comp_start
        ndim: int = len(self._param_names)

        # Use cached positions array (no list conversion needed)
        active: NDArray[np.float64] = self._positions[active_start:active_end]
        complement: NDArray[np.float64] = self._positions[comp_start:comp_end]
        active_lp: NDArray[np.float64] = self._log_probs[active_start:active_end]

        # Sample z values: z = ((a-1)*U + 1)^2 / a
        u: NDArray[np.float64] = rng.uniform(size=n_active)
        z: NDArray[np.float64] = ((self._a - 1.0) * u + 1.0) ** 2 / self._a

        # Select random complementary walkers
        comp_idx: NDArray[np.int64] = rng.integers(0, n_comp, size=n_active)
        x_comp: NDArray[np.float64] = complement[comp_idx]

        # Proposal: y = x_comp + z * (x_active - x_comp)
        z_col: NDArray[np.float64] = z[:, np.newaxis]
        proposals: NDArray[np.float64] = x_comp + z_col * (active - x_comp)

        # Log acceptance factor: (ndim - 1) * log(z)
        log_factors: NDArray[np.float64] = (ndim - 1) * np.log(z)

        # Accept/reject each proposal
        accepts: int = 0
        for i in range(n_active):
            prop_dict: dict[str, float] = {}
            for j in range(ndim):
                prop_val: float = cast("float", proposals[i, j])
                prop_dict[self._param_names[j]] = prop_val
            prop_lp: float = log_prob_fn(prop_dict)

            if not np.isfinite(prop_lp):
                continue

            log_factor_i: float = cast("float", log_factors[i])
            active_lp_i: float = cast("float", active_lp[i])
            log_alpha: float = log_factor_i + prop_lp - active_lp_i
            log_u: float = float(np.log(rng.uniform()))

            if log_u < log_alpha:
                self._states[active_start + i] = prop_dict
                self._positions[active_start + i, :] = proposals[i, :]
                self._log_probs[active_start + i] = prop_lp
                accepts += 1

        return accepts

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate."""
        if self._step_count == 0:
            return 0.0
        return self._accept_count / self._step_count

    # -------------------------------------------------------------------------
    # Stateless interface (for backwards compatibility)
    # -------------------------------------------------------------------------

    def step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], bool]:
        """
        Take one step (stateless interface).

        For ensemble sampler, this advances all walkers and returns
        the first walker's state.
        """
        if not self._states:
            raise RuntimeError("EnsembleSampler not initialized. Call initialize() first.")

        self.advance(log_prob_fn, rng)
        return self._states[0].copy(), True

    def warmup_step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        step_num: int,
    ) -> tuple[dict[str, float], bool]:
        """Same as step (ensemble sampler doesn't adapt during warmup)."""
        return self.step(current, log_prob_fn, rng)
