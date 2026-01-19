"""Adaptive Metropolis sampler."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from minibayes.exceptions import ModelSpecError
from minibayes.samplers.base import Sampler
from minibayes.utils import check_finite


class AdaptiveMetropolis(Sampler):
    """
    Adaptive Metropolis with covariance tuning.

    During warmup, adapts proposal covariance based on sample history
    using the algorithm of Haario et al. (2001).
    Uses the 2.38²/d scaling factor (optimal for Gaussian targets).

    Parameters
    ----------
    initial_scale : float
        Initial proposal scale before adaptation.

    References
    ----------
    Haario et al. (2001) "An adaptive Metropolis algorithm"
    Roberts & Rosenthal (2001) "Optimal scaling for various MH algorithms"
    """

    def __init__(
        self,
        initial_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if initial_scale <= 0:
            raise ModelSpecError("initial_scale must be positive")

        self._initial_scale: float = initial_scale
        self._sample_history: list[dict[str, float]] = []
        self._param_names: list[str] | None = None
        self._cov: NDArray[np.float64] | None = None
        self._frozen: bool = False

    def _compute_covariance(self) -> None:
        """Compute adapted covariance from sample history."""
        if self._param_names is None or len(self._sample_history) < 2:
            return

        d: int = len(self._param_names)

        # Convert samples to array (n_samples, d)
        samples_list: list[list[float]] = [[s[name] for name in self._param_names] for s in self._sample_history]
        samples: NDArray[np.float64] = np.array(samples_list, dtype=np.float64)

        # Empirical covariance with optimal scaling: 2.38²/d
        emp_cov: NDArray[np.float64] = np.cov(samples.T)

        # Handle 1D case where np.cov returns scalar
        if d == 1:
            scalar_cov: float = float(emp_cov)
            cov_1d: NDArray[np.float64] = np.zeros((1, 1), dtype=np.float64)
            cov_1d[0, 0] = scalar_cov
            emp_cov = cov_1d

        scale: float = 2.38**2 / d

        # Regularize for numerical stability (ensure positive definite)
        identity: NDArray[np.float64] = np.eye(d, dtype=np.float64)
        self._cov = scale * emp_cov + 1e-6 * identity

    def _propose(self, current: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
        """Generate proposal using adapted covariance."""
        if self._param_names is None:
            self._param_names = list(current.keys())

        if self._cov is None:
            # Fall back to independent proposals with initial_scale
            proposal: dict[str, float] = {}
            for name, val in current.items():
                noise: float = float(rng.normal(0.0, self._initial_scale))
                proposal[name] = val + noise
            return proposal

        # Multivariate normal proposal with adapted covariance
        current_values: list[float] = [current[name] for name in self._param_names]
        current_vec: NDArray[np.float64] = np.array(current_values, dtype=np.float64)
        proposal_vec: NDArray[np.float64] = rng.multivariate_normal(current_vec, self._cov)

        result: dict[str, float] = {}
        for i, name in enumerate(self._param_names):
            prop_val: np.float64 = proposal_vec[i]
            result[name] = float(prop_val)
        return result

    def step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], bool]:
        """
        Take one MCMC step with current (possibly adapted) covariance.

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
        # Initialize param names on first call
        if self._param_names is None:
            self._param_names = list(current.keys())

        # Compute log_prob at current position
        log_prob_current: float = log_prob_fn(current)
        check_finite(log_prob_current, "log_prob(current)")

        # Generate proposal
        proposal: dict[str, float] = self._propose(current, rng)

        # Compute log_prob at proposal
        log_prob_proposal: float = log_prob_fn(proposal)

        # Accept/reject (handle non-finite proposal by rejecting)
        if not np.isfinite(log_prob_proposal):
            return current, False

        log_alpha: float = log_prob_proposal - log_prob_current
        # Use 1-U instead of U to avoid log(0) when U=0
        # Since U ~ Uniform(0,1), 1-U ~ Uniform(0,1) with same distribution
        # but 1-U is never exactly 0 (since U is never exactly 1)
        log_u: float = float(np.log(1.0 - rng.uniform()))

        if log_u < log_alpha:
            return proposal, True
        return current, False

    def _step_cached(
        self,
        current: dict[str, float],
        log_prob_current: float,
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], bool, float]:
        """
        Take one MCMC step with cached log_prob.

        This avoids recomputing log_prob(current) when the previous
        proposal was rejected.

        Parameters
        ----------
        current : dict
            Current parameter values (unconstrained space).
        log_prob_current : float
            Cached log_prob at current position.
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
        new_log_prob : float
            Log prob of the returned state (for caching).
        """
        # Initialize param names on first call
        if self._param_names is None:
            self._param_names = list(current.keys())

        check_finite(log_prob_current, "log_prob(current)")

        # Generate proposal
        proposal: dict[str, float] = self._propose(current, rng)

        # Compute log_prob at proposal
        log_prob_proposal: float = log_prob_fn(proposal)

        # Accept/reject (handle non-finite proposal by rejecting)
        if not np.isfinite(log_prob_proposal):
            return current, False, log_prob_current

        log_alpha: float = log_prob_proposal - log_prob_current
        log_u: float = float(np.log(1.0 - rng.uniform()))

        if log_u < log_alpha:
            return proposal, True, log_prob_proposal
        return current, False, log_prob_current

    def advance(
        self,
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        warmup: bool = False,
        step_num: int = 0,
    ) -> float:
        """
        Advance all chains by one step using cached log_probs.

        This override uses cached log_prob values to avoid redundant
        computations when proposals are rejected.
        """
        if not self._states:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")

        accepts: int = 0
        for i in range(len(self._states)):
            if warmup:
                # Warmup uses standard step (for adaptation)
                new_state, accepted = self.warmup_step(
                    self._states[i], log_prob_fn, rng, step_num
                )
                if accepted:
                    self._log_probs[i] = log_prob_fn(new_state)
            else:
                # Use cached version for sampling
                new_state, accepted, new_lp = self._step_cached(
                    self._states[i], self._log_probs[i], log_prob_fn, rng
                )
                self._log_probs[i] = new_lp

            self._states[i] = new_state
            # Update positions array (use base class _param_names via get_param_names)
            if self._positions is not None:
                param_names: list[str] = self.get_param_names()
                for j, name in enumerate(param_names):
                    self._positions[i, j] = new_state[name]
            accepts += int(accepted)

        return accepts / len(self._states)

    def warmup_step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        step_num: int,
    ) -> tuple[dict[str, float], bool]:
        """
        Take one warmup step with adaptation.

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
        if self._frozen:
            return self.step(current, log_prob_fn, rng)

        # Take MH step
        new_state, accepted = self.step(current, log_prob_fn, rng)

        # Store sample for covariance estimation
        self._sample_history.append(new_state.copy())

        # Update covariance periodically (every 50 steps after collecting 100 samples)
        # Wait for 100 samples to get a more stable covariance estimate
        if step_num >= 100 and step_num % 50 == 0:
            self._compute_covariance()

        return new_state, accepted

    def freeze(self) -> None:
        """
        Stop adaptation and free memory.

        Call this after warmup to prevent further adaptation
        and release sample history memory.
        """
        self._frozen = True
        self._sample_history = []

    def post_warmup(self) -> None:
        """Freeze adaptation after warmup completes."""
        self.freeze()
