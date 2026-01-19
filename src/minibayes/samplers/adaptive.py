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
        self._cov: NDArray[np.float64] | None = None
        self._frozen: bool = False

    def _compute_covariance(self) -> None:
        """Compute adapted covariance from sample history."""
        if self._param_names is None or len(self._sample_history) < 2:
            return

        d: int = len(self._param_names)

        # Convert samples to array (n_samples, d)
        samples_list: list[list[float]] = [
            [s[name] for name in self._param_names] for s in self._sample_history
        ]
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

    def _propose(
        self, current: dict[str, float], rng: np.random.Generator
    ) -> dict[str, float]:
        """Generate proposal using adapted covariance."""
        if self._param_names is None:
            raise RuntimeError("Sampler not initialized")

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
        proposal_vec: NDArray[np.float64] = rng.multivariate_normal(
            current_vec, self._cov
        )

        result: dict[str, float] = {}
        for i, name in enumerate(self._param_names):
            prop_val: np.float64 = proposal_vec[i]
            result[name] = float(prop_val)
        return result

    def advance(
        self,
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        warmup: bool = False,
        step_num: int = 0,
    ) -> float:
        """
        Advance all chains by one step.

        During warmup, adapts proposal covariance based on sample history.
        Uses cached log_prob values to avoid redundant computations.
        """
        if not self._states:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")

        accepts: int = 0
        for i in range(len(self._states)):
            current: dict[str, float] = self._states[i]
            log_prob_current: float = self._log_probs[i]
            check_finite(log_prob_current, "log_prob(current)")

            # Generate proposal
            proposal: dict[str, float] = self._propose(current, rng)

            # Compute log_prob at proposal
            log_prob_proposal: float = log_prob_fn(proposal)

            # Accept/reject (reject non-finite proposals)
            accepted: bool = False
            if np.isfinite(log_prob_proposal):
                log_alpha: float = log_prob_proposal - log_prob_current
                log_u: float = float(np.log(1.0 - rng.uniform()))

                if log_u < log_alpha:
                    self._states[i] = proposal
                    self._log_probs[i] = log_prob_proposal
                    self._update_positions(i, proposal)
                    accepted = True
                    accepts += 1
                else:
                    self._update_positions(i, current)
            else:
                self._update_positions(i, current)

            # During warmup: store sample and adapt covariance
            if warmup and not self._frozen:
                new_state: dict[str, float] = proposal if accepted else current
                self._sample_history.append(new_state.copy())

        # Update covariance periodically during warmup
        if warmup and not self._frozen and step_num >= 100 and step_num % 50 == 0:
            self._compute_covariance()

        return accepts / len(self._states)

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
