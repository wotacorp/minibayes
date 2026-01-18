"""Random walk Metropolis-Hastings sampler."""

from collections.abc import Callable

import numpy as np

from minibayes.exceptions import ModelSpecError
from minibayes.samplers.base import Sampler
from minibayes.utils import check_finite


class MetropolisHastings(Sampler):
    """
    Random walk Metropolis-Hastings sampler.

    Parameters
    ----------
    proposal_scale : float or dict
        Standard deviation of Gaussian proposal.
        If dict, specifies per-parameter scales.
    """

    def __init__(
        self,
        proposal_scale: float | dict[str, float] = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(proposal_scale, dict):
            for name, scale in proposal_scale.items():
                if scale <= 0:
                    raise ModelSpecError(f"proposal_scale[{name}] must be positive")
            self._proposal_scale: float | dict[str, float] = proposal_scale
        else:
            if proposal_scale <= 0:
                raise ModelSpecError("proposal_scale must be positive")
            self._proposal_scale = proposal_scale

    def _get_scale(self, param_name: str) -> float:
        """Get proposal scale for a parameter."""
        if isinstance(self._proposal_scale, dict):
            return self._proposal_scale.get(param_name, 1.0)
        return self._proposal_scale

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
        # Compute log_prob at current position
        log_prob_current: float = log_prob_fn(current)
        check_finite(log_prob_current, "log_prob(current)")

        # Generate proposal
        proposal: dict[str, float] = {}
        for name, value in current.items():
            scale: float = self._get_scale(name)
            noise: float = float(rng.normal(0.0, scale))
            proposal[name] = value + noise

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
        check_finite(log_prob_current, "log_prob(current)")

        # Generate proposal
        proposal: dict[str, float] = {}
        for name, value in current.items():
            scale: float = self._get_scale(name)
            noise: float = float(rng.normal(0.0, scale))
            proposal[name] = value + noise

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
                # Warmup uses standard step (no caching, simpler)
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
            # Update positions array
            if self._positions is not None and self._param_names is not None:
                for j, name in enumerate(self._param_names):
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
        Take one warmup step.

        For basic MH, warmup is identical to regular sampling (no adaptation).

        Parameters
        ----------
        current : dict
            Current parameter values (unconstrained space).
        log_prob_fn : Callable
            Function params -> log_prob (in unconstrained space).
        rng : Generator
            NumPy random generator.
        step_num : int
            Current warmup step number (unused in basic MH).

        Returns
        -------
        new_state : dict
            New parameter values.
        accepted : bool
            Whether proposal was accepted.
        """
        return self.step(current, log_prob_fn, rng)
