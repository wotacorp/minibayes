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
        log_u: float = float(np.log(rng.uniform()))

        if log_u < log_alpha:
            return proposal, True
        return current, False

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
