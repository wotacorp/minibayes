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

    def advance(
        self,
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        warmup: bool = False,
        step_num: int = 0,
    ) -> float:
        """
        Advance all chains by one step.

        Uses cached log_prob values to avoid redundant computations
        when proposals are rejected. For basic MH, warmup and sampling
        are identical (no adaptation).
        """
        if not self._states:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")

        accepts: int = 0
        for i in range(len(self._states)):
            current: dict[str, float] = self._states[i]
            log_prob_current: float = self._log_probs[i]
            check_finite(log_prob_current, "log_prob(current)")

            # Generate proposal
            proposal: dict[str, float] = {}
            for name, value in current.items():
                scale: float = self._get_scale(name)
                noise: float = float(rng.normal(0.0, scale))
                proposal[name] = value + noise

            # Compute log_prob at proposal
            log_prob_proposal: float = log_prob_fn(proposal)

            # Accept/reject (reject non-finite proposals)
            if np.isfinite(log_prob_proposal):
                log_alpha: float = log_prob_proposal - log_prob_current
                # Use 1-U instead of U to avoid log(0) when U=0
                log_u: float = float(np.log(1.0 - rng.uniform()))

                if log_u < log_alpha:
                    self._states[i] = proposal
                    self._log_probs[i] = log_prob_proposal
                    self._update_positions(i, proposal)
                    accepts += 1
                else:
                    self._update_positions(i, current)
            else:
                self._update_positions(i, current)

        return accepts / len(self._states)
