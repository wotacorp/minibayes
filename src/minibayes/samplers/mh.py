"""Random walk Metropolis-Hastings sampler."""

from typing import Callable

import numpy as np

from minibayes.samplers.base import Sampler


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
        raise NotImplementedError()

    def step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], bool]:
        raise NotImplementedError()

    def warmup_step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        step_num: int,
    ) -> tuple[dict[str, float], bool]:
        raise NotImplementedError()
