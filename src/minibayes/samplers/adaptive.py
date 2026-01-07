"""Adaptive Metropolis sampler."""

from typing import Callable

import numpy as np

from minibayes.samplers.base import Sampler


class AdaptiveMetropolis(Sampler):
    """
    Adaptive Metropolis with covariance tuning.

    During warmup, adapts proposal covariance based on sample history.
    Uses the 2.38^2/d scaling factor (optimal for Gaussian targets).

    Parameters
    ----------
    initial_scale : float
        Initial proposal scale before adaptation.
    target_acceptance : float
        Target acceptance rate (0.234 optimal for Gaussians).
    """

    def __init__(
        self,
        initial_scale: float = 1.0,
        target_acceptance: float = 0.234,
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
