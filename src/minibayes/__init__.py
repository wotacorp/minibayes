"""
minibayes - Minimal Bayesian inference library.

A lightweight library for Bayesian inference designed for production
deployment on resource-constrained environments.
"""

from minibayes import distributions as dist
from minibayes.comparison import WAICResult, waic
from minibayes.inference import sample
from minibayes.model import Model
from minibayes.predictive import sample_posterior_predictive, sample_prior_predictive
from minibayes.results import InferenceResult

__version__ = "0.1.0"

__all__ = [
    "sample",
    "sample_posterior_predictive",
    "sample_prior_predictive",
    "Model",
    "dist",
    "InferenceResult",
    "waic",
    "WAICResult",
    "__version__",
]
