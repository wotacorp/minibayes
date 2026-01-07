"""
minibayes - Minimal Bayesian inference library.

A lightweight library for Bayesian inference designed for production
deployment on resource-constrained environments.
"""

from minibayes import distributions as dist
from minibayes.inference import sample
from minibayes.model import Model
from minibayes.results import InferenceResult

__version__ = "0.1.0"

__all__ = [
    "sample",
    "Model",
    "dist",
    "InferenceResult",
    "__version__",
]
