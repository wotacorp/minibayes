"""Custom exceptions for minibayes."""


class MinibayesError(Exception):
    """Base exception for minibayes."""


class SamplingError(MinibayesError):
    """Raised when sampling fails."""


class SamplingTimeoutError(SamplingError):
    """Raised when sampling exceeds the specified timeout."""


class ConvergenceWarning(UserWarning):
    """Raised when diagnostics suggest non-convergence."""


class NumericalError(MinibayesError):
    """Raised on numerical issues (NaN, Inf in log_prob)."""


class ModelSpecError(MinibayesError):
    """Raised when model is mis-specified."""
