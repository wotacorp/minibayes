"""Shared pytest fixtures for minibayes tests."""

import pytest
import numpy as np


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_params() -> dict[str, float]:
    """Sample parameter dictionary for testing."""
    return {"alpha": 1.0, "beta": 0.5, "sigma": 2.0}
