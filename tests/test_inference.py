"""End-to-end tests for mb.sample()."""

import pytest


class TestSample:
    """Tests for mb.sample() function."""

    def test_with_model(self) -> None:
        """Test sample() with Model instance."""
        pytest.skip("Not implemented")

    def test_with_callable(self) -> None:
        """Test sample() with log_prob function."""
        pytest.skip("Not implemented")

    def test_returns_inference_result(self) -> None:
        """Test return type is InferenceResult."""
        pytest.skip("Not implemented")

    def test_samples_shape(self) -> None:
        """Test samples have correct shape."""
        pytest.skip("Not implemented")

    def test_multiple_chains(self) -> None:
        """Test multiple chains."""
        pytest.skip("Not implemented")

    def test_seed_reproducibility(self) -> None:
        """Test same seed gives same results."""
        pytest.skip("Not implemented")

    def test_normal_normal_analytical(self) -> None:
        """Test Normal-Normal model matches analytical posterior."""
        pytest.skip("Not implemented")

    def test_beta_binomial_analytical(self) -> None:
        """Test Beta-Binomial model matches analytical posterior."""
        pytest.skip("Not implemented")
