"""Tests for posterior and prior predictive sampling."""

import numpy as np
import pytest
from numpy.typing import NDArray

import minibayes as mb
from minibayes import dist
from minibayes.predictive import (
    _get_param_dict,
    _get_total_samples,
    sample_posterior_predictive,
    sample_prior_predictive,
)
from minibayes.results import InferenceResult


class TestHelpers:
    """Tests for helper functions."""

    def test_get_total_samples_single_chain(self) -> None:
        """Test _get_total_samples with single chain."""
        samples: dict[str, NDArray[np.float64]] = {
            "mu": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
        assert _get_total_samples(samples) == 5

    def test_get_total_samples_multi_chain(self) -> None:
        """Test _get_total_samples with multi chain."""
        samples: dict[str, NDArray[np.float64]] = {
            "mu": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # 2 chains, 3 samples
        }
        assert _get_total_samples(samples) == 6

    def test_get_param_dict_single_chain(self) -> None:
        """Test _get_param_dict with single chain."""
        samples: dict[str, NDArray[np.float64]] = {
            "mu": np.array([1.0, 2.0, 3.0]),
            "sigma": np.array([0.1, 0.2, 0.3]),
        }
        params = _get_param_dict(samples, 1)
        assert params == {"mu": 2.0, "sigma": 0.2}

    def test_get_param_dict_multi_chain(self) -> None:
        """Test _get_param_dict with multi chain (flattened)."""
        samples: dict[str, NDArray[np.float64]] = {
            "mu": np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 chains, 2 samples
        }
        # Flattened: [1.0, 2.0, 3.0, 4.0]
        params = _get_param_dict(samples, 2)
        assert params["mu"] == 3.0


class TestSamplePosteriorPredictive:
    """Tests for sample_posterior_predictive."""

    @pytest.fixture
    def simple_result(self) -> InferenceResult:
        """Create a simple inference result for testing."""
        return InferenceResult(
            samples={"mu": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])},
            samples_unconstrained={"mu": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])},
            acceptance_rate=np.array([0.5]),
            num_samples=5,
            num_warmup=0,
            num_chains=1,
            sampler="test",
            elapsed_time=0.0,
        )

    def test_basic_usage(self, simple_result: InferenceResult) -> None:
        """Test basic posterior predictive sampling."""

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": np.array([params["mu"] * 2])}

        ppc = sample_posterior_predictive(simple_result, predictive, seed=42)

        assert "y" in ppc
        assert ppc["y"].shape == (5, 1)
        expected: NDArray[np.float64] = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])
        np.testing.assert_array_equal(ppc["y"], expected)

    def test_with_stochastic_predictive(self, simple_result: InferenceResult) -> None:
        """Test predictive with random sampling using dist."""

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            samples: NDArray[np.float64] = dist.Normal(params["mu"], 0.1).sample(
                size=3, rng=rng
            )
            return {"y": samples}

        ppc = sample_posterior_predictive(simple_result, predictive, seed=42)

        assert ppc["y"].shape == (5, 3)
        # Check values are near the means (mu = 1, 2, 3, 4, 5)
        means: NDArray[np.float64] = np.mean(ppc["y"], axis=1)
        expected_means: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(means, expected_means, atol=0.5)

    def test_num_samples_thinning(self, simple_result: InferenceResult) -> None:
        """Test num_samples parameter for thinning."""

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": np.array([params["mu"]])}

        ppc = sample_posterior_predictive(simple_result, predictive, num_samples=2)

        assert ppc["y"].shape[0] == 2

    def test_multi_chain(self) -> None:
        """Test with multi-chain result (flattens chains)."""
        result = InferenceResult(
            samples={"mu": np.array([[1.0, 2.0], [3.0, 4.0]])},  # 2 chains, 2 samples
            samples_unconstrained={"mu": np.array([[1.0, 2.0], [3.0, 4.0]])},
            acceptance_rate=np.array([0.5, 0.5]),
            num_samples=2,
            num_warmup=0,
            num_chains=2,
            sampler="test",
            elapsed_time=0.0,
        )

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": np.array([params["mu"]])}

        ppc = sample_posterior_predictive(result, predictive)

        # Should have 4 predictions (2 chains * 2 samples)
        assert ppc["y"].shape == (4, 1)
        # Values should be [1, 2, 3, 4] (flattened)
        expected: NDArray[np.float64] = np.array([[1.0], [2.0], [3.0], [4.0]])
        np.testing.assert_array_equal(ppc["y"], expected)

    def test_reproducibility(self, simple_result: InferenceResult) -> None:
        """Test same seed gives same results."""

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": dist.Normal(params["mu"], 1.0).sample(size=3, rng=rng)}

        ppc1 = sample_posterior_predictive(simple_result, predictive, seed=123)
        ppc2 = sample_posterior_predictive(simple_result, predictive, seed=123)

        np.testing.assert_array_equal(ppc1["y"], ppc2["y"])

    def test_multiple_outputs(self, simple_result: InferenceResult) -> None:
        """Test predictive returning multiple outputs."""

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            mu: float = params["mu"]
            return {
                "mean": np.array([mu]),
                "sample": dist.Normal(mu, 0.1).sample(size=2, rng=rng),
            }

        ppc = sample_posterior_predictive(simple_result, predictive, seed=42)

        assert "mean" in ppc
        assert "sample" in ppc
        assert ppc["mean"].shape == (5, 1)
        assert ppc["sample"].shape == (5, 2)

    def test_empty_samples_raises(self) -> None:
        """Test ValueError for empty samples."""
        result = InferenceResult(
            samples={},
            samples_unconstrained={},
            acceptance_rate=np.array([0.5]),
            num_samples=0,
            num_warmup=0,
            num_chains=1,
            sampler="test",
            elapsed_time=0.0,
        )

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": np.array([0.0])}

        with pytest.raises(ValueError, match="No samples"):
            sample_posterior_predictive(result, predictive)


class TestSamplePriorPredictive:
    """Tests for sample_prior_predictive."""

    def test_basic_usage(self) -> None:
        """Test basic prior predictive sampling."""
        model = mb.Model(
            priors={"mu": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": dist.Normal(params["mu"], 0.1).sample(size=3, rng=rng)}

        ppc = sample_prior_predictive(model, predictive, num_samples=100, seed=42)

        assert "y" in ppc
        assert ppc["y"].shape == (100, 3)

    def test_reproducibility(self) -> None:
        """Test same seed gives same results."""
        model = mb.Model(
            priors={"mu": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": dist.Normal(params["mu"], 1.0).sample(size=3, rng=rng)}

        ppc1 = sample_prior_predictive(model, predictive, num_samples=50, seed=123)
        ppc2 = sample_prior_predictive(model, predictive, num_samples=50, seed=123)

        np.testing.assert_array_equal(ppc1["y"], ppc2["y"])

    def test_multiple_params(self) -> None:
        """Test with multiple parameters."""
        model = mb.Model(
            priors={
                "mu": dist.Normal(0, 10),
                "sigma": dist.HalfNormal(1),
            },
            log_likelihood=lambda p, d: 0.0,
        )

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": dist.Normal(params["mu"], params["sigma"]).sample(size=5, rng=rng)}

        ppc = sample_prior_predictive(model, predictive, num_samples=100, seed=42)

        assert ppc["y"].shape == (100, 5)


class TestInferenceResultPredict:
    """Tests for InferenceResult.predict() convenience method."""

    def test_predict_method(self) -> None:
        """Test predict() method delegates to sample_posterior_predictive."""
        result = InferenceResult(
            samples={"mu": np.array([[1.0, 2.0, 3.0]])},
            samples_unconstrained={"mu": np.array([[1.0, 2.0, 3.0]])},
            acceptance_rate=np.array([0.5]),
            num_samples=3,
            num_warmup=0,
            num_chains=1,
            sampler="test",
            elapsed_time=0.0,
        )

        def predictive(
            params: dict[str, float], rng: np.random.Generator
        ) -> dict[str, NDArray[np.float64]]:
            return {"y": np.array([params["mu"] + 1])}

        ppc = result.predict(predictive, seed=42)

        assert ppc["y"].shape == (3, 1)
        expected: NDArray[np.float64] = np.array([[2.0], [3.0], [4.0]])
        np.testing.assert_array_equal(ppc["y"], expected)
