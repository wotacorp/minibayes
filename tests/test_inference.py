"""End-to-end tests for mb.sample()."""

import numpy as np
import pytest

import minibayes as mb
from minibayes import dist
from minibayes.results import InferenceResult


class TestSample:
    """Tests for mb.sample() function."""

    def test_with_model(self) -> None:
        """Test sample() with Model instance."""
        model = mb.Model(
            priors={"mu": dist.Normal(0, 10)},
            log_likelihood=lambda p, d: float(dist.Normal(p["mu"], 1).log_prob(d).sum()),
        )
        data = np.array([1.0, 2.0, 3.0])

        result = mb.sample(model, data=data, num_samples=100, num_warmup=50, seed=42)

        assert "mu" in result.samples
        assert result.samples["mu"].shape == (1, 100)

    def test_with_callable_raises(self) -> None:
        """Test sample() raises for non-Model input."""

        # sample() now only accepts Model, not Callable
        def log_prob(params: dict[str, float], data: object) -> float:
            return float(-(params["x"] ** 2))

        with pytest.raises(AttributeError):
            # This should fail since log_prob doesn't have Model attributes
            mb.sample(log_prob, data=None, initial={"x": 0.0}, num_samples=10)  # type: ignore[arg-type]

    def test_returns_inference_result(self) -> None:
        """Test return type is InferenceResult."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=50, num_warmup=10, seed=42)

        assert isinstance(result, InferenceResult)
        assert result.num_samples == 50
        assert result.num_warmup == 10
        assert result.num_chains == 1
        assert result.sampler == "adaptive_mh"

    def test_samples_shape_single_chain(self) -> None:
        """Test samples have correct shape for single chain."""
        model = mb.Model(
            priors={"a": dist.Normal(0, 1), "b": dist.HalfNormal(1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=100, num_warmup=20, num_chains=1, seed=42)

        # Always (num_chains, num_samples) for consistency
        assert result.samples["a"].shape == (1, 100)
        assert result.samples["b"].shape == (1, 100)
        assert result.samples_unconstrained["a"].shape == (1, 100)
        assert result.samples_unconstrained["b"].shape == (1, 100)
        assert result.acceptance_rate.shape == (1,)

    def test_samples_shape_multiple_chains(self) -> None:
        """Test samples have correct shape for multiple chains."""
        model = mb.Model(
            priors={"a": dist.Normal(0, 1), "b": dist.HalfNormal(1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=100, num_warmup=20, num_chains=3, seed=42)

        assert result.samples["a"].shape == (3, 100)
        assert result.samples["b"].shape == (3, 100)
        assert result.num_chains == 3

    def test_acceptance_rate_shape(self) -> None:
        """Test acceptance rate is always array with shape (num_chains,)."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=50, num_warmup=20, num_chains=2, seed=42)

        assert isinstance(result.acceptance_rate, np.ndarray)
        assert result.acceptance_rate.shape == (2,)

    def test_seed_reproducibility(self) -> None:
        """Test same seed gives same results."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: float(-d * p["x"] ** 2),
        )
        data = 1.0

        result1 = mb.sample(model, data=data, num_samples=50, num_warmup=20, seed=42)
        result2 = mb.sample(model, data=data, num_samples=50, num_warmup=20, seed=42)

        np.testing.assert_array_equal(result1.samples["x"], result2.samples["x"])

    def test_different_seeds_give_different_results(self) -> None:
        """Test different seeds give different results."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result1 = mb.sample(model, num_samples=50, num_warmup=20, seed=42)
        result2 = mb.sample(model, num_samples=50, num_warmup=20, seed=123)

        assert not np.allclose(result1.samples["x"], result2.samples["x"])

    def test_normal_normal_analytical(self) -> None:
        """Test Normal-Normal model matches analytical posterior."""
        # Prior: mu ~ N(0, tau^2) with tau = 1
        # Likelihood: y_i ~ N(mu, sigma^2) with sigma = 1 (known)
        # Analytical posterior: mu | y ~ N(mu_post, sigma_post^2)
        # where mu_post = n * y_bar / (n + 1), sigma_post^2 = 1 / (n + 1)

        rng = np.random.default_rng(42)
        true_mu = 2.0
        sigma = 1.0  # known
        tau = 1.0  # prior std
        n = 20
        data = rng.normal(true_mu, sigma, size=n)

        # Analytical posterior
        y_bar = float(np.mean(data))
        posterior_mean = (n * y_bar / sigma**2) / (n / sigma**2 + 1 / tau**2)
        posterior_std = float(np.sqrt(1 / (n / sigma**2 + 1 / tau**2)))

        # Define model
        model = mb.Model(
            priors={"mu": dist.Normal(0, tau)},
            log_likelihood=lambda p, d: float(np.sum(dist.Normal(p["mu"], sigma).log_prob(d))),
        )

        # Run MCMC
        result = mb.sample(
            model,
            data=data,
            num_samples=5000,
            num_warmup=1000,
            sampler="adaptive_mh",
            seed=42,
        )

        samples: np.ndarray[tuple[int], np.dtype[np.float64]] = result.samples["mu"]
        sample_mean: float = float(np.mean(samples))
        sample_std: float = float(np.std(samples))

        # Check posterior mean and std are close to analytical
        assert abs(sample_mean - posterior_mean) < 0.1
        assert abs(sample_std - posterior_std) < 0.05

    def test_constrained_transform(self) -> None:
        """Test that constrained samples are properly transformed."""
        model = mb.Model(
            priors={"sigma": dist.HalfNormal(5)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=100, num_warmup=50, seed=42)

        # Constrained samples should all be positive
        assert np.all(result.samples["sigma"] > 0)

        # Unconstrained samples can be any real number
        # (they are log-transformed)
        assert result.samples_unconstrained["sigma"].shape == (1, 100)

    def test_sampler_mh(self) -> None:
        """Test using basic MH sampler."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(
            model,
            num_samples=50,
            num_warmup=20,
            sampler="mh",
            sampler_kwargs={"proposal_scale": 0.5},
            seed=42,
        )

        assert result.sampler == "mh"
        assert result.samples["x"].shape == (1, 50)

    def test_invalid_sampler_raises(self) -> None:
        """Test invalid sampler name raises error."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        from minibayes.exceptions import ModelSpecError

        with pytest.raises(ModelSpecError):
            mb.sample(model, num_samples=10, sampler="invalid")

    def test_initial_values(self) -> None:
        """Test providing initial parameter values."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=10, num_warmup=5, initial={"x": 5.0}, seed=42)

        # First sample should be near initial value (after warmup it may have moved)
        assert result.num_samples == 10

    def test_elapsed_time_recorded(self) -> None:
        """Test that elapsed time is recorded."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=50, num_warmup=20, seed=42)

        assert result.elapsed_time > 0


class TestSampleProgress:
    """Tests for progress bar functionality."""

    def test_progress_false_no_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test progress=False produces no stderr output."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        mb.sample(model, num_samples=10, num_warmup=5, progress=False, seed=42)

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_progress_true_produces_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test progress=True produces stderr output."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        mb.sample(model, num_samples=10, num_warmup=5, progress=True, seed=42)

        captured = capsys.readouterr()
        assert "Warmup" in captured.err
        assert "Sampling" in captured.err

    def test_progress_multiple_chains(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test progress shows chain numbers."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        mb.sample(model, num_samples=10, num_warmup=5, num_chains=2, progress=True, seed=42)

        captured = capsys.readouterr()
        assert "Chain 1/2" in captured.err
        assert "Chain 2/2" in captured.err


class TestSampleTimeout:
    """Tests for timeout functionality."""

    def test_timeout_none_no_error(self) -> None:
        """Test timeout=None doesn't raise."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=10, num_warmup=5, timeout=None, seed=42)
        assert result.num_samples == 10

    def test_timeout_sufficient_no_error(self) -> None:
        """Test large timeout doesn't raise."""
        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=10, num_warmup=5, timeout=60.0, seed=42)
        assert result.num_samples == 10

    def test_timeout_exceeded_raises(self) -> None:
        """Test timeout raises SamplingTimeoutError."""
        import time as time_module

        from minibayes.exceptions import SamplingTimeoutError

        def slow_likelihood(p: dict[str, float], d: object) -> float:
            time_module.sleep(0.05)  # 50ms per step
            return 0.0

        model = mb.Model(
            priors={"x": dist.Normal(0, 1)},
            log_likelihood=slow_likelihood,
        )

        with pytest.raises(SamplingTimeoutError):
            mb.sample(model, num_samples=100, num_warmup=50, timeout=0.1, seed=42)

    def test_timeout_error_is_sampling_error(self) -> None:
        """Test SamplingTimeoutError inherits from SamplingError."""
        from minibayes.exceptions import SamplingError, SamplingTimeoutError

        assert issubclass(SamplingTimeoutError, SamplingError)
