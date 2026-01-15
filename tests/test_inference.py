"""End-to-end tests for mb.sample()."""

import time

import numpy as np
import pytest

import minibayes as mb
from minibayes import dist
from minibayes.model import StructuredParams
from minibayes.params import ParamContext
from minibayes.results import InferenceResult


# Module-level likelihood functions for parallel testing (must be picklable)
def _zero_likelihood(params: StructuredParams, data: object) -> float:
    """A likelihood that always returns 0 (flat)."""
    return 0.0


def _quadratic_likelihood(params: StructuredParams, data: object) -> float:
    """A simple quadratic likelihood centered at 0."""
    x = params["x"]
    if isinstance(x, np.ndarray):
        return float(-np.sum(x**2))
    return float(-(x**2))


# Module-level priors function for parallel testing
def _simple_priors(p: ParamContext) -> None:
    """A simple prior for x."""
    p("x", dist.Normal(0, 1))


class TestSample:
    """Tests for mb.sample() function."""

    def test_with_model(self) -> None:
        """Test sample() with Model instance."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 10))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: float(np.sum(dist.Normal(p["mu"], 1).log_prob(d))),
        )
        data = np.array([1.0, 2.0, 3.0])

        result = mb.sample(model, data=data, num_samples=100, num_warmup=50, seed=42)

        assert "mu" in result.samples
        assert result.samples["mu"].shape == (1, 100)

    def test_with_callable_raises(self) -> None:
        """Test sample() raises for non-Model input."""

        # sample() now only accepts Model, not Callable
        def log_prob(params: StructuredParams, data: object) -> float:
            return float(-(params["x"] ** 2))

        with pytest.raises(AttributeError):
            # This should fail since log_prob doesn't have Model attributes
            mb.sample(log_prob, data=None, initial={"x": 0.0}, num_samples=10)  # type: ignore[arg-type]

    def test_returns_inference_result(self) -> None:
        """Test return type is InferenceResult."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
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

        def priors(p: ParamContext) -> None:
            p("a", dist.Normal(0, 1))
            p("b", dist.HalfNormal(1))

        model = mb.Model(
            priors=priors,
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

        def priors(p: ParamContext) -> None:
            p("a", dist.Normal(0, 1))
            p("b", dist.HalfNormal(1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=100, num_warmup=20, num_chains=3, seed=42)

        assert result.samples["a"].shape == (3, 100)
        assert result.samples["b"].shape == (3, 100)
        assert result.num_chains == 3

    def test_acceptance_rate_shape(self) -> None:
        """Test acceptance rate is always array with shape (num_chains,)."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=50, num_warmup=20, num_chains=2, seed=42)

        assert isinstance(result.acceptance_rate, np.ndarray)
        assert result.acceptance_rate.shape == (2,)

    def test_seed_reproducibility(self) -> None:
        """Test same seed gives same results."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: float(-d * p["x"] ** 2),
        )
        data = 1.0

        result1 = mb.sample(model, data=data, num_samples=50, num_warmup=20, seed=42)
        result2 = mb.sample(model, data=data, num_samples=50, num_warmup=20, seed=42)

        np.testing.assert_array_equal(result1.samples["x"], result2.samples["x"])

    def test_different_seeds_give_different_results(self) -> None:
        """Test different seeds give different results."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
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
        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, tau))

        model = mb.Model(
            priors=priors,
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

        def priors(p: ParamContext) -> None:
            p("sigma", dist.HalfNormal(5))

        model = mb.Model(
            priors=priors,
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

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
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

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        from minibayes.exceptions import ModelSpecError

        with pytest.raises(ModelSpecError):
            mb.sample(model, num_samples=10, sampler="invalid")

    def test_initial_values(self) -> None:
        """Test providing initial parameter values."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=10, num_warmup=5, initial={"x": 5.0}, seed=42)

        # First sample should be near initial value (after warmup it may have moved)
        assert result.num_samples == 10

    def test_elapsed_time_recorded(self) -> None:
        """Test that elapsed time is recorded."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=50, num_warmup=20, seed=42)

        assert result.elapsed_time > 0


class TestSampleProgress:
    """Tests for progress bar functionality."""

    def test_progress_false_no_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test progress=False produces no stderr output."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        mb.sample(model, num_samples=10, num_warmup=5, progress=False, seed=42)

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_progress_true_produces_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test progress=True produces stderr output."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        mb.sample(model, num_samples=10, num_warmup=5, progress=True, seed=42)

        captured = capsys.readouterr()
        assert "Warmup" in captured.err
        assert "Sampling" in captured.err

    def test_progress_multiple_chains(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test progress shows chain numbers."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
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

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=10, num_warmup=5, timeout=None, seed=42)
        assert result.num_samples == 10

    def test_timeout_sufficient_no_error(self) -> None:
        """Test large timeout doesn't raise."""

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=10, num_warmup=5, timeout=60.0, seed=42)
        assert result.num_samples == 10

    def test_timeout_exceeded_raises(self) -> None:
        """Test timeout raises SamplingTimeoutError."""
        import time as time_module

        from minibayes.exceptions import SamplingTimeoutError

        def slow_likelihood(p: StructuredParams, d: object) -> float:
            time_module.sleep(0.05)  # 50ms per step
            return 0.0

        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=slow_likelihood,
        )

        with pytest.raises(SamplingTimeoutError):
            mb.sample(model, num_samples=100, num_warmup=50, timeout=0.1, seed=42)

    def test_timeout_error_is_sampling_error(self) -> None:
        """Test SamplingTimeoutError inherits from SamplingError."""
        from minibayes.exceptions import SamplingError, SamplingTimeoutError

        assert issubclass(SamplingTimeoutError, SamplingError)


class TestSampleParallel:
    """Tests for parallel chain execution."""

    def test_parallel_chains_shape(self) -> None:
        """Test parallel sampling produces correct shape."""
        # Uses module-level function (required for parallel)
        model = mb.Model(
            priors=_simple_priors,
            log_likelihood=_zero_likelihood,
        )

        result = mb.sample(model, num_samples=50, num_warmup=20, num_chains=4, parallel=True, seed=42)

        assert result.samples["x"].shape == (4, 50)
        assert result.num_chains == 4
        assert result.acceptance_rate.shape == (4,)

    def test_parallel_same_as_sequential(self) -> None:
        """Test parallel and sequential give same results with same seed."""
        # Uses module-level function (required for parallel)
        model = mb.Model(
            priors=_simple_priors,
            log_likelihood=_quadratic_likelihood,
        )

        result_seq = mb.sample(model, num_samples=50, num_warmup=20, num_chains=2, parallel=False, seed=42)
        result_par = mb.sample(model, num_samples=50, num_warmup=20, num_chains=2, parallel=True, seed=42)

        np.testing.assert_array_equal(result_seq.samples["x"], result_par.samples["x"])

    def test_parallel_single_chain_fallback(self) -> None:
        """Test parallel=True with single chain works (falls back to sequential)."""

        # Single chain falls back to sequential, so lambdas still work
        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=50, num_warmup=20, num_chains=1, parallel=True, seed=42)

        assert result.samples["x"].shape == (1, 50)

    def test_parallel_speedup_with_large_workload(self) -> None:
        """Test that parallel execution provides speedup for large workloads.

        Note: For small workloads, process startup overhead may make parallel
        slower. This test uses a larger workload where parallel should help.
        """
        model = mb.Model(
            priors=_simple_priors,
            log_likelihood=_zero_likelihood,
        )

        # Use larger workload to amortize process startup cost
        num_samples = 2000
        num_warmup = 500

        # Sequential timing
        t0 = time.perf_counter()
        mb.sample(
            model,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=4,
            parallel=False,
            seed=42,
        )
        seq_time = time.perf_counter() - t0

        # Parallel timing
        t0 = time.perf_counter()
        mb.sample(
            model,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=4,
            parallel=True,
            seed=42,
        )
        par_time = time.perf_counter() - t0

        # Print timing for debugging (visible with pytest -v)
        print(f"\nSequential: {seq_time:.2f}s, Parallel: {par_time:.2f}s")

        # Parallel should be faster for large workloads
        # Use generous margin (0.95) since CI environments vary
        assert par_time < seq_time * 0.95, f"Parallel ({par_time:.2f}s) not faster than sequential ({seq_time:.2f}s)"

    def test_progress_shows_elapsed_time(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test progress=True shows elapsed time at end."""

        # Sequential mode, lambdas work fine
        def priors(p: ParamContext) -> None:
            p("x", dist.Normal(0, 1))

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        mb.sample(model, num_samples=10, num_warmup=5, progress=True, seed=42)

        captured = capsys.readouterr()
        assert "Sampling complete" in captured.err


class TestVectorParameters:
    """Tests for vector parameter support in inference."""

    def test_vector_parameter_shape(self) -> None:
        """Test vector parameters have correct 3D shape in results."""

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 1))
            p("theta", dist.Normal(0, 1), size=5)

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=50, num_warmup=20, seed=42)

        # Scalar param: (chains, samples)
        assert result.samples["mu"].shape == (1, 50)
        # Vector param: (chains, samples, size)
        assert result.samples["theta"].shape == (1, 50, 5)

    def test_vector_parameter_multiple_chains(self) -> None:
        """Test vector parameters with multiple chains."""

        def priors(p: ParamContext) -> None:
            p("theta", dist.Normal(0, 1), size=3)

        model = mb.Model(
            priors=priors,
            log_likelihood=lambda p, d: 0.0,
        )

        result = mb.sample(model, num_samples=50, num_warmup=20, num_chains=2, seed=42)

        # (num_chains, num_samples, size)
        assert result.samples["theta"].shape == (2, 50, 3)

    def test_hierarchical_model_inference(self) -> None:
        """Test inference with hierarchical model."""
        J = 4

        def priors(p: ParamContext) -> None:
            mu = p("mu", dist.Normal(0, 5))
            tau = p("tau", dist.HalfNormal(2))
            p("theta", dist.Normal(mu, tau), size=J)

        def log_likelihood(params: StructuredParams, data: object) -> float:
            y, sigma = data
            theta = params["theta"]
            assert isinstance(theta, np.ndarray)
            ll: float = 0.0
            for j in range(J):
                ll += float(dist.Normal(theta[j], sigma[j]).log_prob(y[j]))
            return ll

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)

        # Simple data
        y = np.array([1.0, 2.0, 3.0, 4.0])
        sigma = np.array([1.0, 1.0, 1.0, 1.0])

        result = mb.sample(model, data=(y, sigma), num_samples=100, num_warmup=50, seed=42)

        assert result.samples["mu"].shape == (1, 100)
        assert result.samples["tau"].shape == (1, 100)
        assert result.samples["theta"].shape == (1, 100, J)

        # tau should be positive
        assert np.all(result.samples["tau"] > 0)
