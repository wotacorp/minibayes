"""Tests for model comparison metrics."""

import numpy as np
from numpy.typing import NDArray

import minibayes as mb
from minibayes import dist
from minibayes.comparison import WAICResult
from minibayes.model import StructuredParams
from minibayes.params import ParamContext


class TestWAIC:
    """Tests for WAIC computation."""

    def test_waic_simple_linear(self) -> None:
        """Test WAIC on simple linear regression with known good fit."""
        rng = np.random.default_rng(42)
        n = 50
        true_mu = 5.0
        true_sigma = 1.0
        y = rng.normal(true_mu, true_sigma, n)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 10))
            p("sigma", dist.HalfNormal(5))

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], params["sigma"]).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(
            model, data=y, num_samples=500, num_warmup=200, seed=42
        )

        waic_result = mb.waic(result, model, y)

        # WAIC should be finite
        assert np.isfinite(waic_result.waic)
        assert np.isfinite(waic_result.p_waic)
        assert np.isfinite(waic_result.lppd)
        assert np.isfinite(waic_result.se)

        # p_waic should be close to 2 (mu and sigma)
        assert 1.0 < waic_result.p_waic < 5.0

    def test_waic_overfitting(self) -> None:
        """More complex model should have higher p_waic."""
        rng = np.random.default_rng(42)
        n = 30
        y = rng.normal(0, 1, n)

        # Simple model: just mean
        def priors_simple(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 10))

        def ll_simple(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        # Complex model: mean per observation (overfitting)
        def priors_complex(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 10), size=n)

        def ll_complex(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            mu = params["mu"]
            assert isinstance(mu, np.ndarray)
            ll: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
            for i in range(n):
                ll[i] = float(dist.Normal(mu[i], 1.0).log_prob(data[i]))
            return ll

        model_simple = mb.Model(priors=priors_simple, log_likelihood=ll_simple)
        model_complex = mb.Model(priors=priors_complex, log_likelihood=ll_complex)

        result_simple = mb.sample(
            model_simple, data=y, num_samples=300, num_warmup=100, seed=42
        )
        result_complex = mb.sample(
            model_complex, data=y, num_samples=300, num_warmup=100, seed=42
        )

        waic_simple = mb.waic(result_simple, model_simple, y)
        waic_complex = mb.waic(result_complex, model_complex, y)

        # Complex model should have much higher p_waic (more effective parameters)
        assert waic_complex.p_waic > waic_simple.p_waic

    def test_waic_result_attributes(self) -> None:
        """WAICResult has all expected fields."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 20)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(model, data=y, num_samples=100, num_warmup=50, seed=42)

        waic_result = mb.waic(result, model, y)

        assert isinstance(waic_result, WAICResult)
        assert hasattr(waic_result, "waic")
        assert hasattr(waic_result, "p_waic")
        assert hasattr(waic_result, "lppd")
        assert hasattr(waic_result, "se")
        assert hasattr(waic_result, "pointwise")

    def test_pointwise_sums_to_total(self) -> None:
        """sum(pointwise) ≈ waic."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 15)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(model, data=y, num_samples=200, num_warmup=100, seed=42)

        waic_result = mb.waic(result, model, y)

        # Sum of pointwise should equal total WAIC
        np.testing.assert_allclose(
            np.sum(waic_result.pointwise), waic_result.waic, rtol=1e-10
        )

    def test_se_positive(self) -> None:
        """Standard error is positive."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 20)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(model, data=y, num_samples=100, num_warmup=50, seed=42)

        waic_result = mb.waic(result, model, y)

        assert waic_result.se > 0

    def test_waic_single_chain(self) -> None:
        """WAIC works with single chain."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 10)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(
            model, data=y, num_samples=100, num_warmup=50, num_chains=1, seed=42
        )

        assert result.num_chains == 1
        waic_result = mb.waic(result, model, y)
        assert np.isfinite(waic_result.waic)

    def test_waic_multi_chain(self) -> None:
        """WAIC correctly flattens multi-chain samples."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 10)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(
            model, data=y, num_samples=50, num_warmup=20, num_chains=2, seed=42
        )

        assert result.num_chains == 2
        waic_result = mb.waic(result, model, y)

        # Pointwise should have n_obs elements
        assert len(waic_result.pointwise) == len(y)

    def test_convenience_method(self) -> None:
        """Test result.waic() convenience method."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 10)

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(model, data=y, num_samples=100, num_warmup=50, seed=42)

        # Both methods should give same result
        waic_fn = mb.waic(result, model, y)
        waic_method = result.waic(model, y)

        assert waic_fn.waic == waic_method.waic
        assert waic_fn.p_waic == waic_method.p_waic


class TestWAICEdgeCases:
    """Edge case tests for WAIC."""

    def test_single_observation(self) -> None:
        """Edge case: n=1 observation."""
        y = np.array([1.5])

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 5))

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(model, data=y, num_samples=100, num_warmup=50, seed=42)

        waic_result = mb.waic(result, model, y)

        assert np.isfinite(waic_result.waic)
        assert len(waic_result.pointwise) == 1

    def test_waic_repr(self) -> None:
        """Test WAICResult string representation."""
        result = WAICResult(
            waic=123.45,
            p_waic=2.34,
            lppd=-60.56,
            se=5.67,
            pointwise=np.array([1.0, 2.0]),
        )

        repr_str = repr(result)
        assert "123.45" in repr_str
        assert "2.34" in repr_str
        assert "WAICResult" in repr_str


class TestWAICVsManual:
    """Verify WAIC formula against hand-computed example."""

    def test_waic_formula_correctness(self) -> None:
        """Verify WAIC computation matches manual formula."""
        # Create a simple case where we can verify the formula
        n_obs = 5
        y = np.zeros(n_obs)  # All zeros

        def priors(p: ParamContext) -> None:
            p("mu", dist.Normal(0, 0.1))  # Tight prior near 0

        def log_likelihood(
            params: StructuredParams, data: object
        ) -> NDArray[np.float64]:
            assert isinstance(data, np.ndarray)
            return dist.Normal(params["mu"], 1.0).log_prob(data)

        model = mb.Model(priors=priors, log_likelihood=log_likelihood)
        result = mb.sample(model, data=y, num_samples=100, num_warmup=50, seed=42)

        waic_result = mb.waic(result, model, y)

        # Manually compute from the log-likelihood matrix
        n_chains = result.num_chains
        n_samples = result.num_samples
        total_samples = n_chains * n_samples

        log_lik: NDArray[np.float64] = np.zeros((total_samples, n_obs), dtype=np.float64)
        idx = 0
        for chain_idx in range(n_chains):
            for sample_idx in range(n_samples):
                mu = float(result.samples["mu"][chain_idx, sample_idx])
                log_lik[idx, :] = dist.Normal(mu, 1.0).log_prob(y)
                idx += 1

        # lppd = sum_i log(mean_s exp(ll[s, i]))
        from minibayes.utils.numerical import log_sum_exp

        log_s = np.log(total_samples)
        lppd_i = np.array([log_sum_exp(log_lik[:, i]) - log_s for i in range(n_obs)])
        lppd_manual = np.sum(lppd_i)

        # p_waic = sum_i var_s(ll[s, i])
        p_waic_manual = np.sum(np.var(log_lik, axis=0, ddof=1))

        # WAIC = -2 * (lppd - p_waic)
        waic_manual = -2 * (lppd_manual - p_waic_manual)

        np.testing.assert_allclose(waic_result.waic, waic_manual, rtol=1e-10)
        np.testing.assert_allclose(waic_result.lppd, lppd_manual, rtol=1e-10)
        np.testing.assert_allclose(waic_result.p_waic, p_waic_manual, rtol=1e-10)
