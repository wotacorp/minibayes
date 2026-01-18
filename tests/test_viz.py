"""Tests for visualization module."""

import numpy as np
import pytest

from minibayes.results import InferenceResult


@pytest.fixture
def mock_result():
    """Create a mock InferenceResult for testing."""
    rng = np.random.default_rng(42)
    n_chains = 2
    n_samples = 100

    samples = {
        "mu": rng.normal(2.0, 0.5, (n_chains, n_samples)),
        "sigma": rng.exponential(1.0, (n_chains, n_samples)),
    }

    return InferenceResult(
        samples=samples,
        samples_unconstrained=samples.copy(),
        acceptance_rate=np.array([0.25, 0.28]),
        num_samples=n_samples,
        num_warmup=100,
        num_chains=n_chains,
        sampler="mh",
        elapsed_time=1.5,
    )


@pytest.fixture
def mock_samples():
    """Create mock samples dict for testing."""
    rng = np.random.default_rng(42)
    return {
        "alpha": rng.normal(1.0, 0.2, (2, 100)),
        "beta": rng.normal(2.0, 0.3, (2, 100)),
    }


class TestVizImport:
    """Test viz module imports."""

    def test_import_viz(self):
        """Test that viz module can be imported."""
        from minibayes import viz

        assert hasattr(viz, "plot_density")
        assert hasattr(viz, "plot_samples")
        assert hasattr(viz, "plot_forest")
        assert hasattr(viz, "plot_predictive")
        assert hasattr(viz, "plot_autocorr")
        assert hasattr(viz, "plot_distribution")
        assert hasattr(viz, "summary_table")

    def test_import_style(self):
        """Test that style components can be imported."""
        from minibayes.viz import CHAIN_COLORS, PALETTE, style

        assert isinstance(PALETTE, dict)
        assert "blue" in PALETTE
        assert isinstance(CHAIN_COLORS, list)
        assert len(CHAIN_COLORS) == 10
        assert callable(style)


class TestPlotDensity:
    """Test plot_density function."""

    def test_with_inference_result(self, mock_result):
        """Test plot_density with InferenceResult."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        from minibayes.viz import plot_density

        fig = plot_density(mock_result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_samples_dict(self, mock_samples):
        """Test plot_density with samples dict."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_density

        fig = plot_density(mock_samples)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_single_param(self, mock_result):
        """Test plot_density with single parameter."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_density

        fig = plot_density(mock_result, params=["mu"])
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_derived_param(self, mock_result):
        """Test plot_density with derived parameter."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_density

        # Add derived parameter
        derived = mock_result.samples["mu"] + mock_result.samples["sigma"]
        mock_result.add_derived("mu_plus_sigma", derived)

        # Should be able to plot derived param
        fig = plot_density(mock_result, params=["mu_plus_sigma"])
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotSamples:
    """Test plot_samples function."""

    def test_with_inference_result(self, mock_result):
        """Test plot_samples with InferenceResult."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_samples

        fig = plot_samples(mock_result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotForest:
    """Test plot_forest function."""

    def test_with_inference_result(self, mock_result):
        """Test plot_forest with InferenceResult."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_forest

        fig = plot_forest(mock_result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotPredictive:
    """Test plot_predictive function."""

    def test_basic(self):
        """Test plot_predictive with basic data."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_predictive

        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y_pred = rng.normal(x, 1.0, (100, 50))

        fig = plot_predictive(x, y_pred)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_observed(self):
        """Test plot_predictive with observed data."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_predictive

        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y_pred = rng.normal(x, 1.0, (100, 50))
        y_obs = x + rng.normal(0, 1.0, 50)

        fig = plot_predictive(x, y_pred, y_obs=y_obs)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_separate_x_obs(self):
        """Test plot_predictive with separate x_obs coordinates."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_predictive

        rng = np.random.default_rng(42)
        # Prediction on fine grid
        x_pred = np.linspace(0, 10, 100)
        y_pred = rng.normal(x_pred, 1.0, (50, 100))
        # Observed data on coarse grid
        x_obs = np.linspace(0, 10, 20)
        y_obs = x_obs + rng.normal(0, 1.0, 20)

        fig = plot_predictive(x_pred, y_pred, x_obs=x_obs, y_obs=y_obs)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotAutocorr:
    """Test plot_autocorr function."""

    def test_with_inference_result(self, mock_result):
        """Test plot_autocorr with InferenceResult."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_autocorr

        fig = plot_autocorr(mock_result)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestSummaryTable:
    """Test summary_table function."""

    def test_with_inference_result(self, mock_result):
        """Test summary_table with InferenceResult."""
        from minibayes.viz import summary_table

        table = summary_table(mock_result)
        assert isinstance(table, str)
        assert "mu" in table
        assert "sigma" in table
        assert "mean" in table
        assert "ess" in table

    def test_with_samples_dict(self, mock_samples):
        """Test summary_table with samples dict."""
        from minibayes.viz import summary_table

        table = summary_table(mock_samples)
        assert isinstance(table, str)
        assert "alpha" in table
        assert "beta" in table


class TestStyle:
    """Test style context manager."""

    def test_style_context(self):
        """Test style context manager."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from minibayes.viz import style

        original_dpi = plt.rcParams["figure.dpi"]

        with style():
            # Style should be applied
            assert plt.rcParams["figure.dpi"] == 150

        # Style should be restored
        assert plt.rcParams["figure.dpi"] == original_dpi


class TestPlotDistribution:
    """Test plot_distribution function."""

    def test_single_continuous(self):
        """Test plot_distribution with single continuous distribution."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes import dist
        from minibayes.viz import plot_distribution

        fig = plot_distribution(dist.Normal(0, 1))
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multiple_continuous(self):
        """Test plot_distribution with multiple continuous distributions."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes import dist
        from minibayes.viz import plot_distribution

        distributions = {
            "Normal(0,1)": dist.Normal(0, 1),
            "Normal(0,2)": dist.Normal(0, 2),
        }
        fig = plot_distribution(distributions)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_discrete_bernoulli(self):
        """Test plot_distribution with Bernoulli distribution."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes import dist
        from minibayes.viz import plot_distribution

        fig = plot_distribution(dist.Bernoulli(0.7))
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_discrete_poisson(self):
        """Test plot_distribution with Poisson distributions."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes import dist
        from minibayes.viz import plot_distribution

        distributions = {
            "Poisson(1)": dist.Poisson(1),
            "Poisson(5)": dist.Poisson(5),
        }
        fig = plot_distribution(distributions, k_max=15)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_positive_support(self):
        """Test plot_distribution with POSITIVE support distributions."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes import dist
        from minibayes.viz import plot_distribution

        distributions = {
            "Gamma(2,1)": dist.Gamma(2, 1),
            "Exponential(1)": dist.Exponential(1),
        }
        fig = plot_distribution(distributions)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_unit_support(self):
        """Test plot_distribution with UNIT support distribution."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes import dist
        from minibayes.viz import plot_distribution

        fig = plot_distribution(dist.Beta(2, 5))
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_bounded_support(self):
        """Test plot_distribution with BOUNDED support distribution."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes import dist
        from minibayes.viz import plot_distribution

        fig = plot_distribution(dist.Uniform(2, 5))
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_x(self):
        """Test plot_distribution with custom x range."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes import dist
        from minibayes.viz import plot_distribution

        x = np.linspace(-10, 10, 500)
        fig = plot_distribution(dist.Normal(0, 3), x=x)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_ax(self):
        """Test plot_distribution with custom axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from minibayes import dist
        from minibayes.viz import plot_distribution

        fig, ax = plt.subplots()
        result_fig = plot_distribution(dist.Normal(0, 1), ax=ax)
        assert result_fig is fig
        plt.close(fig)


class TestPlotPair:
    """Test plot_pair function."""

    def test_basic(self, mock_samples) -> None:
        """Test basic 2-param scatter."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from minibayes.viz import plot_pair

        fig = plot_pair(mock_samples)
        assert fig is not None
        plt.close(fig)

    def test_with_markers(self, mock_samples) -> None:
        """Test scatter with reference markers."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from minibayes.viz import plot_pair

        fig = plot_pair(mock_samples, markers={"True": (1.0, 2.0)})
        assert fig is not None
        plt.close(fig)

    def test_with_inference_result(self, mock_result) -> None:
        """Test plot_pair with InferenceResult."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from minibayes.viz import plot_pair

        fig = plot_pair(mock_result, params=["mu", "sigma"])
        assert fig is not None
        plt.close(fig)

    def test_subsample(self) -> None:
        """Test that large samples get subsampled."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from minibayes.viz import plot_pair

        rng = np.random.default_rng(42)
        large_samples = {
            "a": rng.normal(0, 1, (2, 50000)),
            "b": rng.normal(0, 1, (2, 50000)),
        }
        fig = plot_pair(large_samples, subsample=1000)
        assert fig is not None
        plt.close(fig)

    def test_requires_two_params(self) -> None:
        """Test that plot_pair requires at least 2 parameters."""
        import matplotlib

        matplotlib.use("Agg")
        from minibayes.viz import plot_pair

        single_param = {"a": np.random.randn(2, 100)}
        with pytest.raises(ValueError, match="requires at least 2 parameters"):
            plot_pair(single_param)


class TestPlotCompare:
    """Tests for plot_compare function."""

    def test_plot_compare_returns_figure(self) -> None:
        """plot_compare returns matplotlib Figure."""
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        from minibayes.comparison import WAICResult
        from minibayes.viz import plot_compare

        waic_results = {
            "model_1": WAICResult(
                waic=100.0, p_waic=2.0, lppd=-48.0, se=5.0, pointwise=np.array([50.0, 50.0])
            ),
            "model_2": WAICResult(
                waic=110.0, p_waic=3.0, lppd=-51.5, se=6.0, pointwise=np.array([55.0, 55.0])
            ),
        }

        fig = plot_compare(waic_results)
        assert fig is not None
        plt.close(fig)

    def test_plot_compare_ordering(self) -> None:
        """Models are ordered by WAIC (best at top)."""
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        from minibayes.comparison import WAICResult
        from minibayes.viz import plot_compare

        # model_b has lower (better) WAIC
        waic_results = {
            "model_a": WAICResult(
                waic=120.0, p_waic=3.0, lppd=-57.0, se=5.0, pointwise=np.array([60.0, 60.0])
            ),
            "model_b": WAICResult(
                waic=100.0, p_waic=2.0, lppd=-48.0, se=4.0, pointwise=np.array([50.0, 50.0])
            ),
        }

        fig = plot_compare(waic_results)
        ax = fig.axes[0]
        # Get y-tick labels to verify ordering
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # model_b should be first (at top, which is position 0)
        assert labels[0] == "model_b"
        assert labels[1] == "model_a"
        plt.close(fig)

    def test_plot_compare_single_model(self) -> None:
        """Works with single model."""
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        from minibayes.comparison import WAICResult
        from minibayes.viz import plot_compare

        waic_results = {
            "only_model": WAICResult(
                waic=100.0, p_waic=2.0, lppd=-48.0, se=5.0, pointwise=np.array([50.0, 50.0])
            ),
        }

        fig = plot_compare(waic_results)
        assert fig is not None
        plt.close(fig)
