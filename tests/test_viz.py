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
