# Visualization

Lightweight visualization module for MCMC diagnostics and Bayesian analysis.

## Installation

The viz module is an optional dependency:

```bash
pip install minibayes[viz]
```

This installs matplotlib (>= 3.7.0) as a dependency.

## Quick start

```python
import minibayes as mb
from minibayes import viz

# After sampling...
result = mb.sample(model, num_samples=2000, num_chains=4)

# View summary statistics
print(viz.summary_table(result))

# Plot posterior distributions
viz.plot_density(result)

# Check convergence
viz.plot_samples(result)
```

## API reference

All plot functions accept flexible input types:

```python
viz.plot_density(result)                    # InferenceResult
viz.plot_density(result.samples)            # dict[str, ndarray]
viz.plot_density(result.samples["mu"])      # single ndarray
```

All functions return a `matplotlib.figure.Figure` and accept an optional `ax` parameter for composing custom layouts.

### `plot_density(data, params=None, ax=None, bins=30, show_mean=True)`

Posterior density histograms. Multi-chain samples are overlaid with transparency.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | InferenceResult, dict, ndarray | MCMC samples |
| `params` | list[str], optional | Parameters to plot (None = all) |
| `ax` | Axes, optional | Existing axes |
| `bins` | int | Histogram bins (default: 30) |
| `show_mean` | bool | Show vertical mean line (default: True) |

### `plot_samples(data, params=None, ax=None)`

Samples over iteration for convergence diagnostics. Each chain plotted in a distinct color.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | InferenceResult, dict, ndarray | MCMC samples |
| `params` | list[str], optional | Parameters to plot (None = all) |
| `ax` | Axes, optional | Existing axes |

### `plot_autocorr(data, params=None, ax=None, max_lag=50)`

Autocorrelation by lag to diagnose mixing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | InferenceResult, dict, ndarray | MCMC samples |
| `params` | list[str], optional | Parameters to plot (None = all) |
| `ax` | Axes, optional | Existing axes |
| `max_lag` | int | Maximum lag to compute (default: 50) |

### `plot_forest(data, params=None, ax=None)`

Horizontal box plots showing parameter distributions (IQR, median, whiskers). Outliers are hidden.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | InferenceResult, dict | MCMC samples |
| `params` | list[str], optional | Parameters to plot (None = all) |
| `ax` | Axes, optional | Existing axes |

### `plot_predictive(x, y_pred, x_obs=None, y_obs=None, ax=None, credible_interval=0.9)`

Posterior predictive plot with uncertainty bands.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | ndarray | X values for predictions, shape (n_points,) |
| `y_pred` | ndarray | Predicted values, shape (n_samples, n_points) |
| `x_obs` | ndarray, optional | X values for observed data |
| `y_obs` | ndarray, optional | Observed y values to overlay |
| `ax` | Axes, optional | Existing axes |
| `credible_interval` | float | CI width, 0-1 (default: 0.9) |

### `plot_compare(waic_results, ax=None)`

Model comparison plot with WAIC values and error bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `waic_results` | dict[str, WAICResult] | Mapping of model names to WAICResult objects |
| `ax` | Axes, optional | Existing axes |

Displays models ranked by WAIC (best at top) with point estimates and ±2 standard error intervals.

```python
waic1 = result1.waic(model1, data)
waic2 = result2.waic(model2, data)
fig = viz.plot_compare({"Simple": waic1, "Complex": waic2})
```

### `plot_distribution(distributions, x=None, ax=None, k_max=20)`

Plot PDF/PMF for one or more distributions. Automatically detects continuous vs discrete.

| Parameter | Type | Description |
|-----------|------|-------------|
| `distributions` | Distribution or dict | Single distribution or dict mapping labels to distributions |
| `x` | ndarray, optional | Points for continuous PDFs (auto-generated if None) |
| `ax` | Axes, optional | Existing axes |
| `k_max` | int | Max k for discrete PMF plots (default: 20) |

```python
from minibayes import dist, viz
viz.plot_distribution({"N(0,1)": dist.Normal(0, 1), "N(0,2)": dist.Normal(0, 2)})
```

### `plot_pair(data, params=None, markers=None, subsample=2000, alpha=0.15, ax=None)`

Joint posterior as 2D scatter plot. Shows correlation structure between two parameters.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | InferenceResult, dict | MCMC samples |
| `params` | list[str], optional | Exactly 2 parameters to plot (None = first 2) |
| `markers` | dict, optional | Named markers: `{"True": (x, y)}` |
| `subsample` | int | Max points to plot for performance (default: 2000) |
| `alpha` | float | Point transparency (default: 0.15) |
| `ax` | Axes, optional | Existing axes |

```python
viz.plot_pair(result, params=["mu", "sigma"], markers={"True": (0, 1)})
```

### `plot_prior_posterior(prior, posterior_samples, parameter_name="parameter", ax=None, bins=30, xlim=None)`

Compare prior distribution vs posterior samples for a single parameter.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prior` | Distribution | Prior distribution object |
| `posterior_samples` | ndarray | MCMC samples (any shape, will be flattened) |
| `parameter_name` | str | X-axis label |
| `ax` | Axes, optional | Existing axes |
| `bins` | int | Histogram bins (default: 30) |
| `xlim` | tuple, optional | X-axis limits (min, max) |

```python
from minibayes import dist, viz
viz.plot_prior_posterior(dist.Normal(0, 10), result.samples["mu"], parameter_name="mu")
```

### `plot_ppc(y_observed, posterior_predictive, prior_predictive=None, ax=None, bins=30, xlim=None)`

Posterior predictive check comparing observed data to predictions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_observed` | ndarray | Observed outcome data (1D) |
| `posterior_predictive` | ndarray | Simulated y from posterior predictive |
| `prior_predictive` | ndarray, optional | Simulated y from prior predictive (optional) |
| `ax` | Axes, optional | Existing axes |
| `bins` | int | Histogram bins (default: 30) |
| `xlim` | tuple, optional | X-axis limits (min, max) |

```python
post_pred = result.predict(predictive_fn)["y_pred"].flatten()
viz.plot_ppc(y_obs, post_pred)
```

### `summary_table(data, params=None, percentiles=None)`

Formatted ASCII table of summary statistics. Returns a string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | InferenceResult, dict | MCMC samples or summary dict |
| `params` | list[str], optional | Parameters to include (None = all) |
| `percentiles` | list[int], optional | Percentiles (default: [5, 50, 95]) |

Example output:

```
 param |   mean |    std |     5% |    50% |    95% |   ess | r_hat
-------+--------+--------+--------+--------+--------+-------+------
 alpha | 1.5284 | 0.1723 | 1.2451 | 1.5301 | 1.8062 | 152.3 | 1.002
  beta | 2.0143 | 0.0891 | 1.8672 | 2.0098 | 2.1614 | 189.7 | 1.001
 sigma | 0.4921 | 0.0412 | 0.4261 | 0.4898 | 0.5612 | 201.5 | 1.000
```

## Composable plots

All functions support custom axes for multi-panel figures:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

viz.plot_density(result, params=["mu"], ax=axes[0, 0])
viz.plot_samples(result, params=["mu"], ax=axes[0, 1])
viz.plot_density(result, params=["sigma"], ax=axes[1, 0])
viz.plot_samples(result, params=["sigma"], ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

## Style system

Apply minibayes styling to custom plots:

```python
from minibayes.viz import style, apply_style, PALETTE, CHAIN_COLORS

# Context manager for temporary styling
with style():
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=PALETTE["terracotta"])

# Or apply globally
apply_style()
```

**Colors**: `PALETTE` provides named colors (blue, terracotta, sage, pink, lavender, mustard, sand, gray). `CHAIN_COLORS` provides 10 distinct colors for multi-chain plots.

**Style specs**: White background, light gray grid (#F0F0F0), charcoal text (#4A4A4A), minimal spines, 150 DPI.

## Summary

| Function | Purpose |
|----------|---------|
| `plot_density()` | Posterior histograms |
| `plot_samples()` | Trace plots (samples over iteration) |
| `plot_autocorr()` | Autocorrelation by lag |
| `plot_forest()` | Parameter box plots |
| `plot_predictive()` | Predictions with uncertainty bands |
| `plot_compare()` | Model comparison with WAIC |
| `plot_distribution()` | Prior/likelihood PDF/PMF |
| `plot_pair()` | Joint posterior scatter |
| `plot_prior_posterior()` | Prior vs posterior comparison |
| `plot_ppc()` | Posterior predictive check |
| `summary_table()` | Formatted summary statistics |
