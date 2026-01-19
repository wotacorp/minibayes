# minibayes.viz

Lightweight visualization module for MCMC diagnostics and Bayesian analysis.

## Installation

The viz module is an optional dependency:

```bash
pip install minibayes[viz]
```

This installs matplotlib (>= 3.7.0) as a dependency.

## Quick Start

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

## API Reference

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

## Composable Plots

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

## Style System

### Color Palette

```python
from minibayes.viz import PALETTE, CHAIN_COLORS

# Named colors for custom plots
PALETTE = {
    "blue": "#88B4D4",       # Pale blue
    "terracotta": "#C97B63", # Terracotta
    "sage": "#8FB996",       # Sage green
    "pink": "#E8B4BC",       # Pastel pink
    "lavender": "#A89CC8",   # Lavender
    "mustard": "#D4B86A",    # Muted gold
    "sand": "#C9BDA8",       # Warm sand
    "gray": "#8E9AAB",       # Cool slate
}

# 10 distinct colors for multi-chain plots
CHAIN_COLORS = [
    "#88B4D4", "#C97B63", "#8FB996", "#E8B4BC", "#A89CC8",
    "#D4B86A", "#6BAAAA", "#E9C46A", "#C9A0DC", "#7DAFA5",
]
```

### Style Context Manager

Apply minibayes style to custom matplotlib plots:

```python
from minibayes.viz import style, PALETTE

with style():
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=PALETTE["terracotta"])
    ax.plot(x, line, c=PALETTE["sage"])
    plt.show()
```

### Global Style

Apply style to all subsequent plots:

```python
from minibayes.viz import apply_style

apply_style()  # All plots now use minibayes style
```

## Style Specifications

- **Background**: White (#FFFFFF)
- **Grid**: Light gray (#F0F0F0), subtle
- **Text**: Charcoal (#4A4A4A)
- **Spines**: Minimal (left/bottom only, #CCCCCC)
- **DPI**: 150 for crisp display

## Summary

| Function | Purpose |
|----------|---------|
| `plot_density()` | Posterior histograms |
| `plot_samples()` | Samples over iteration |
| `plot_autocorr()` | Autocorrelation by lag |
| `plot_forest()` | Parameter box plots |
| `plot_predictive()` | Predictions with uncertainty |
| `plot_compare()` | Model comparison with WAIC |
| `summary_table()` | Formatted summary statistics |
