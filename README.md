# minibayes

Lightweight Bayesian inference with (only) NumPy.

## What is it?

Bayesian modelling and MCMC in pure Python, with NumPy as the only dependency.

minibayes runs Bayesian inference without the heavyweight dependencies. No PyTensor, no JAX, no compilation step. Install it, import it, fit your model. The package is about 5 MB.

It covers the most common use cases in Bayesian analysis: regression, A/B tests, hierarchical models with partial pooling. minibayes is designed for situations where you need something lightweight that deploys easily and integrates cleanly into existing systems.

## Installation

Core package (NumPy only):
```bash
pip install minibayes
```

With visualization (adds matplotlib):
```bash
pip install minibayes[viz]
```

Everything (viz + dev tools):
```bash
pip install minibayes[all]
```

## Quick Start

A robust linear regression that handles outliers. The Student-t likelihood downweights extreme observations instead of letting them dominate the fit.

```python
import numpy as np
import minibayes as mb
from minibayes import Model, dist

# Data with an outlier at index 7
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
y = np.array([1.5, 2.4, 3.6, 4.3, 5.7, 6.2, 7.1, 12.0, 9.5, 10.8])

def priors(p):
    p("alpha", dist.Normal(0, 10))      # intercept
    p("beta", dist.Normal(0, 10))       # slope
    p("sigma", dist.HalfNormal(5))      # noise scale

def log_likelihood(params, data):
    x, y = data
    mu = params["alpha"] + params["beta"] * x
    # Student-t with df=4 has heavier tails than Normal
    return dist.StudentT(df=4, loc=mu, scale=params["sigma"]).log_prob(y)

model = Model(priors=priors, log_likelihood=log_likelihood)
result = mb.sample(model, data=(x, y), num_samples=2000, seed=42)
print(result.summary())
```

The `priors` function defines your parameter distributions. The `log_likelihood` function scores how well parameters explain the data. `Model` combines them, and `sample()` runs MCMC.

## Why minibayes?

There are excellent Bayesian libraries available. PyMC is mature and feature-rich. NumPyro offers high performance with JAX. minibayes fills a different niche.

**Simplicity.** The API is minimal: write a function for your priors, a function for your likelihood, and call `sample()`. No graph compilation, no effect handlers, no special syntax to learn.

**Deployability.** The core has no dependencies beyond NumPy. This makes it straightforward to deploy in Docker containers, on embedded systems, or as part of larger applications without dependency conflicts.

**Transparency.** The internals are accessible. `model.transforms["sigma"]` shows you the parameter transform. `model.log_prob(params, data)` returns the log posterior at any point. No hidden state.

**When to consider alternatives:**
- For complex hierarchical models with hundreds of parameters, PyMC provides more advanced samplers
- For GPU acceleration or very large datasets, NumPyro is better suited
- For time series and state space models, PyMC or specialized libraries offer dedicated tools

## Features

**Distributions (16)**

Normal, HalfNormal, StudentT, Cauchy, Laplace, Exponential, Gamma, LogNormal, InverseGamma, Beta, Uniform, TruncatedNormal, Bernoulli, Poisson, MultivariateNormal, LKJCholesky

**Samplers**
- Metropolis-Hastings — manual tuning via `proposal_scale`
- Adaptive Metropolis — learns proposal covariance during warmup
- Ensemble sampler — emcee-style, handles multimodal posteriors

**Model features**
- Automatic transforms derived from distribution support (Log for positive, Logit for unit interval, etc.)
- Hierarchical models via `p()` API with `size=` for vector parameters
- Jacobian corrections handled automatically

**Diagnostics & output**
- ESS (effective sample size), R-hat convergence diagnostic
- WAIC for model comparison
- Save/load results in NPZ or JSON format
- Memory safety with configurable `max_samples` and `max_memory_mb` limits

## More Examples

**Hierarchical model with partial pooling:**

```python
def priors(p):
    mu = p("mu", dist.Normal(0, 5))           # population mean
    tau = p("tau", dist.HalfNormal(5))        # population sd
    theta = p("theta", dist.Normal(mu, tau), size=8)  # group-level effects
```

The `size=8` creates a vector parameter. Each `theta[i]` is drawn from `Normal(mu, tau)`, sharing information across groups.

**Posterior predictive sampling:**

```python
x_new = np.array([5.0, 6.0, 7.0])

def predict(params, rng):
    mu = params["alpha"] + params["beta"] * x_new
    return {"y": dist.Normal(mu, params["sigma"]).sample(size=len(x_new), rng=rng)}

predictions = result.predict(predict, num_samples=500)
# predictions["y"] has shape (500, 3)
```

**Model comparison with WAIC:**

```python
waic_result = result.waic(model, data)
print(f"WAIC: {waic_result.waic:.1f} (SE: {waic_result.se:.1f})")
```

Lower WAIC indicates better out-of-sample predictive performance.

## Visualization

Install with `pip install minibayes[viz]` to get plotting support. Requires matplotlib.

```python
from minibayes import viz

viz.plot_density(result)       # posterior distributions
viz.plot_samples(result)       # trace plots
viz.plot_forest(result)        # parameter estimates with credible intervals
viz.plot_autocorr(result)      # mixing diagnostics
viz.plot_predictive(x, preds)  # predictions with uncertainty bands
```

<!-- TODO: Add example plot images -->
<!-- ![Density plot](docs/images/density.png) -->
<!-- ![Forest plot](docs/images/forest.png) -->
<!-- ![Predictive plot](docs/images/predictive.png) -->

## Notebooks

See the `notebooks/` folder for worked examples:

| Notebook | Description |
|----------|-------------|
| `01_mh_examples` | Basic Metropolis-Hastings, Normal-Normal conjugate model |
| `02_adaptive_mh_examples` | Adaptive sampler, robust regression with Student-t |
| `03_viz_showcase` | All visualization functions |
| `04_distributions_gallery` | Visual reference for all 16 distributions |
| `05_hierarchical_models` | Eight Schools problem, partial pooling |
| `06_multivariate_normal` | Covariance estimation, LKJCholesky prior |
| `07_ensemble_sampler` | Affine-invariant ensemble (emcee-style) |
| `08_model_comparison` | WAIC for comparing polynomial models |

## Documentation

Detailed documentation is available in the source tree:

| Document | Description |
|----------|-------------|
| [API Reference](src/minibayes/README.md) | Model class, `sample()` function, results, diagnostics, WAIC |
| [Distributions](src/minibayes/distributions/README.md) | All 16 distributions with parameters, formulas, and use cases |
| [Samplers](src/minibayes/samplers/README.md) | MCMC algorithms: MH, Adaptive MH, Ensemble sampler |
| [Transforms](src/minibayes/transforms/README.md) | Parameter transforms (Log, Logit, Affine) and Jacobian corrections |
| [Visualization](src/minibayes/viz/README.md) | Plotting functions, style system, color palette |
| [Utilities](src/minibayes/utils/README.md) | Numerical helpers: `log_sum_exp`, RNG handling |

## Development

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/wotacorp/minibayes.git
cd minibayes
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run checks:
```bash
uv run pytest              # tests
uv run mypy src            # type checking (strict mode)
uv run ruff check src      # linting
uv run ruff format src     # formatting
```

## Maintainer

[@theoradusz](https://github.com/theoradusz)
