# minibayes

A minimal, lightweight Bayesian inference library for Python.

## Why minibayes?

Modern probabilistic programming libraries (PyMC, NumPyro, Stan) are powerful but heavyweight. They require complex dependencies, use "magic" context managers, and can be overkill for simple models.

minibayes provides:
- **Explicit over magic**: No context-dependent behavior, no global state
- **Minimal dependencies**: Python 3.11+ and NumPy only
- **Inspectable**: See exactly what transforms are applied, what log_prob computes
- **Production-ready**: Deployable in containers on edge devices

## Quick Start

```python
import numpy as np
import minibayes as mb
from minibayes import dist

# Define model
model = mb.Model(
    priors={"mu": dist.Normal(0, 10), "sigma": dist.HalfNormal(5)},
    likelihood=lambda p, d: float(np.sum(dist.Normal(p["mu"], p["sigma"]).log_prob(d))),
)

# Run inference
result = mb.sample(model, data=y, num_samples=2000, num_chains=2, seed=42)

# View results
print(result.summary())
result.save("posterior.npz")
```

## Model Class

The structured way to define Bayesian models. Priors and likelihood are specified separately.

```python
model = mb.Model(
    priors={"mu": dist.Normal(0, 10), "sigma": dist.HalfNormal(5)},
    likelihood=lambda params, data: dist.Normal(params["mu"], params["sigma"]).log_prob(data).sum(),
)

# Inspect (no magic - everything is explicit)
model.param_names                       # ['mu', 'sigma']
model.transforms                        # {'mu': Identity, 'sigma': Log}
model.sample_prior()                    # {'mu': 2.3, 'sigma': 1.8}
model.log_prob({"mu": 0, "sigma": 1}, data)  # float
```

### Core Methods

| Method | Description |
|--------|-------------|
| `param_names` | List of parameter names in fixed order |
| `transforms` | Dict of transforms derived from distribution support |
| `sample_prior(rng)` | Draw one sample from the joint prior |
| `prior_means()` | Return mean of each prior distribution |
| `log_prior(params)` | Sum of log_prob for each prior |
| `log_likelihood(params, data)` | User-provided likelihood function |
| `log_prob(params, data)` | log_prior + log_likelihood (unnormalized posterior) |
| `to_unconstrained(params)` | Transform constrained → unconstrained space |
| `to_constrained(unconstrained)` | Transform unconstrained → constrained space |
| `log_prob_unconstrained(unconstrained, data)` | log_prob with Jacobian correction |
| `validate_params(params)` | Check names and values are valid |

### Automatic Transforms

Transforms are derived from distribution support:

| Support | Domain | Transform | Distributions |
|---------|--------|-----------|---------------|
| REAL | (-∞, +∞) | IdentityTransform | Normal |
| POSITIVE | (0, +∞) | LogTransform | HalfNormal, Exponential, Gamma |
| UNIT | (0, 1) | LogitTransform | Beta |
| BOUNDED | (a, b) | AffineTransform | Uniform |

MCMC samplers work in unconstrained space. The `log_prob_unconstrained()` method includes the Jacobian correction automatically.

## The sample() Function

The main entry point for inference.

```python
result = mb.sample(
    model,                    # Model instance
    data=y,                   # Observed data passed to likelihood
    initial=None,             # Initial params (optional, uses prior means if None)
    num_samples=1000,         # Samples to draw (post-warmup)
    num_warmup=500,           # Warmup samples (discarded, used for adaptation)
    num_chains=1,             # Number of independent chains
    sampler="adaptive_mh",    # "mh" or "adaptive_mh"
    sampler_kwargs=None,      # Extra args for sampler (e.g., proposal_scale)
    seed=None,                # Random seed for reproducibility
)
```

### What Happens Inside

1. **Validate inputs**: Check sampler name is valid ("mh" or "adaptive_mh")
2. **Set up RNG**: Create random generator from seed, spawn child RNGs for each chain
3. **For each chain**:
   - Get initial state (use prior means if not provided, transform to unconstrained)
   - Create fresh sampler instance
   - **Warmup phase**: Run `num_warmup` steps with `sampler.warmup_step()` (adapts proposal)
   - **Finalize**: Call `sampler.post_warmup()` (freezes adaptation, frees memory)
   - **Sampling phase**: Run `num_samples` steps with `sampler.step()`, store samples
4. **Transform samples**: Convert unconstrained samples back to constrained space
5. **Return InferenceResult**: Package samples, acceptance rates, timing

### Default Initialization

When `initial=None`, the sampler uses prior means as the starting point:

| Distribution | Mean |
|--------------|------|
| Normal(μ, σ) | μ |
| HalfNormal(σ) | σ√(2/π) |
| Exponential(λ) | 1/λ |
| Gamma(α, β) | α/β |
| Beta(α, β) | α/(α+β) |
| Uniform(a, b) | (a+b)/2 |

This provides deterministic, robust initialization without needing manual tuning. If prior means yield invalid log_prob (rare), sampling falls back to random prior draws.

### Sampler Options

| Sampler | Description | When to Use |
|---------|-------------|-------------|
| `"adaptive_mh"` | Adaptive Metropolis with covariance tuning | Default, learns correlations |
| `"mh"` | Random-walk Metropolis-Hastings | Simple models, known proposal scale |

```python
# Custom proposal scale for MH
result = mb.sample(model, data, sampler="mh", sampler_kwargs={"proposal_scale": 0.5})

# Per-parameter scales
result = mb.sample(model, data, sampler="mh",
    sampler_kwargs={"proposal_scale": {"mu": 0.5, "sigma": 0.1}})
```

## Inference Results

After sampling, results are stored in an `InferenceResult` dataclass:

```python
result.samples["mu"]       # NDArray: (num_samples,) or (num_chains, num_samples)
result.samples["sigma"]
result.acceptance_rate     # float or NDArray per chain
result.elapsed_time        # Sampling time in seconds

# Summary statistics
summary = result.summary()
# {'mu': {'mean': 2.01, 'std': 0.05, '5%': 1.93, '50%': 2.01, '95%': 2.09, 'ess': 890.2, 'r_hat': 1.001}, ...}
```

## Diagnostics

| Diagnostic | Good Value | Concern |
|------------|------------|---------|
| ESS | > 400 | Low ESS indicates high autocorrelation |
| R-hat | < 1.01 | R-hat > 1.01 suggests chains haven't converged |

```python
from minibayes.diagnostics import effective_sample_size, r_hat

ess = effective_sample_size(result.samples["mu"])  # Single chain
rhat = r_hat(result.samples["mu"])  # Multi-chain: shape (num_chains, num_samples)
```

## Saving and Loading

```python
result.save("posterior.npz")           # NPZ format (default)
result.save("posterior.json", format="json")  # JSON format

loaded = mb.InferenceResult.load("posterior.npz")
```

## Design Principles

1. **No magic**: Every operation is an explicit method call
2. **Dict ordering preserved**: Parameter order is fixed at initialization
3. **Priors are independent**: No hierarchical structure (v1.0 limitation)
4. **Inspectable**: Users can examine transforms, priors, computed probabilities

## References

- [Gelman et al. BDA3 - Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/)
- [Andrieu & Thoms (2008) - A Tutorial on Adaptive MCMC](https://link.springer.com/article/10.1007/s11222-008-9110-y)
- [Stan User's Guide: Changes of Variables](https://mc-stan.org/docs/stan-users-guide/changes-of-variables.html)
