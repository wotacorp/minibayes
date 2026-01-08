# minibayes

A minimal, lightweight Bayesian inference library for Python.

## Why minibayes?

Modern probabilistic programming libraries (PyMC, NumPyro, Stan) are powerful but heavyweight. They require complex dependencies, use "magic" context managers, and can be overkill for simple models.

minibayes provides:
- **Explicit over magic**: No context-dependent behavior, no global state
- **Minimal dependencies**: Python 3.11+ and NumPy only
- **Inspectable**: See exactly what transforms are applied, what log_prob computes
- **Production-ready**: Deployable in containers on edge devices

## The Two APIs

minibayes provides two ways to specify models:

| API | Use Case | Transforms | Verbosity |
|-----|----------|------------|-----------|
| **Model class** | Standard models, automatic transforms | Automatic | Low |
| **Direct log_prob** | Full control, complex models | Manual | High |

Both APIs feed into the same samplers.

## Model Class

The structured way to define Bayesian models. Priors and likelihood are specified separately.

```python
import minibayes as mb
from minibayes import dist

# Define model
model = mb.Model(
    priors={
        "mu": dist.Normal(0, 10),
        "sigma": dist.HalfNormal(5),
    },
    likelihood=lambda params, data: dist.Normal(params["mu"], params["sigma"]).log_prob(data).sum(),
)

# Inspect model (no magic - everything is explicit)
model.param_names                    # ['mu', 'sigma']
model.transforms                     # {'mu': Identity, 'sigma': Log}
model.sample_prior()                 # {'mu': 2.3, 'sigma': 1.8}
model.log_prior({"mu": 0, "sigma": 1})  # float
```

## Core Methods

| Method | Description |
|--------|-------------|
| `param_names` | List of parameter names in fixed order |
| `transforms` | Dict of transforms derived from distribution support |
| `sample_prior(rng)` | Draw one sample from the joint prior |
| `log_prior(params)` | Sum of log_prob for each prior |
| `log_likelihood(params, data)` | User-provided likelihood function |
| `log_prob(params, data)` | log_prior + log_likelihood (unnormalized posterior) |
| `to_unconstrained(params)` | Transform constrained → unconstrained space |
| `to_constrained(unconstrained)` | Transform unconstrained → constrained space |
| `log_prob_unconstrained(unconstrained, data)` | log_prob with Jacobian correction (for samplers) |
| `validate_params(params)` | Check names and values are valid |

## Transform Handling

Transforms are derived automatically from distribution support:

| Support | Domain | Transform | Distributions |
|---------|--------|-----------|---------------|
| REAL | (-∞, +∞) | IdentityTransform | Normal |
| POSITIVE | (0, +∞) | LogTransform | HalfNormal, Exponential, Gamma |
| UNIT | (0, 1) | LogitTransform | Beta |
| BOUNDED | (a, b) | AffineTransform | Uniform |

MCMC samplers work in unconstrained space. The Model class handles the transformation automatically.

## The Jacobian Correction

When transforming from constrained θ to unconstrained φ, probability densities must be adjusted:

```
log p(φ | data) = log p(θ | data) + Σ log|dθ_k/dφ_k|
```

The `log_prob_unconstrained()` method includes this Jacobian correction automatically. This is what samplers call internally.

### Example: Log Transform

For a positive parameter σ with log transform (φ = log(θ)):
- Inverse: θ = exp(φ)
- Jacobian: dθ/dφ = exp(φ) = θ
- Log Jacobian: log(θ)

```python
# Constrained space
sigma = 2.0
log_prob_constrained = model.log_prob({"sigma": sigma}, data)

# Unconstrained space (includes Jacobian)
phi = np.log(sigma)  # = 0.693
log_prob_unconstrained = model.log_prob_unconstrained({"sigma": phi}, data)

# log_prob_unconstrained = log_prob_constrained + log(sigma)
```

## Complete Example: Bayesian Linear Regression

```python
import numpy as np
import minibayes as mb
from minibayes import dist

# Generate data
np.random.seed(42)
X = np.random.randn(100)
true_mu, true_sigma = 2.0, 0.5
y = true_mu + np.random.normal(0, true_sigma, size=100)

# Define likelihood
def likelihood(params, data):
    y = data
    mu, sigma = params["mu"], params["sigma"]
    return float(np.sum(dist.Normal(mu, sigma).log_prob(y)))

# Create model
model = mb.Model(
    priors={
        "mu": dist.Normal(0, 10),
        "sigma": dist.HalfNormal(5),
    },
    likelihood=likelihood,
)

# Inspect
print(model.param_names)      # ['mu', 'sigma']
print(model.transforms)       # {'mu': IdentityTransform, 'sigma': LogTransform}

# Sample from prior
prior_sample = model.sample_prior()
print(prior_sample)           # {'mu': 3.2, 'sigma': 2.1}

# Evaluate log probability
params = {"mu": 2.0, "sigma": 0.5}
print(model.log_prior(params))           # -4.35...
print(model.log_likelihood(params, y))   # -73.2...
print(model.log_prob(params, y))         # -77.5...

# Transform to unconstrained space
unconstrained = model.to_unconstrained(params)
print(unconstrained)  # {'mu': 2.0, 'sigma': -0.693...}

# Validate parameters
model.validate_params(params)  # True
# model.validate_params({"mu": 0, "sigma": -1})  # Raises ModelSpecError
```

## Direct log_prob API

For full control, you can bypass the Model class and provide a raw log_prob function:

```python
def log_prob(params, data):
    # Unpack (working in unconstrained space)
    mu = params["mu"]
    log_sigma = params["log_sigma"]  # Unconstrained!
    sigma = np.exp(log_sigma)

    # Priors
    lp = dist.Normal(0, 10).log_prob(mu)
    lp += dist.HalfNormal(5).log_prob(sigma)
    lp += log_sigma  # Jacobian correction for log transform

    # Likelihood
    lp += dist.Normal(mu, sigma).log_prob(data).sum()

    return float(lp)

# Pass to sampler directly
# result = mb.sample(model=log_prob, data=y, initial={"mu": 0, "log_sigma": 0})
```

## Inference Results

After sampling, results are stored in an `InferenceResult` dataclass:

```python
# result = mb.sample(model, data, ...)

# Access samples (constrained space)
result.samples["mu"]       # NDArray of shape (num_samples,) or (num_chains, num_samples)
result.samples["sigma"]

# Metadata
result.num_samples         # Number of samples per chain
result.num_chains          # Number of chains
result.acceptance_rate     # float or NDArray per chain
result.elapsed_time        # Sampling time in seconds

# Summary statistics
summary = result.summary()
# {'mu': {'mean': 2.01, 'std': 0.05, '5%': 1.93, '50%': 2.01, '95%': 2.09, 'ess': 890.2},
#  'sigma': {'mean': 0.51, 'std': 0.04, '5%': 0.45, '50%': 0.51, '95%': 0.58, 'ess': 756.1, 'r_hat': 1.001}}

# Custom percentiles
result.summary(percentiles=[2.5, 50, 97.5])

# Filter parameters
result.summary(params=["mu"])
```

## Diagnostics

Convergence diagnostics are available via the `diagnostics` module:

```python
from minibayes.diagnostics import effective_sample_size, r_hat, summary

# Effective Sample Size (ESS) - accounts for autocorrelation
samples = result.samples["mu"]  # 1D array
ess = effective_sample_size(samples)  # float, typically < len(samples)

# Gelman-Rubin R-hat (requires multiple chains)
chains = result.samples["mu"]  # shape (num_chains, num_samples)
rhat = r_hat(chains)  # ~1.0 for converged chains, >1.01 indicates issues

# Summary for all parameters
stats = summary(result.samples)
# Returns dict with mean, std, percentiles, ess, and r_hat (if multi-chain)
```

### Interpreting Diagnostics

| Diagnostic | Good Value | Concern |
|------------|------------|---------|
| ESS | > 400 | Low ESS indicates high autocorrelation |
| R-hat | < 1.01 | R-hat > 1.01 suggests chains haven't converged |

## Saving and Loading Results

Results can be saved in NPZ (NumPy compressed) or JSON format:

```python
# Save to file
result.save("posterior.npz")           # NPZ format (default, efficient)
result.save("posterior.json", format="json")  # JSON format (portable)

# Load from file
loaded = mb.InferenceResult.load("posterior.npz")
loaded = mb.InferenceResult.load("posterior.json")

# Convert to dict (for JSON serialization)
data = result.to_dict()
```

## Design Principles

1. **No magic**: Every operation is an explicit method call
2. **Dict ordering preserved**: Parameter order is fixed at initialization
3. **Priors are independent**: No hierarchical structure (v1.0 limitation)
4. **Inspectable**: Users can examine transforms, priors, computed probabilities

## References

- [Stan User's Guide: Changes of Variables](https://mc-stan.org/docs/stan-users-guide/changes-of-variables.html)
- [PyMC Transforms](https://www.pymc.io/projects/docs/en/latest/api/distributions/transforms.html)
- [NumPyro Transforms](https://num.pyro.ai/en/stable/distributions.html#transforms)
- [Gelman et al. BDA3 - Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/)
- [Andrieu & Thoms (2008) - A Tutorial on Adaptive MCMC](https://link.springer.com/article/10.1007/s11222-008-9110-y)
