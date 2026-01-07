# Probability Distributions

Probability distributions for specifying priors and likelihoods in Bayesian inference.

## Why Distributions?

In Bayesian inference, we need probability distributions to:
- Define prior beliefs about parameters before observing data
- Compute likelihoods of data given parameters
- Sample from posteriors via MCMC

Each distribution provides two key operations:
- `log_prob(x)`: Evaluate log probability density at a point
- `sample(size, rng)`: Draw random samples

## Available Distributions

| Distribution | Support | Parameters | Use Case |
|--------------|---------|------------|----------|
| **Normal** | (-∞, +∞) | loc (μ), scale (σ) | Unbounded continuous parameters |
| **HalfNormal** | (0, +∞) | scale (σ) | Scale parameters, positive values |
| **Exponential** | (0, +∞) | rate (λ) | Waiting times, sparse priors |
| **Gamma** | (0, +∞) | shape (α), rate (β) | Positive parameters, precision |
| **Beta** | (0, 1) | alpha (α), beta (β) | Probabilities, proportions |
| **Uniform** | [low, high] | low, high | Bounded parameters, non-informative |

## Log-Probability Formulas

| Distribution | log p(x) |
|--------------|----------|
| **Normal(μ, σ)** | -½log(2π) - log(σ) - (x-μ)²/(2σ²) |
| **HalfNormal(σ)** | log(√(2/π)) - log(σ) - x²/(2σ²) for x > 0 |
| **Exponential(λ)** | log(λ) - λx for x > 0 |
| **Gamma(α, β)** | α·log(β) + (α-1)·log(x) - βx - log(Γ(α)) for x > 0 |
| **Beta(α, β)** | (α-1)·log(x) + (β-1)·log(1-x) - log(B(α,β)) for x ∈ (0,1) |
| **Uniform(a, b)** | -log(b-a) for x ∈ [a,b] |

All distributions return -∞ outside their support.

## Support and Transforms

Distributions are automatically paired with transforms based on their support:

| Support | Domain | Transform | Distributions |
|---------|--------|-----------|---------------|
| REAL | (-∞, +∞) | IdentityTransform | Normal |
| POSITIVE | (0, +∞) | LogTransform | HalfNormal, Exponential, Gamma |
| UNIT | (0, 1) | LogitTransform | Beta |
| BOUNDED | (a, b) | AffineTransform | Uniform |

Call `dist.default_transform()` to get the appropriate transform.

## Usage

```python
from minibayes import dist

# Create distributions
prior_mu = dist.Normal(loc=0, scale=10)
prior_sigma = dist.HalfNormal(scale=5)
prior_p = dist.Beta(alpha=2, beta=2)

# Evaluate log probability
prior_mu.log_prob(2.5)        # float
prior_mu.log_prob([1, 2, 3])  # ndarray

# Sample from distributions
import numpy as np
rng = np.random.default_rng(42)
samples = prior_sigma.sample(size=1000, rng=rng)

# Get transform for MCMC
transform = prior_sigma.default_transform()  # LogTransform
```

## Parameterization Notes

### Rate vs Scale

Some libraries use scale parameterization, others use rate. minibayes uses:

| Distribution | minibayes | NumPy | SciPy |
|--------------|-----------|-------|-------|
| Exponential | rate (λ) | scale (1/λ) | scale (1/λ) |
| Gamma | shape, rate | shape, scale | a, scale |

The rate parameterization is common in Bayesian inference (Stan, PyMC).

### Gamma Distribution

The Gamma PDF with shape α and rate β is:
```
p(x) = (β^α / Γ(α)) x^(α-1) exp(-βx)
```

To convert from NumPy/SciPy (shape, scale) to minibayes (shape, rate):
```python
rate = 1 / scale
```

## References

- [Wikipedia: Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
- [Wikipedia: Half-Normal Distribution](https://en.wikipedia.org/wiki/Half-normal_distribution)
- [Wikipedia: Exponential Distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
- [Wikipedia: Gamma Distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
- [Wikipedia: Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution)
- [Stan Functions Reference](https://mc-stan.org/docs/functions-reference/)
- [PyMC Distributions](https://www.pymc.io/projects/docs/en/latest/api/distributions.html)
