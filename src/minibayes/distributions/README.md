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

### Continuous - Unbounded (REAL)

| Distribution | Support | Parameters | Use Case |
|--------------|---------|------------|----------|
| **Normal** | (-∞, +∞) | loc (μ), scale (σ) | Unbounded continuous parameters |
| **StudentT** | (-∞, +∞) | df (ν), loc (μ), scale (σ) | Robust regression, heavy tails |
| **Cauchy** | (-∞, +∞) | loc, scale | Very heavy tails, robust priors |
| **Laplace** | (-∞, +∞) | loc, scale | Sparse priors (L1 regularization) |
| **MultivariateNormal** | ℝᵈ | mean (d,), cov (d,d) | Correlated multivariate outcomes |
| **LKJCholesky** | Cholesky(d) | dim (d), eta (η) | Correlation matrix prior |

### Continuous - Positive (POSITIVE)

| Distribution | Support | Parameters | Use Case |
|--------------|---------|------------|----------|
| **HalfNormal** | (0, +∞) | scale (σ) | Scale parameters, positive values |
| **Exponential** | (0, +∞) | rate (λ) | Waiting times, sparse priors |
| **Gamma** | (0, +∞) | shape (α), rate (β) | Positive parameters, precision |
| **LogNormal** | (0, +∞) | loc (μ), scale (σ) | Prices, concentrations, skewed data |
| **InverseGamma** | (0, +∞) | shape (α), scale (β) | Variance priors, conjugate for Normal |

### Continuous - Unit Interval (UNIT)

| Distribution | Support | Parameters | Use Case |
|--------------|---------|------------|----------|
| **Beta** | (0, 1) | alpha (α), beta (β) | Probabilities, proportions |

### Continuous - Bounded (BOUNDED)

| Distribution | Support | Parameters | Use Case |
|--------------|---------|------------|----------|
| **Uniform** | [low, high] | low, high | Bounded parameters, non-informative |

### Discrete

| Distribution | Support | Parameters | Use Case |
|--------------|---------|------------|----------|
| **Bernoulli** | {0, 1} | prob (p) | Binary outcomes, classification |
| **Poisson** | {0, 1, 2, ...} | rate (λ) | Count data, rare events |

## Log-Probability Formulas

### Continuous

| Distribution | log p(x) |
|--------------|----------|
| **Normal(μ, σ)** | -½log(2π) - log(σ) - (x-μ)²/(2σ²) |
| **StudentT(ν, μ, σ)** | log(Γ((ν+1)/2)) - log(Γ(ν/2)) - ½log(νπσ²) - ((ν+1)/2)·log(1 + z²/ν) |
| **Cauchy(μ, σ)** | -log(πσ) - log(1 + ((x-μ)/σ)²) |
| **Laplace(μ, σ)** | -log(2σ) - \|x-μ\|/σ |
| **HalfNormal(σ)** | log(√(2/π)) - log(σ) - x²/(2σ²) for x > 0 |
| **Exponential(λ)** | log(λ) - λx for x > 0 |
| **Gamma(α, β)** | α·log(β) + (α-1)·log(x) - βx - log(Γ(α)) for x > 0 |
| **LogNormal(μ, σ)** | -½log(2π) - log(σ) - log(x) - (log(x)-μ)²/(2σ²) for x > 0 |
| **InverseGamma(α, β)** | α·log(β) - log(Γ(α)) - (α+1)·log(x) - β/x for x > 0 |
| **Beta(α, β)** | (α-1)·log(x) + (β-1)·log(1-x) - log(B(α,β)) for x ∈ (0,1) |
| **Uniform(a, b)** | -log(b-a) for x ∈ [a,b] |
| **MultivariateNormal(μ, Σ)** | -½(d·log(2π) + log\|Σ\| + (x-μ)ᵀΣ⁻¹(x-μ)) |
| **LKJCholesky(d, η)** | Σₖ₌₂ᵈ (d - k + 2η - 2)·log(L[k,k]) |

### Discrete

| Distribution | log P(X=k) |
|--------------|------------|
| **Bernoulli(p)** | k·log(p) + (1-k)·log(1-p) for k ∈ {0, 1} |
| **Poisson(λ)** | k·log(λ) - λ - log(k!) for k ∈ {0, 1, 2, ...} |

All distributions return -∞ outside their support.

## Support and Transforms

Distributions are automatically paired with transforms based on their support:

| Support | Domain | Transform | Distributions |
|---------|--------|-----------|---------------|
| REAL | (-∞, +∞) | IdentityTransform | Normal, StudentT, Cauchy, Laplace |
| POSITIVE | (0, +∞) | LogTransform | HalfNormal, Exponential, Gamma, LogNormal, InverseGamma |
| UNIT | (0, 1) | LogitTransform | Beta |
| BOUNDED | (a, b) | AffineTransform | Uniform |
| BINARY | {0, 1} | IdentityTransform | Bernoulli |
| NATURAL | {0, 1, ...} | IdentityTransform | Poisson |

Call `dist.default_transform()` to get the appropriate transform.

**Note:** Discrete distributions (Bernoulli, Poisson) and MultivariateNormal are primarily used in likelihood functions, not as sampled parameters. LKJCholesky uses `CorrCholeskyTransform` for sampling.

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

# MultivariateNormal for correlated outcomes (use in likelihood)
mean = np.array([0.0, 0.0])
cov = np.array([[1.0, 0.5], [0.5, 1.0]])
mvn = dist.MultivariateNormal(mean=mean, cov=cov)
mvn.log_prob(np.array([0.1, 0.2]))      # Single observation -> float
mvn.log_prob(np.array([[0, 0], [1, 1]])) # Batch -> ndarray
mvn.obs_logp(data)                       # Sum of log_prob for batch

# LKJCholesky for correlation matrix priors (use shape= for matrix params)
lkj = dist.LKJCholesky(dim=2, eta=2.0)
L = lkj.sample(rng=rng)                  # (2, 2) lower triangular Cholesky
corr = L @ L.T                           # Correlation matrix
lkj.log_prob(L)                          # Log density of Cholesky factor

# In a model with unknown covariance:
def priors(p):
    sigma_1 = p("sigma_1", dist.HalfNormal(10))
    sigma_2 = p("sigma_2", dist.HalfNormal(10))
    L_corr = p("L_corr", dist.LKJCholesky(dim=2, eta=2.0), shape=(2, 2))
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
- [Wikipedia: Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)
- [Wikipedia: Cauchy Distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
- [Wikipedia: Laplace Distribution](https://en.wikipedia.org/wiki/Laplace_distribution)
- [Wikipedia: Half-Normal Distribution](https://en.wikipedia.org/wiki/Half-normal_distribution)
- [Wikipedia: Exponential Distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
- [Wikipedia: Gamma Distribution](https://en.wikipedia.org/wiki/Gamma_distribution)
- [Wikipedia: Log-Normal Distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)
- [Wikipedia: Inverse-Gamma Distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
- [Wikipedia: Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution)
- [Wikipedia: Bernoulli Distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)
- [Wikipedia: Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
- [Wikipedia: Multivariate Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
- [Stan LKJ Correlation Distribution](https://mc-stan.org/docs/functions-reference/correlation_matrix_distributions.html)
- [Stan Functions Reference](https://mc-stan.org/docs/functions-reference/)
- [PyMC Distributions](https://www.pymc.io/projects/docs/en/latest/api/distributions.html)
