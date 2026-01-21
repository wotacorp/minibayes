# Probability Distributions

Probability distributions for specifying priors and likelihoods in Bayesian inference.

## Quick start

```python
from minibayes import dist
import numpy as np

# Create distributions
prior_mu = dist.Normal(loc=0, scale=10)
prior_sigma = dist.HalfNormal(scale=5)

# Evaluate log probability
prior_mu.log_prob(2.5)        # float
prior_mu.log_prob([1, 2, 3])  # ndarray

# Sample from distributions
rng = np.random.default_rng(42)
samples = prior_sigma.sample(size=1000, rng=rng)

# Get automatic transform for MCMC
prior_sigma.default_transform()  # LogTransform
```

## Available distributions

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
| **TruncatedNormal** | [lower, upper] | mu, sigma, lower, upper | Bounded normal, lower-bounded positive params |

### Discrete

| Distribution | Support | Parameters | Use Case |
|--------------|---------|------------|----------|
| **Bernoulli** | {0, 1} | prob (p) | Binary outcomes, classification |
| **Poisson** | {0, 1, 2, ...} | rate (λ) | Count data, rare events |

## Log-probability formulas

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
| **TruncatedNormal(μ, σ, a, b)** | Normal log-prob - log(Φ((b-μ)/σ) - Φ((a-μ)/σ)) for x ∈ [a,b] |
| **MultivariateNormal(μ, Σ)** | -½(d·log(2π) + log\|Σ\| + (x-μ)ᵀΣ⁻¹(x-μ)) |
| **LKJCholesky(d, η)** | Σₖ₌₂ᵈ (d - k + 2η - 2)·log(L[k,k]) |

### Discrete

| Distribution | log P(X=k) |
|--------------|------------|
| **Bernoulli(p)** | k·log(p) + (1-k)·log(1-p) for k ∈ {0, 1} |
| **Poisson(λ)** | k·log(λ) - λ - log(k!) for k ∈ {0, 1, 2, ...} |

All distributions return -∞ outside their support.

## Support and transforms

Distributions are automatically paired with transforms based on their support:

| Support | Domain | Transform | Distributions |
|---------|--------|-----------|---------------|
| REAL | (-∞, +∞) | IdentityTransform | Normal, StudentT, Cauchy, Laplace |
| POSITIVE | (0, +∞) | LogTransform | HalfNormal, Exponential, Gamma, LogNormal, InverseGamma |
| UNIT | (0, 1) | LogitTransform | Beta |
| BOUNDED | (a, b) | AffineTransform | Uniform, TruncatedNormal |
| BINARY | {0, 1} | IdentityTransform | Bernoulli |
| NATURAL | {0, 1, ...} | IdentityTransform | Poisson |

**Note**: Discrete distributions and MultivariateNormal are for likelihoods, not sampled parameters. LKJCholesky uses `CorrCholeskyTransform`.

## Multivariate distributions

```python
# MultivariateNormal for correlated outcomes
mvn = dist.MultivariateNormal(mean=np.zeros(2), cov=np.eye(2))
mvn.log_prob(np.array([0.1, 0.2]))  # Single observation

# LKJCholesky for correlation matrix priors
lkj = dist.LKJCholesky(dim=2, eta=2.0)
L = lkj.sample(rng=rng)  # (2, 2) lower triangular Cholesky
corr = L @ L.T           # Correlation matrix
```

## Parameterization

minibayes uses **rate parameterization** (common in Bayesian inference, matching Stan/PyMC):

| Distribution | minibayes | NumPy/SciPy |
|--------------|-----------|-------------|
| Exponential | rate (λ) | scale (1/λ) |
| Gamma | shape, rate | shape, scale |

Convert from NumPy/SciPy: `rate = 1 / scale`

## References

- [Stan Functions Reference](https://mc-stan.org/docs/functions-reference/) - Comprehensive distribution documentation
- [PyMC Distributions](https://www.pymc.io/projects/docs/en/latest/api/distributions.html)
- [Stan LKJ Correlation Distribution](https://mc-stan.org/docs/functions-reference/correlation_matrix_distributions.html)
