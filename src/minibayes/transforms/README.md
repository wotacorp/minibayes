# Parameter Transforms

Bijective transforms for mapping constrained parameters to unconstrained space in MCMC sampling.

## Why Transforms?

Many Bayesian parameters have constrained domains:
- Standard deviations must be positive (σ > 0)
- Probabilities lie in (0, 1)
- Correlation coefficients lie in (-1, 1)

MCMC samplers like Metropolis-Hastings work best in unconstrained space (ℝ). Transforms map constrained parameters θ to unconstrained parameters φ, allowing the sampler to explore freely.

## The Jacobian Correction

When we transform variables, probability densities must be adjusted by the Jacobian determinant:

```
log p(φ) = log p(θ) + log|dθ/dφ|
```

Where:
- θ = constrained parameter (original space)
- φ = unconstrained parameter (transformed space)
- log|dθ/dφ| = log absolute Jacobian of the **inverse** transform

This correction ensures samples in transformed space correspond to the correct distribution in original space.

## Reference Table

| Transform | θ domain | Forward: φ = f(θ) | Inverse: θ = g(φ) | log\|dθ/dφ\| |
|-----------|----------|-------------------|-------------------|--------------|
| **Identity** | ℝ | φ = θ | θ = φ | 0 |
| **Log** | (0, ∞) | φ = log(θ) | θ = exp(φ) | log(θ) |
| **Logit** | (0, 1) | φ = log(θ/(1-θ)) | θ = 1/(1+e⁻ᵠ) | log(θ) + log(1-θ) |
| **Affine** | (a, b) | φ = logit((θ-a)/(b-a)) | θ = a + (b-a)·σ(φ) | log(θ-a) + log(b-θ) - log(b-a) |
| **ShiftedLog** | (a, ∞) | φ = log(θ-a) | θ = exp(φ) + a | log(θ-a) |
| **CorrCholesky** | Cholesky(d) | φ = arctanh(normalized off-diag) | θ = Cholesky factor | See implementation |

Where σ(φ) = 1/(1+e⁻ᵠ) is the sigmoid function.

## Derivations

### Log Transform
For θ > 0, we use φ = log(θ):
- Inverse: θ = exp(φ)
- Jacobian: dθ/dφ = exp(φ) = θ
- Log Jacobian: log(θ)

### Logit Transform
For θ ∈ (0, 1), we use φ = logit(θ) = log(θ/(1-θ)):
- Inverse: θ = sigmoid(φ) = 1/(1+e⁻ᵠ)
- Jacobian: dθ/dφ = θ(1-θ)
- Log Jacobian: log(θ) + log(1-θ)

### Affine Transform
For θ ∈ (a, b), we scale to (0,1) then apply logit:
- Let z = (θ-a)/(b-a), then φ = logit(z)
- Inverse: z = sigmoid(φ), θ = a + (b-a)z
- Jacobian: dθ/dφ = (b-a) · z(1-z)
- Log Jacobian: log(θ-a) + log(b-θ) - log(b-a)

### ShiftedLog Transform
For θ ∈ (a, +∞), we use φ = log(θ-a):
- Inverse: θ = exp(φ) + a
- Jacobian: dθ/dφ = exp(φ) = θ - a
- Log Jacobian: log(θ-a) = φ

Useful for lower-bounded parameters (e.g., TruncatedNormal with only a lower bound).

### CorrCholesky Transform
For correlation matrix Cholesky factors L (d×d lower triangular where L·Lᵀ is a correlation matrix):
- Maps d(d-1)/2 off-diagonal elements to unconstrained space via arctanh
- Each off-diagonal is normalized by the remaining variance before transformation
- Diagonal elements are determined by the unit row norm constraint

This transform is used automatically for LKJCholesky distributions.

## Usage in minibayes

Transforms are automatically selected based on distribution support:

```python
from minibayes.distributions import Normal, HalfNormal, Beta

Normal(0, 1).default_transform()    # IdentityTransform (REAL)
HalfNormal(1).default_transform()   # LogTransform (POSITIVE)
Beta(2, 2).default_transform()      # LogitTransform (UNIT)
```

For bounded distributions, override `default_transform()` to provide bounds.

## References

- [Stan User's Guide: Changes of Variables](https://mc-stan.org/docs/stan-users-guide/changes-of-variables.html)
- [PyMC Transforms](https://www.pymc.io/projects/docs/en/latest/api/distributions/transforms.html)
- [TensorFlow Probability Bijectors](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors)
- [Turing.jl Bijectors](https://turinglang.org/docs/developers/transforms/bijectors/)
