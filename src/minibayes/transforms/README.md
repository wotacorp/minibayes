# Parameter Transforms

Bijective transforms for mapping constrained parameters to unconstrained space in MCMC sampling.

## Quick start

```python
from minibayes import dist

# Transforms are automatic based on distribution support
dist.Normal(0, 1).default_transform()    # IdentityTransform (REAL)
dist.HalfNormal(1).default_transform()   # LogTransform (POSITIVE)
dist.Beta(2, 2).default_transform()      # LogitTransform (UNIT)
dist.Uniform(0, 10).default_transform()  # AffineTransform (BOUNDED)
```

## Why transforms?

Many Bayesian parameters have constrained domains:
- Standard deviations must be positive (σ > 0)
- Probabilities lie in (0, 1)
- Correlation coefficients lie in (-1, 1)

MCMC samplers like Metropolis-Hastings work best in unconstrained space (ℝ). Transforms map constrained parameters θ to unconstrained parameters φ, allowing the sampler to explore freely.

## Jacobian correction

When we transform variables, probability densities must be adjusted by the Jacobian determinant:

```
log p(φ) = log p(θ) + log|dθ/dφ|
```

Where:
- θ = constrained parameter (original space)
- φ = unconstrained parameter (transformed space)
- log|dθ/dφ| = log absolute Jacobian of the **inverse** transform

This correction ensures samples in transformed space correspond to the correct distribution in original space.

## Reference table

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

| Transform | Jacobian dθ/dφ | Log Jacobian |
|-----------|----------------|--------------|
| **Log** (θ > 0) | exp(φ) = θ | log(θ) |
| **Logit** (θ ∈ (0,1)) | θ(1-θ) | log(θ) + log(1-θ) |
| **Affine** (θ ∈ (a,b)) | (b-a)·z(1-z) | log(θ-a) + log(b-θ) - log(b-a) |
| **ShiftedLog** (θ > a) | θ - a | log(θ-a) |
| **CorrCholesky** | See implementation | Complex (matrix-valued) |

**CorrCholesky**: Maps d(d-1)/2 off-diagonal elements to unconstrained space via arctanh. Used automatically for LKJCholesky distributions.

## References

- [Stan User's Guide: Changes of Variables](https://mc-stan.org/docs/stan-users-guide/changes-of-variables.html)
- [PyMC Transforms](https://www.pymc.io/projects/docs/en/latest/api/distributions/transforms.html)
- [TensorFlow Probability Bijectors](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors)
- [Turing.jl Bijectors](https://turinglang.org/docs/developers/transforms/bijectors/)
