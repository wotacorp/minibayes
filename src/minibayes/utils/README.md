# Numerical Utilities

Helper functions for numerically stable computations in MCMC sampling.

## Quick start

```python
import numpy as np
from minibayes.utils import log_sum_exp, ensure_rng, check_finite

# Numerically stable log(sum(exp(x)))
log_probs = np.array([-1000, -1001, -1002])
log_sum_exp(log_probs)  # -999.59 (direct exp() would underflow)

# Reproducible random number generation
rng = ensure_rng(42)      # int → Generator
rng = ensure_rng(rng)     # Generator → same Generator
rng = ensure_rng(None)    # None → new Generator

# Guard against NaN/Inf
check_finite(-500.0, "log_prob")  # OK
check_finite(np.nan, "log_prob")  # Raises NumericalError
```

## Why numerical stability?

MCMC algorithms operate in log-space to avoid underflow/overflow:
- Likelihoods can be extremely small (e.g., 10⁻¹⁰⁰⁰)
- Products become sums in log-space
- Ratios become differences

Naive implementations often fail on real problems. These utilities provide stable building blocks.

## Functions

### `log_sum_exp(x)`

Compute `log(sum(exp(x)))` without overflow or underflow.

**The Problem:** Direct computation overflows for large values and underflows for small values.

**The Solution:**
```
log(Σ exp(xᵢ)) = max(x) + log(Σ exp(xᵢ - max(x)))
```

By shifting values, the largest becomes 0 and others are negative but computable.

### `ensure_rng(seed)`

Normalize seed/Generator/None to a NumPy Generator for reproducible sampling.

**Why it matters:**
- Reproducibility requires explicit RNG handling
- Never use `np.random.*` global functions
- Pass generators explicitly through the call stack

### `check_finite(value, name)`

Guard against NaN/Inf propagation in log-probability computations.

**Why it matters:**
- NaN/Inf in `log_prob` silently corrupts MCMC chains
- Early detection with clear error messages aids debugging

## Log-space arithmetic

| Operation | Direct | Log-Space |
|-----------|--------|-----------|
| a × b | `a * b` | `log_a + log_b` |
| a / b | `a / b` | `log_a - log_b` |
| a + b | `a + b` | `log_sum_exp([log_a, log_b])` |
| Σ aᵢ | `sum(a)` | `log_sum_exp(log_a)` |

## References

- [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
- [Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
