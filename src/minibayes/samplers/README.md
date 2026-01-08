# MCMC Samplers

Markov Chain Monte Carlo samplers for drawing samples from Bayesian posterior distributions.

## Why MCMC?

In Bayesian inference, we need to compute expectations over the posterior distribution:

```
E[f(θ)|data] = ∫ f(θ) p(θ|data) dθ
```

For most models, this integral is intractable. MCMC solves this by:
- Constructing a Markov chain whose stationary distribution is the target posterior
- Running the chain to generate samples θ₁, θ₂, ..., θₙ
- Approximating expectations as sample averages: E[f(θ)] ≈ (1/n) Σ f(θᵢ)

The key insight: we only need to evaluate the posterior up to a normalizing constant, since MCMC uses ratios of densities.

## Available Samplers

| Sampler | Description | Adaptation |
|---------|-------------|------------|
| **MetropolisHastings** | Random walk MH with Gaussian proposals | None (fixed scale) |
| **AdaptiveMetropolis** | MH with covariance tuning during warmup | During warmup only |

## The Metropolis-Hastings Algorithm

### Algorithm Steps

1. **Initialize** at starting point θ₀
2. **For each iteration t:**
   - Propose θ' from proposal distribution q(θ'|θₜ)
   - Compute acceptance ratio α
   - With probability min(1, α): accept θₜ₊₁ = θ'
   - Otherwise: reject θₜ₊₁ = θₜ

### Acceptance Ratio

For symmetric proposals (like Gaussian random walk), q(θ'|θ) = q(θ|θ'), so:

```
α = p(θ'|data) / p(θ|data) = exp(log p(θ') - log p(θ))
```

In log space (for numerical stability):
```
log α = log p(θ') - log p(θ)
accept if log(u) < log α, where u ~ Uniform(0,1)
```

### Tuning Guidelines

| Dimension | Target Acceptance Rate |
|-----------|----------------------|
| 1D | ~44% |
| High-d | ~23% |

Diagnosing problems:
- **0% acceptance**: Proposal scale too large (proposals always in low-probability regions)
- **100% acceptance**: Proposal scale too small (not exploring the space)

Rule of thumb: adjust `proposal_scale` until acceptance is in the target range.

## Usage

```python
import numpy as np
from minibayes.samplers import MetropolisHastings

# Create sampler with fixed proposal scale
sampler = MetropolisHastings(proposal_scale=1.0)

# Or per-parameter scales
sampler = MetropolisHastings(proposal_scale={"mu": 0.5, "sigma": 0.1})

# Define log probability function
def log_prob(params: dict[str, float]) -> float:
    mu = params["mu"]
    return -0.5 * mu ** 2  # Standard normal

# Run sampler
rng = np.random.default_rng(42)
state = {"mu": 0.0}

samples = []
for _ in range(1000):
    state, accepted = sampler.step(state, log_prob, rng)
    samples.append(state["mu"])
```

## Sampler Interface

All samplers implement the `Sampler` abstract base class:

```python
class Sampler(ABC):
    def step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], bool]:
        """Take one MCMC step. Returns (new_state, accepted)."""

    def warmup_step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        step_num: int,
    ) -> tuple[dict[str, float], bool]:
        """Take one warmup step (may adapt internal state)."""
```

Key points:
- Samplers work in **unconstrained space** (transforms are handled by Model)
- `log_prob_fn` returns the unnormalized log posterior (including Jacobian)
- Pass `rng` explicitly for reproducibility

## References

- [Wikipedia: Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
- [Stan User's Guide: MCMC Sampling](https://mc-stan.org/docs/stan-users-guide/mcmc.html)
- [PyMC Step Methods](https://www.pymc.io/projects/docs/en/latest/api/samplers.html)
- Roberts, Gelman & Gilks (1997). "Weak convergence and optimal scaling of random walk Metropolis algorithms"
