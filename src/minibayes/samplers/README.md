# MCMC Samplers

Markov Chain Monte Carlo samplers for drawing samples from Bayesian posterior distributions.

## Sampler Philosophy

minibayes focuses on **gradient-free** MCMC methods. This is a deliberate design choice:

| Approach | Pros | Cons |
|----------|------|------|
| **Gradient-free (our choice)** | Simple, robust, educational, minimal dependencies | Slower for high-d |
| **Gradient-based (HMC/NUTS)** | Efficient for high-d, better mixing | Requires AD framework, complex |

For most models minibayes targets (regression, hierarchical models with <50 params), well-tuned gradient-free samplers perform adequately. For high-dimensional problems, we recommend NumPyro or PyMC.

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
| **EnsembleSampler** | Affine-invariant ensemble (emcee-style) | None (affine-invariant) |

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

## Adaptive Metropolis

Adapts the proposal covariance during warmup using the algorithm of Haario et al. (2001).

### How It Works

1. **During warmup**: Accumulates samples and periodically updates proposal covariance
2. **Optimal scaling**: Uses `Σ = (2.38²/d) × Σ_empirical + ε×I` (optimal for Gaussian targets)
3. **After warmup**: Freezes covariance and frees memory

The 2.38²/d scaling factor was proven optimal by Roberts & Rosenthal (2001), achieving ~23% acceptance rate for Gaussian targets.

### Usage

```python
import numpy as np
from minibayes.samplers import AdaptiveMetropolis

sampler = AdaptiveMetropolis(initial_scale=1.0, target_acceptance=0.234)

# Define log probability
def log_prob(params: dict[str, float]) -> float:
    return -0.5 * (params["x"]**2 + params["y"]**2)

rng = np.random.default_rng(42)
state = {"x": 0.0, "y": 0.0}

# Warmup (adaptation happens here)
for i in range(1000):
    state, _ = sampler.warmup_step(state, log_prob, rng, step_num=i)

sampler.freeze()  # Stop adaptation, free memory

# Sampling (fixed covariance)
samples = []
for _ in range(5000):
    state, _ = sampler.step(state, log_prob, rng)
    samples.append(state.copy())
```

### When to Use

| Use Case | Recommendation |
|----------|----------------|
| Known parameter scales | Use `MetropolisHastings` with tuned scales |
| Unknown scales | Use `AdaptiveMetropolis` |
| Correlated parameters | Use `AdaptiveMetropolis` (learns correlations) |
| Single parameter | Either works; MH is simpler |

## Ensemble Sampler (emcee-style)

An affine-invariant ensemble sampler based on Goodman & Weare (2010). Uses K parallel "walkers" that move using stretch moves, making it **affine-invariant** (no tuning required regardless of parameter scaling).

### How It Works

The ensemble sampler maintains K walkers (typically K >= 2 * ndim). At each iteration:

1. **Split walkers** into two complementary sets (red-blue pattern)
2. **For each walker j** in the active set:
   - Select random walker k from complementary set
   - Generate stretch factor z from: `z = ((a-1)*U + 1)² / a` where U ~ Uniform(0,1)
   - Propose: `y = x_k + z * (x_j - x_k)`
   - Accept with probability: `min(1, z^(d-1) * p(y)/p(x_j))`
3. **Update second half** using newly updated first half as complement

The stretch scale `a` (default 2.0) controls proposal aggressiveness. This is the only tuning parameter, and the default works well for most problems.

### Key Properties

- **Affine-invariant**: Works well regardless of parameter scaling or correlations
- **No tuning**: Unlike MH, doesn't require proposal scale tuning
- **Parallel exploration**: Walkers can explore different modes simultaneously
- **Simple acceptance**: Only requires likelihood evaluations (no gradients)

### Usage

```python
import numpy as np
from minibayes.samplers import EnsembleSampler

# Create sampler
sampler = EnsembleSampler(stretch_scale=2.0)  # 2.0 is default, rarely needs changing

def log_prob(params: dict[str, float]) -> float:
    return -0.5 * (params["x"]**2 + params["y"]**2)

rng = np.random.default_rng(42)

# Initialize K walkers (K must be even, >= 2*ndim recommended)
initial_states = [{"x": rng.normal(), "y": rng.normal()} for _ in range(16)]
sampler.initialize(initial_states, log_prob)

# Warmup (ensemble doesn't adapt, but helps walkers spread out)
for _ in range(500):
    sampler.advance(log_prob, rng)

# Sampling
samples = []
for _ in range(2000):
    sampler.advance(log_prob, rng)
    for state in sampler.get_states():
        samples.append(state.copy())

print(f"Acceptance rate: {sampler.acceptance_rate:.1%}")
```

### Using with mb.sample()

For the ensemble sampler, `num_chains` controls the number of walkers:

```python
import minibayes as mb

result = mb.sample(
    model,
    data=y,
    num_samples=5000,
    num_warmup=1000,
    num_chains=24,          # = num_walkers (must be even, >= 2*ndim)
    sampler="ensemble",
    sampler_kwargs={"stretch_scale": 2.0},  # optional
    seed=42,
)

# result.samples["mu"] has shape (24, 5000) - 24 walkers × 5000 samples
```

### When to Use

| Use Case | Recommendation |
|----------|----------------|
| Multimodal posteriors | **Ensemble** - walkers can explore different modes |
| Unknown parameter scales | **Ensemble** - affine-invariant, no tuning |
| Correlated parameters | Either Ensemble or AdaptiveMH |
| Maximizing ESS per second | Test both - problem dependent |
| Memory constrained | AdaptiveMH (fewer parallel states) |

### Comparison with Other Samplers

| Aspect | MH | AdaptiveMH | Ensemble |
|--------|-----|------------|----------|
| Tuning required | Yes (proposal_scale) | No (auto-adapts) | No (affine-invariant) |
| Learns correlations | No | Yes (during warmup) | Implicit (via stretch moves) |
| Multimodal | May get stuck | May get stuck | Can span modes |
| Memory | O(1) | O(warmup) | O(num_walkers) |
| Typical acceptance | 20-50% | 20-40% | 30-60% |

## MetropolisHastings Usage

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

All samplers implement the `Sampler` abstract base class with two interfaces:

### Stateless Interface (caller manages state)

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

### Stateful Interface (sampler manages state)

Used internally by `mb.sample()` for unified handling of single-chain and ensemble samplers:

```python
class Sampler(ABC):
    def initialize(
        self,
        initial_states: list[dict[str, float]],
        log_prob_fn: Callable[[dict[str, float]], float],
    ) -> None:
        """Initialize with one or more states (1 for MH, K for ensemble)."""

    def advance(
        self,
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
        warmup: bool = False,
        step_num: int = 0,
    ) -> float:
        """Advance all chains by one step. Returns average acceptance rate."""

    def get_states(self) -> list[dict[str, float]]:
        """Get current states of all chains."""

    @property
    def num_chains(self) -> int:
        """Number of chains being managed."""
```

Key points:
- Samplers work in **unconstrained space** (transforms are handled by Model)
- `log_prob_fn` returns the unnormalized log posterior (including Jacobian)
- Pass `rng` explicitly for reproducibility
- The stateful interface enables `mb.sample()` to handle all sampler types uniformly

## References

- [Wikipedia: Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
- [Stan User's Guide: MCMC Sampling](https://mc-stan.org/docs/stan-users-guide/mcmc.html)
- [PyMC Step Methods](https://www.pymc.io/projects/docs/en/latest/api/samplers.html)
- Roberts, Gelman & Gilks (1997). "Weak convergence and optimal scaling of random walk Metropolis algorithms"
- [Haario et al. (2001). "An adaptive Metropolis algorithm"](https://projecteuclid.org/journals/bernoulli/volume-7/issue-2/An-adaptive-Metropolis-algorithm/bj/1080222083.full)
- Roberts & Rosenthal (2001). "Optimal scaling for various Metropolis-Hastings algorithms"
- [Goodman & Weare (2010). "Ensemble samplers with affine invariance"](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf)
- [Foreman-Mackey et al. (2013). "emcee: The MCMC Hammer"](https://arxiv.org/abs/1202.3665)
