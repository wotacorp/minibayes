# MCMC Samplers

Markov Chain Monte Carlo samplers for drawing samples from Bayesian posterior distributions.

## Design philosophy

minibayes uses **gradient-free** MCMC methods:

| Approach | Pros | Cons |
|----------|------|------|
| **Gradient-free (our choice)** | Simple, robust, minimal dependencies | Slower for high-d |
| **Gradient-based (HMC/NUTS)** | Efficient for high-d, better mixing | Requires AD framework |

For models with <50 parameters, gradient-free samplers perform well. For larger problems, use NumPyro or PyMC.

## Available samplers

| Sampler | Description | Tuning |
|---------|-------------|--------|
| `MetropolisHastings` | Random walk MH with Gaussian proposals | Manual (`proposal_scale`) |
| `AdaptiveMetropolis` | MH with covariance tuning during warmup | Auto-adapts |
| `EnsembleSampler` | Affine-invariant ensemble (emcee-style) | None needed |

## Metropolis-Hastings

Basic random-walk sampler with fixed proposal scale.

```python
import minibayes as mb

# Via mb.sample() - recommended
result = mb.sample(model, data, sampler="mh", sampler_kwargs={"proposal_scale": 0.5})

# Per-parameter scales
result = mb.sample(model, data, sampler="mh",
    sampler_kwargs={"proposal_scale": {"mu": 0.5, "sigma": 0.1}})
```

**Tuning**: Target ~23% acceptance (high-d) to ~44% (1D). Adjust `proposal_scale` if acceptance is too low (scale too large) or too high (scale too small).

## Adaptive Metropolis

Learns proposal covariance during warmup (Haario et al. 2001). Uses optimal scaling factor 2.38²/d.

```python
result = mb.sample(model, data, sampler="adaptive_mh", num_warmup=1000)
```

**When to use**: Unknown parameter scales, correlated parameters. Default choice for most problems.

## Ensemble sampler

Affine-invariant ensemble sampler (Goodman & Weare 2010). Uses K parallel "walkers" with stretch moves.

```python
result = mb.sample(
    model, data,
    sampler="ensemble",
    num_chains=24,  # = num_walkers (must be even, >= 2*ndim)
    sampler_kwargs={"stretch_scale": 2.0},  # optional, default works well
)
# result.samples["mu"] has shape (24, 5000) - 24 walkers × 5000 samples
```

**Key properties**: Affine-invariant (no tuning needed), can explore multimodal posteriors, walkers share information via stretch moves.

## Comparison

| Aspect | MH | AdaptiveMH | Ensemble |
|--------|-----|------------|----------|
| Tuning required | Yes | No | No |
| Learns correlations | No | Yes | Implicit |
| Multimodal | May get stuck | May get stuck | Can span modes |
| Memory | O(1) | O(warmup) | O(num_walkers) |
| Typical acceptance | 20-50% | 20-40% | 30-60% |

## When to use which

| Use Case | Recommendation |
|----------|----------------|
| Default choice | `adaptive_mh` - learns structure automatically |
| Multimodal posteriors | `ensemble` - walkers explore different modes |
| Known parameter scales | `mh` with tuned proposal |
| Memory constrained | `mh` or `adaptive_mh` |

## Low-level interface

All samplers work in unconstrained space (transforms handled by Model). For direct use:

```python
from minibayes.samplers import MetropolisHastings

sampler = MetropolisHastings(proposal_scale=1.0)
state, accepted = sampler.step(current_state, log_prob_fn, rng)
```

## References

- [Haario et al. (2001). "An adaptive Metropolis algorithm"](https://projecteuclid.org/journals/bernoulli/volume-7/issue-2/An-adaptive-Metropolis-algorithm/bj/1080222083.full)
- [Goodman & Weare (2010). "Ensemble samplers with affine invariance"](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf)
- [Foreman-Mackey et al. (2013). "emcee: The MCMC Hammer"](https://arxiv.org/abs/1202.3665)
