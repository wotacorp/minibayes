# MiniBayes Implementation Roadmap

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FOUNDATION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   utils.py ──────────┬──────────────► transforms.py                     │
│   (numerical)        │                (Identity, Log, Logit)            │
│                      │                       │                          │
│                      ▼                       ▼                          │
│              distributions/base.py ◄─────────┘                          │
│              (Support enum, ABC)                                        │
│                      │                                                  │
│                      ▼                                                  │
│              distributions/continuous.py                                │
│              (Normal, HalfNormal, Beta, Gamma, Uniform, Exponential)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                 CORE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   model.py ◄──────── distributions + transforms                         │
│   (Model class: priors, log_likelihood, log_prob, transforms)           │
│                      │                                                  │
│                      ▼                                                  │
│   samplers/base.py ──► samplers/mh.py ──► samplers/adaptive.py          │
│   (Sampler ABC)        (basic MH)         (Adaptive MH)                 │
│                                                                         │
│   results.py ◄─────── samplers                                          │
│   (InferenceResult dataclass)                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              INTERFACE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   __init__.py ◄────── model + samplers + results                        │
│   (mb.sample entry point, public API)                                   │
│                                                                         │
│   diagnostics.py ◄─── results                                           │
│   (ESS, R-hat, summary)                                                 │
│                                                                         │
│   export.py ◄──────── results                                           │
│   (save/load npz, json)                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Block-by-Block Implementation

### Block 1: `utils.py` (~30 lines)

**Purpose:** Numerical helpers used everywhere.

```python
def log_sum_exp(a, b): ...           # Stable log(exp(a) + exp(b))
def check_finite(x, name): ...       # Raise NumericalError if NaN/Inf
def ensure_rng(seed): ...            # Convert seed/None/Generator → Generator
```

**Test:** Unit tests for edge cases (overflow, underflow).

---

### Block 2: `transforms.py` (~80 lines)

**Purpose:** Bijective maps between constrained ↔ unconstrained space.

```python
class Transform(ABC):
    def forward(self, x): ...        # Constrained → Unconstrained
    def inverse(self, y): ...        # Unconstrained → Constrained
    def log_det_jacobian(self, x): ...  # For density correction

class IdentityTransform(Transform): ...    # x ↔ x
class LogTransform(Transform): ...         # x ↔ log(x), for x > 0
class LogitTransform(Transform): ...       # x ↔ logit(x), for x ∈ (0,1)
```

**Key formulas:**

| Transform | Forward | Inverse | Log-Jacobian |
|-----------|---------|---------|--------------|
| Identity | y = x | x = y | 0 |
| Log | y = log(x) | x = exp(y) | log(x) = y |
| Logit | y = log(x/(1-x)) | x = σ(y) | log(x(1-x)) |

**Test:** `inverse(forward(x)) ≈ x` for random x in support.

---

### Block 3: `distributions/base.py` (~40 lines)

**Purpose:** Abstract base and support enum.

```python
class Support(Enum):
    REAL = "real"           # (-∞, +∞)
    POSITIVE = "positive"   # (0, +∞)
    UNIT = "unit"           # (0, 1)

class Distribution(ABC):
    @property
    @abstractmethod
    def support(self) -> Support: ...

    @abstractmethod
    def log_prob(self, x) -> float | np.ndarray: ...

    @abstractmethod
    def sample(self, size=None, rng=None): ...

    def default_transform(self) -> Transform:
        """Derive from support."""
```

**Test:** None directly (abstract).

---

### Block 4: `distributions/continuous.py` (~150 lines)

**Purpose:** Concrete distributions with analytical log_prob.

**Distributions:** Normal, HalfNormal, Exponential, Beta, Gamma, Uniform

**Log-prob formulas:**

| Distribution | Log-prob |
|--------------|----------|
| Normal(μ,σ) | -½log(2π) - log(σ) - (x-μ)²/(2σ²) |
| HalfNormal(σ) | log(2) - ½log(2π) - log(σ) - x²/(2σ²) |
| Exponential(λ) | log(λ) - λx |
| Beta(α,β) | (α-1)log(x) + (β-1)log(1-x) - log(B(α,β)) |
| Gamma(α,β) | α·log(β) + (α-1)log(x) - βx - log(Γ(α)) |

**Test:** Compare `log_prob` against `scipy.stats.*.logpdf`.

---

### Block 5: `model.py` (~120 lines)

**Purpose:** Combine priors + log-likelihood, handle transforms automatically.

```python
class Model:
    def __init__(self, priors: dict[str, Distribution], log_likelihood: Callable): ...

    @property
    def param_names(self) -> list[str]: ...

    @property
    def transforms(self) -> dict[str, Transform]: ...

    def sample_prior(self, rng=None) -> dict[str, float]: ...
    def log_prior(self, params: dict) -> float: ...
    def log_likelihood(self, params: dict, data) -> float: ...
    def log_prob(self, params: dict, data) -> float: ...

    def to_unconstrained(self, params: dict) -> dict: ...
    def to_constrained(self, unconstrained: dict) -> dict: ...
    def log_prob_unconstrained(self, unconstrained: dict, data) -> float: ...
```

**Test:**
- `to_constrained(to_unconstrained(params)) ≈ params`
- `log_prob_unconstrained` includes Jacobian

---

### Block 6: `samplers/base.py` (~25 lines)

**Purpose:** Abstract sampler interface.

```python
class Sampler(ABC):
    @abstractmethod
    def step(self, current: dict, log_prob_fn: Callable, rng) -> tuple[dict, bool]:
        """Single MCMC step. Returns (new_state, accepted)."""

    def warmup_step(self, current, log_prob_fn, rng, step_num):
        """Override for adaptive warmup."""
        return self.step(current, log_prob_fn, rng)
```

---

### Block 7: `samplers/mh.py` (~40 lines)

**Purpose:** Basic random-walk Metropolis-Hastings.

```python
class MetropolisHastings(Sampler):
    def __init__(self, proposal_scale: float | dict = 1.0): ...

    def step(self, current, log_prob_fn, rng):
        # 1. Propose: current + Normal(0, scale)
        # 2. Compute log_ratio = log_prob(proposal) - log_prob(current)
        # 3. Accept if log(uniform()) < log_ratio
```

**Test:** Normal-Normal model recovers analytical posterior.

---

### Block 8: `samplers/adaptive.py` (~80 lines)

**Purpose:** Adaptive Metropolis with covariance learning.

```python
class AdaptiveMetropolis(Sampler):
    def __init__(self, initial_scale=1.0, target_acceptance=0.234): ...

    def warmup_step(self, current, log_prob_fn, rng, step_num):
        # Store sample, update covariance estimate
        # Proposal covariance = (2.38²/d) × empirical_cov

    def step(self, current, log_prob_fn, rng):
        # Propose using learned covariance

    def freeze(self):
        """Call after warmup to stop adaptation."""
```

**Test:** Recovers correlated posterior (banana distribution).

---

### Block 9: `results.py` (~60 lines)

**Purpose:** Container for MCMC output.

```python
@dataclass
class InferenceResult:
    samples: dict[str, np.ndarray]              # Constrained space
    samples_unconstrained: dict[str, np.ndarray]
    acceptance_rate: float
    num_samples: int
    num_warmup: int
    num_chains: int
    sampler: str
    elapsed_time: float

    def summary(self, percentiles=[5, 50, 95]) -> dict: ...
    def to_dict(self) -> dict: ...
```

---

### Block 10: `diagnostics.py` (~80 lines)

**Purpose:** Convergence diagnostics.

```python
def effective_sample_size(samples: np.ndarray) -> float:
    """ESS via autocorrelation (FFT method)."""

def r_hat(chains: np.ndarray) -> float:
    """Gelman-Rubin diagnostic. chains: (num_chains, num_samples)"""
```

**Test:**
- ESS < n for correlated, ESS ≈ n for independent
- R-hat ≈ 1.0 for converged chains

---

### Block 11: `export.py` (~40 lines)

**Purpose:** Persistence.

```python
def save_npz(result: InferenceResult, path: str): ...
def load_npz(path: str) -> InferenceResult: ...
def save_json(result: InferenceResult, path: str): ...
```

---

### Block 12: `inference.py` — Entry Point (~100 lines)

**Purpose:** Main API, orchestrates everything.

```python
def sample(
    model: Model,
    data = None,
    initial: dict | None = None,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    sampler: str = "adaptive_mh",
    sampler_kwargs: dict | None = None,
    seed: int | None = None,
) -> InferenceResult:
    """
    1. Validate inputs (sampler name)
    2. Set up RNG, spawn child RNGs for chains
    3. For each chain:
       - Get initial state (sample from prior if None)
       - Warmup phase (sampler.warmup_step)
       - Call sampler.post_warmup()
       - Sampling phase (sampler.step)
    4. Transform samples to constrained space
    5. Return InferenceResult
    """
```

---

## Testing Strategy

| Block | Test Type | What to Verify |
|-------|-----------|----------------|
| transforms | Unit | Roundtrip, Jacobian correctness |
| distributions | Unit | log_prob matches scipy.stats |
| model | Unit | Transform composition, Jacobian included |
| samplers/mh | Integration | Normal-Normal posterior recovery |
| samplers/adaptive | Integration | Correlated posterior recovery |
| diagnostics | Unit | ESS < n for correlated, R-hat formulas |
| mb.sample | End-to-end | Model-based sampling, reproducibility, multi-chain |

---

## Key Gotchas

### 1. Numerical Stability
- Always work in log-space: `log_p1 - log_p2`, never `p1 / p2`
- Clip values near boundaries: `np.clip(x, 1e-10, 1 - 1e-10)` for Beta

### 2. Jacobian Correction
```python
def log_prob_unconstrained(self, unconstrained, data):
    constrained = self.to_constrained(unconstrained)
    lp = self.log_prob(constrained, data)
    for k in self.param_names:
        lp += self._transforms[k].log_det_jacobian(constrained[k])
    return lp
```

### 3. RNG Reproducibility
- Never use `np.random.*` global functions
- Pass `rng: np.random.Generator` explicitly everywhere
- Derive independent chain RNGs from master

### 4. Warmup Freezing
- Adaptive samplers must stop adapting after warmup
- Free memory: `self._sample_history = []`

### 5. Dict Ordering
- Fix parameter order at Model init: `self._param_order = list(priors.keys())`
- Use consistently when converting dict ↔ vector

### 6. Covariance Stability
- Regularize: `cov += 1e-6 * np.eye(d)`
- Check positive definite before using

### 7. Initial Values
- Sample from prior, but verify `log_prob` is finite
- Retry if necessary

### 8. Scipy Parameterization
- Your `Gamma(shape, rate)` vs scipy `gamma(a, scale=1/rate)`
- Document your parameterization clearly

### 9. Acceptance Rate Sanity
- 0% or 100% = definite bug
- >95% = proposal too small
- <5% = proposal too large
- Target: ~23% for high-d, ~44% for 1D

### 10. Memory
- Preallocate arrays, don't append to lists
- `samples = {k: np.empty(num_samples) for k in param_names}`

---

## Note on HMC/NUTS

After analysis, we've decided not to implement gradient-based samplers (HMC/NUTS) in minibayes. See minibayes-spec.md "Design Decisions" section for detailed rationale.

The next phase focuses on the affine-invariant ensemble sampler (emcee-style) for better mixing without the complexity of automatic differentiation.

---

## Library Comparison: minibayes vs PyMC vs NumPyro

### Package Fundamentals

| Criterion | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Core Dependencies** | numpy only | pytensor, arviz, numpy, scipy, cloudpickle, cachetools, pandas, typing-extensions | jax, jaxlib, numpy, tqdm |
| **Optional Dependencies** | matplotlib (viz), scipy (dev) | nutpie, numpyro, blackjax (alternative samplers) | funsor, tensorflow-probability |
| **Installed Size** | ~5-10 MB (estimated) | ~150-300 MB (with deps) | ~100-200 MB (with JAX) |
| **Python Version** | 3.11+ | 3.10+ | 3.9+ |
| **GPU Support** | No | Via JAX backend | Yes (JAX native) |
| **Automatic Differentiation** | No (by design) | Yes (PyTensor) | Yes (JAX) |

### Codebase Metrics

| Criterion | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Lines of Code (core)** | ~7,700 | ~50,000+ | ~30,000+ |
| **Number of Files** | ~45 | ~200+ | ~150+ |
| **Test Coverage** | High (strict mypy) | High | High |
| **Documentation Style** | NumPy docstrings | Sphinx + tutorials | Sphinx + tutorials |

### Distributions

| Criterion | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Total Distributions** | 16 | 60+ | 100+ |
| **Continuous Unbounded** | Normal, StudentT, Cauchy, Laplace | All standard + many exotic | Comprehensive |
| **Continuous Positive** | HalfNormal, Exponential, Gamma, LogNormal, InverseGamma | All standard + Weibull, Pareto, etc. | Comprehensive |
| **Continuous Bounded** | Beta, Uniform, TruncatedNormal | All standard + many truncated | Comprehensive |
| **Discrete** | Bernoulli, Poisson | Binomial, NegBinomial, Categorical, Geometric, etc. | Comprehensive |
| **Multivariate** | MultivariateNormal, LKJCholesky | MVN, Wishart, Dirichlet, LKJ, etc. | Comprehensive |
| **Time Series** | None | AR, GARCH, GaussianRandomWalk | GaussianHMM, etc. |
| **Mixture Models** | None (planned v0.7) | Mixture, NormalMixture | MixtureGeneral |

### Inference Methods (Samplers)

| Criterion | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Total Samplers** | 3 | 5+ | 10+ |
| **Metropolis-Hastings** | Yes | Yes | Yes |
| **Adaptive MH** | Yes (covariance tuning) | Yes | - |
| **Ensemble (emcee-style)** | Yes | Via external | - |
| **NUTS** | No (by design) | Yes (default) | Yes (default) |
| **HMC** | No (by design) | Yes | Yes |
| **Variational Inference** | No | ADVI, FullRank ADVI | SVI, AutoGuides |
| **Sequential Monte Carlo** | No | SMC | - |
| **Slice Sampling** | No | Yes | - |
| **Laplace Approximation** | No | Yes | Yes |

### Model Features

| Criterion | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Hierarchical Models** | Yes (p() API) | Yes (context manager) | Yes (effect handlers) |
| **Vector Parameters** | Yes (size=) | Yes | Yes |
| **Matrix Parameters** | Yes (shape=) | Yes | Yes |
| **Automatic Transforms** | Yes (inspectable) | Yes | Yes |
| **Custom Transforms** | Yes (inherit Transform) | Yes | Yes |
| **Conditional Priors** | Yes (execution order) | Yes (automatic) | Yes (effect handlers) |
| **Model Comparison** | WAIC | WAIC, LOO-CV, BF | WAIC, LOO-CV |
| **Posterior Predictive** | Yes | Yes | Yes |
| **Prior Predictive** | Yes | Yes | Yes |

### Diagnostics

| Criterion | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Effective Sample Size** | Yes (FFT-based) | Yes (via ArviZ) | Yes |
| **R-hat** | Yes | Yes (via ArviZ) | Yes |
| **MCSE** | No | Yes (via ArviZ) | Yes |
| **Trace Plots** | Yes | Yes (via ArviZ) | Via ArviZ |
| **Autocorrelation Plots** | Yes | Yes (via ArviZ) | Via ArviZ |
| **Pair Plots** | Yes | Yes (via ArviZ) | Via ArviZ |
| **Forest Plots** | Yes | Yes (via ArviZ) | Via ArviZ |

### Design Philosophy

| Criterion | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Primary Goal** | Minimal, deployable, educational | Research, flexibility | Performance, scalability |
| **API Style** | Explicit function calls | Context managers | Effect handlers |
| **Magic/Implicit Behavior** | None (by design) | Some (context-aware) | Effect handlers |
| **Learning Curve** | Shallow | Moderate | Steep |
| **Code Readability** | High (MCMC visible) | Moderate | Lower (JAX idioms) |
| **Customization** | Direct inheritance | Extensive API | Effect handlers |

### Deployment & Production

| Criterion | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Edge Deployment** | Excellent (minimal deps) | Difficult (heavy deps) | Difficult (JAX) |
| **Container Size** | Small (~50 MB image) | Large (~500+ MB image) | Large (~400+ MB image) |
| **Memory Limits** | Configurable (max_samples, max_memory_mb) | Via ArviZ | Manual |
| **Serialization** | npz, JSON | NetCDF, ArviZ InferenceData | Custom |
| **Startup Time** | Fast (<1s) | Slow (compilation) | Slow (JIT compilation) |

### Target Use Cases

| Use Case | minibayes | PyMC | NumPyro |
|----------|-----------|------|---------|
| **Simple regression** | Excellent | Good | Good |
| **A/B testing** | Excellent | Good | Good |
| **Hierarchical models (d<50)** | Good | Excellent | Excellent |
| **High-dimensional (d>100)** | Poor (no gradients) | Excellent | Excellent |
| **Complex hierarchical** | Limited | Excellent | Excellent |
| **Time series** | Not supported | Excellent | Good |
| **Gaussian processes** | Planned (v0.8) | Excellent | Excellent |
| **Deep probabilistic models** | Not supported | Limited | Excellent |
| **Research/publication** | Limited | Excellent | Excellent |
| **Production/edge deployment** | Excellent | Poor | Poor |
| **Educational/learning MCMC** | Excellent | Moderate | Poor |
| **Resource-constrained systems** | Excellent | Poor | Poor |

### Performance Benchmarks

| Benchmark | minibayes | PyMC | NumPyro |
|-----------|-----------|------|---------|
| **Eight Schools (10 params)** | TBD | TBD | TBD |
| **Linear Regression (3 params)** | TBD | TBD | TBD |
| **Hierarchical Sensor (20 params)** | TBD | TBD | TBD |
| **Startup time** | TBD | TBD | TBD |
| **Memory usage (peak)** | TBD | TBD | TBD |
| **ESS/second** | TBD | TBD | TBD |

---

### Example Code: Linear Regression

**minibayes**
```python
import minibayes as mb
from minibayes import distributions as dist

def priors(p):
    p("alpha", dist.Normal(0, 10))
    p("beta", dist.Normal(0, 10))
    p("sigma", dist.HalfNormal(5))

def log_likelihood(params, data):
    mu = params["alpha"] + params["beta"] * data["x"]
    return dist.Normal(mu, params["sigma"]).log_prob(data["y"])

model = mb.Model(priors=priors, log_likelihood=log_likelihood)
result = mb.sample(model, data, num_samples=2000, num_warmup=1000, num_chains=4)
```

**PyMC**
```python
import pymc as pm

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = alpha + beta * x
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, chains=4)
```

**NumPyro**
```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(x, y=None):
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    beta = numpyro.sample("beta", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))

    mu = alpha + beta * x
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0), x, y=y)
```

---

### Example Code: Hierarchical Model (Eight Schools)

**minibayes**
```python
def priors(p):
    mu = p("mu", dist.Normal(0, 5))
    tau = p("tau", dist.HalfNormal(5))
    theta = p("theta", dist.Normal(mu, tau), size=8)  # Vector param

def log_likelihood(params, data):
    return dist.Normal(params["theta"], data["sigma"]).log_prob(data["y"])

model = mb.Model(priors=priors, log_likelihood=log_likelihood)
result = mb.sample(model, data, sampler="ensemble", num_chains=20)
```

**PyMC**
```python
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfNormal("tau", sigma=5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)

    y_obs = pm.Normal("y_obs", mu=theta, sigma=sigma, observed=y)
    trace = pm.sample(2000, tune=1000, chains=4)
```

**NumPyro**
```python
def model(sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfNormal(5))

    with numpyro.plate("schools", 8):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("y", dist.Normal(theta, sigma), obs=y)

mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0), sigma, y=y)
```

---

### When to Use Each Library

**minibayes** — Choose when:
- Deploying to edge/IoT devices with minimal dependencies
- Learning MCMC internals (transparent, readable code)
- Building simple to moderate models (d < 50 parameters)
- Working in resource-constrained environments
- Quick prototyping without heavy setup

**PyMC** — Choose when:
- Doing research and publication-quality work
- Building complex hierarchical models
- Analyzing time series data
- Need ArviZ ecosystem integration
- Want extensive documentation and community support

**NumPyro** — Choose when:
- Tackling high-dimensional problems requiring gradients
- GPU acceleration is needed
- Building deep probabilistic models
- Integrating with the JAX ecosystem
- Maximum sampling efficiency is critical
