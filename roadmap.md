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
