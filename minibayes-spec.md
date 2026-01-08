# minibayes Specification v0.2

## Overview

**minibayes** is a minimal, lightweight Bayesian inference library designed for production deployment on resource-constrained environments (edge devices, containers) while remaining educational and hackable.

### Design Philosophy

- **Explicit over magic**: No context-dependent behavior, no global state, no effect handlers
- **Inspectable**: Users can see exactly what transforms are applied, what log_prob computes
- **Minimal but complete**: Cover 80% of use cases with 20% of complexity
- **Familiar patterns**: API inspired by NumPyro/PyMC but without the magic

### Goals

- **Minimal dependencies**: Python 3.10+ and NumPy only
- **Small footprint**: Single-digit MB installed size
- **Educational**: Code readable enough to learn from
- **Production-ready**: Deployable in containers on edge devices
- **No magic**: Every operation is an explicit method call

### Non-Goals

- Replacing PyMC/NumPyro/Stan for research
- Supporting every distribution or sampler
- Automatic differentiation (users provide gradients for HMC, or we use finite differences)
- GPU acceleration
- Hierarchical/multilevel priors (v1.0 limitation)

### Scope Limitations (v1.0)

**Priors must be independent.** This covers:
- Linear / logistic / Poisson regression
- Simple time series models
- A/B testing and conversion models
- Basic classification

This does NOT cover:
- Hierarchical/multilevel models (e.g., `mu ~ Normal(0, tau)` where `tau` is also a parameter)
- Gaussian processes
- Deep Bayesian networks

Users needing hierarchical models can use the direct `log_prob` API or switch to PyMC/NumPyro.

### Target Users

- ML engineers deploying Bayesian models to production
- Developers needing uncertainty quantification on edge devices
- Learners wanting to understand MCMC internals

---

## Architecture

```
minibayes/
├── __init__.py           # Public API exports
├── model.py              # Model class
├── distributions/
│   ├── __init__.py       # Distribution exports
│   ├── base.py           # Abstract base class + Support enum
│   ├── continuous.py     # Normal, HalfNormal, Beta, Gamma, etc.
│   └── discrete.py       # Bernoulli, Poisson, Categorical
├── samplers/
│   ├── __init__.py       # Sampler exports
│   ├── base.py           # Abstract sampler interface
│   ├── mh.py             # Random walk Metropolis-Hastings
│   ├── adaptive.py       # Adaptive Metropolis
│   └── hmc.py            # Hamiltonian Monte Carlo (v0.4+)
├── transforms.py         # Parameter transforms (log, logit, etc.)
├── diagnostics.py        # ESS, R-hat, trace plots
├── results.py            # InferenceResult class
├── export.py             # Save/load results
└── utils.py              # Numerical utilities
```

---

## Core Concepts

### Model API

minibayes uses a `Model` class to specify Bayesian models with priors and likelihood:

| Feature | Description |
|---------|-------------|
| Automatic transforms | Derived from distribution support |
| Jacobian correction | Handled automatically in `log_prob_unconstrained()` |
| Prior sampling | `sample_prior()` draws from joint prior |
| Explicit methods | No context-dependent magic |

For full control, users can access `model.log_prob_unconstrained()` directly or build custom samplers.

### No Context-Dependent Magic

Unlike NumPyro/Pyro, minibayes does NOT have a `sample()` primitive that behaves differently depending on context. Instead:

```python
# NumPyro: same code, different behavior depending on "handler"
def model():
    x = numpyro.sample("x", dist.Normal(0, 1))  # Magic!

# minibayes: explicit methods for each operation
model = mb.Model(priors={"x": dist.Normal(0, 1)}, likelihood=...)

model.sample_prior()      # Draw from prior
model.log_prior(params)   # Compute log prior
model.log_prob(params, data)  # Compute log posterior
```

Each operation is a separate, explicit method call. No surprises.

---

## API Reference

### Top-Level Functions

#### `mb.sample()`

The main inference entry point. Accepts a `Model` instance.

```python
def sample(
    model: Model,
    data: Any = None,
    initial: dict[str, float] | None = None,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    sampler: str = "adaptive_mh",
    sampler_kwargs: dict | None = None,
    seed: int | None = None,
) -> InferenceResult:
    """
    Run MCMC sampling.

    Parameters
    ----------
    model : Model
        A Model instance with priors and likelihood.
    data : Any
        Observed data passed to likelihood
    initial : dict, optional
        Initial parameter values (constrained space). If None, sampled from prior.
    num_samples : int
        Number of samples to draw (post-warmup)
    num_warmup : int
        Number of warmup/tuning samples (discarded)
    num_chains : int
        Number of independent chains
    sampler : str
        One of: "mh", "adaptive_mh", "hmc" (v0.4+)
    sampler_kwargs : dict, optional
        Additional arguments passed to sampler
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    InferenceResult
        Container with samples and diagnostics
    """
```

---

### Model Class

The structured way to define models. Priors and likelihood are specified separately.

```python
class Model:
    """
    A Bayesian model with explicit priors and likelihood.

    Priors are assumed independent (no hierarchical structure).
    Transforms are derived automatically from distribution support.

    Parameters
    ----------
    priors : dict[str, Distribution]
        Prior distributions for each parameter.
    likelihood : Callable[[dict, Any], float]
        Function (params, data) -> log_likelihood
    """

    def __init__(
        self,
        priors: dict[str, Distribution],
        likelihood: Callable[[dict[str, float], Any], float],
    ): ...

    # -------------------------------------------------------------------------
    # Explicit methods — no context-dependent magic
    # -------------------------------------------------------------------------

    def sample_prior(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """
        Draw one sample from the joint prior.

        Returns
        -------
        dict[str, float]
            Parameter values in constrained space
        """

    def log_prior(self, params: dict[str, float]) -> float:
        """
        Compute log prior probability.

        Parameters
        ----------
        params : dict
            Parameter values in constrained space

        Returns
        -------
        float
            Sum of log_prob for each prior
        """

    def log_likelihood(self, params: dict[str, float], data: Any) -> float:
        """
        Compute log likelihood.

        Parameters
        ----------
        params : dict
            Parameter values in constrained space
        data : Any
            Observed data

        Returns
        -------
        float
            Log likelihood value
        """

    def log_prob(self, params: dict[str, float], data: Any) -> float:
        """
        Compute unnormalized log posterior = log_prior + log_likelihood.

        Parameters
        ----------
        params : dict
            Parameter values in constrained space
        data : Any
            Observed data

        Returns
        -------
        float
            Log posterior (unnormalized)
        """

    # -------------------------------------------------------------------------
    # Transform handling — automatic but inspectable
    # -------------------------------------------------------------------------

    @property
    def transforms(self) -> dict[str, Transform]:
        """
        Get transforms for each parameter.

        Transforms are derived from distribution support:
        - REAL -> IdentityTransform
        - POSITIVE -> LogTransform
        - UNIT -> LogitTransform
        - BOUNDED -> AffineTransform

        Returns
        -------
        dict[str, Transform]
            Transform for each parameter
        """

    def to_unconstrained(self, params: dict[str, float]) -> dict[str, float]:
        """
        Transform constrained params to unconstrained space.

        Example: {"sigma": 2.0} -> {"sigma": 0.693}  (log transform)
        """

    def to_constrained(self, unconstrained: dict[str, float]) -> dict[str, float]:
        """
        Transform unconstrained params back to constrained space.

        Example: {"sigma": 0.693} -> {"sigma": 2.0}  (exp transform)
        """

    def log_prob_unconstrained(
        self,
        unconstrained: dict[str, float],
        data: Any,
    ) -> float:
        """
        Compute log_prob in unconstrained space.

        This is what the sampler actually calls. Includes Jacobian
        correction for the change of variables.

        Parameters
        ----------
        unconstrained : dict
            Parameter values in unconstrained space
        data : Any
            Observed data

        Returns
        -------
        float
            Log posterior with Jacobian correction
        """

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    @property
    def param_names(self) -> list[str]:
        """List of parameter names."""

    def validate_params(self, params: dict[str, float]) -> bool:
        """
        Check if params are valid (correct names, within support).

        Raises minibayesError with details if invalid.
        """
```

---

### Example: Bayesian Linear Regression

A complete example showing both APIs.

#### The Problem

```
Model:
    y = α + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
    ε ~ Normal(0, σ)

Priors:
    α ~ Normal(0, 10)
    βⱼ ~ Normal(0, 5)   for j = 1, ..., k
    σ ~ HalfNormal(5)
```

#### Using Model Class (Recommended)

```python
import minibayes as mb
from minibayes import dist
import numpy as np

# Data: n observations, k features
X = np.random.randn(100, 3)  # 100 observations, 3 features
true_alpha = 2.0
true_beta = np.array([1.0, -0.5, 0.3])
true_sigma = 0.8
y = true_alpha + X @ true_beta + np.random.normal(0, true_sigma, size=100)

# Define model
def likelihood(params, data):
    X, y = data
    mu = params["alpha"] + X @ np.array([params["beta_1"], params["beta_2"], params["beta_3"]])
    return dist.Normal(mu, params["sigma"]).log_prob(y).sum()

model = mb.Model(
    priors={
        "alpha": dist.Normal(0, 10),
        "beta_1": dist.Normal(0, 5),
        "beta_2": dist.Normal(0, 5),
        "beta_3": dist.Normal(0, 5),
        "sigma": dist.HalfNormal(5),
    },
    likelihood=likelihood,
)

# Inspect model (no magic — everything is explicit)
model.param_names                    # ['alpha', 'beta_1', 'beta_2', 'beta_3', 'sigma']
model.transforms                     # {'alpha': Identity, 'beta_*': Identity, 'sigma': Log}
model.sample_prior()                 # {'alpha': 3.2, 'beta_1': -1.1, ..., 'sigma': 2.3}
model.log_prior({"alpha": 0, "beta_1": 0, "beta_2": 0, "beta_3": 0, "sigma": 1})  # float

# Run inference
result = mb.sample(
    model=model,
    data=(X, y),
    num_samples=5000,
    num_warmup=1000,
    sampler="adaptive_mh",
)

# Results
result.summary()
#          mean    std     5%    50%    95%    ess  r_hat
# alpha    2.01   0.08   1.88   2.01   2.14   2800   1.00
# beta_1   0.98   0.09   0.84   0.98   1.13   2600   1.00
# beta_2  -0.51   0.08  -0.64  -0.51  -0.38   2700   1.00
# beta_3   0.31   0.08   0.18   0.31   0.44   2750   1.00
# sigma    0.79   0.06   0.70   0.79   0.89   2400   1.00

# Export for deployment
result.save("linear_regression_posterior.npz")
```

#### Direct Access for Power Users

For full control, users can access internal methods directly:

```python
# Access log_prob in unconstrained space (what samplers use internally)
unconstrained = model.to_unconstrained(params)
lp = model.log_prob_unconstrained(unconstrained, data)

# Or build a custom sampling loop
from minibayes.samplers import MetropolisHastings

sampler = MetropolisHastings(proposal_scale=0.5)
rng = np.random.default_rng(42)
state = model.to_unconstrained(model.sample_prior(rng))

for _ in range(1000):
    state, accepted = sampler.step(
        state,
        lambda p: model.log_prob_unconstrained(p, data),
        rng
    )
```

---

### InferenceResult

```python
@dataclass
class InferenceResult:
    """Container for MCMC results."""

    samples: dict[str, np.ndarray]
    # Samples for each parameter in CONSTRAINED space
    # Shape: (num_chains, num_samples) or (num_samples,) if num_chains=1

    samples_unconstrained: dict[str, np.ndarray]
    # Samples in unconstrained space (what sampler actually produced)

    acceptance_rate: float | np.ndarray
    # Acceptance rate(s) per chain

    num_samples: int
    num_warmup: int
    num_chains: int
    sampler: str
    elapsed_time: float  # seconds

    def summary(
        self,
        percentiles: list[int] = [5, 50, 95],
        params: list[str] | None = None,
    ) -> dict:
        """
        Compute summary statistics.

        Returns dict with keys: mean, std, percentiles, ess, r_hat
        """

    def to_dict(self) -> dict:
        """Convert to plain dict (for JSON serialization)."""

    def save(self, path: str, format: str = "npz") -> None:
        """
        Save results to file.

        Formats:
        - "npz": NumPy compressed archive (smallest)
        - "json": JSON (portable, human-readable)
        """

    @classmethod
    def load(cls, path: str) -> "InferenceResult":
        """Load results from file."""
```

---

### Distributions

#### Base Class

```python
class Distribution(ABC):
    """Abstract base class for probability distributions."""

    @property
    @abstractmethod
    def support(self) -> Support:
        """
        Return support of distribution.

        Used to determine automatic transform:
        - REAL -> IdentityTransform
        - POSITIVE -> LogTransform
        - UNIT -> LogitTransform
        - BOUNDED -> AffineTransform
        """

    @abstractmethod
    def log_prob(self, x: np.ndarray | float) -> np.ndarray | float:
        """Compute log probability density/mass at x."""

    @abstractmethod
    def sample(
        self,
        size: int | tuple[int, ...] | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | float:
        """Draw random samples."""

    def default_transform(self) -> Transform:
        """
        Return appropriate transform for this distribution's support.

        Override in subclass if non-standard transform is preferred.
        """
```

#### Support Enum

```python
class Support(Enum):
    REAL = "real"              # (-∞, +∞) -> IdentityTransform
    POSITIVE = "positive"      # (0, +∞)  -> LogTransform
    UNIT = "unit"              # (0, 1)   -> LogitTransform
    BOUNDED = "bounded"        # (a, b)   -> AffineTransform
    SIMPLEX = "simplex"        # Σxᵢ = 1  -> StickBreakingTransform (v2.0+)
```

#### Continuous Distributions

```python
class Normal(Distribution):
    """Normal (Gaussian) distribution."""
    support = Support.REAL

    def __init__(self, loc: float = 0.0, scale: float = 1.0): ...

class HalfNormal(Distribution):
    """Half-normal distribution (positive reals)."""
    support = Support.POSITIVE

    def __init__(self, scale: float = 1.0): ...

class Exponential(Distribution):
    """Exponential distribution."""
    support = Support.POSITIVE

    def __init__(self, rate: float = 1.0): ...

class Uniform(Distribution):
    """Uniform distribution on [low, high]."""
    support = Support.BOUNDED

    def __init__(self, low: float = 0.0, high: float = 1.0): ...

class Beta(Distribution):
    """Beta distribution on (0, 1)."""
    support = Support.UNIT

    def __init__(self, alpha: float = 1.0, beta: float = 1.0): ...

class Gamma(Distribution):
    """Gamma distribution."""
    support = Support.POSITIVE

    def __init__(self, shape: float = 1.0, rate: float = 1.0): ...

class LogNormal(Distribution):
    """Log-normal distribution."""
    support = Support.POSITIVE

    def __init__(self, loc: float = 0.0, scale: float = 1.0): ...

class StudentT(Distribution):
    """Student's t distribution."""
    support = Support.REAL

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0): ...
```

#### Discrete Distributions

```python
class Bernoulli(Distribution):
    """Bernoulli distribution."""
    support = Support.BINARY  # {0, 1}

    def __init__(self, probs: float): ...

class Poisson(Distribution):
    """Poisson distribution."""
    support = Support.NATURAL  # {0, 1, 2, ...}

    def __init__(self, rate: float): ...
```

---

### Transforms

```python
class Transform(ABC):
    """Bijective transform for constrained parameters."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Transform from constrained to unconstrained space."""

    @abstractmethod
    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Transform from unconstrained to constrained space."""

    @abstractmethod
    def log_det_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Log absolute determinant of Jacobian (for constrained x)."""


class IdentityTransform(Transform):
    """No transformation. For REAL support."""

class LogTransform(Transform):
    """
    Log transform for positive parameters.

    forward: y = log(x)      [constrained -> unconstrained]
    inverse: x = exp(y)      [unconstrained -> constrained]
    log_det_jacobian: log(x) = y
    """

class LogitTransform(Transform):
    """
    Logit transform for (0, 1) parameters.

    forward: y = log(x / (1-x))
    inverse: x = 1 / (1 + exp(-y))
    """

class AffineTransform(Transform):
    """
    Affine transform for bounded parameters.

    Maps (low, high) to (-∞, +∞) via scaled logit.
    """
    def __init__(self, low: float, high: float): ...
```

---

### Samplers

#### Base Class

```python
class Sampler(ABC):
    """Abstract base class for MCMC samplers."""

    @abstractmethod
    def step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable[[dict[str, float]], float],
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], bool]:
        """
        Take one MCMC step.

        Parameters
        ----------
        current : dict
            Current parameter values (unconstrained space)
        log_prob_fn : Callable
            Function params -> log_prob (in unconstrained space)
        rng : Generator
            NumPy random generator

        Returns
        -------
        new_state : dict
            New parameter values
        accepted : bool
            Whether proposal was accepted
        """

    @abstractmethod
    def warmup_step(
        self,
        current: dict[str, float],
        log_prob_fn: Callable,
        rng: np.random.Generator,
        step_num: int,
    ) -> tuple[dict[str, float], bool]:
        """
        Take one warmup step (may adapt internal state).
        """
```

#### MetropolisHastings

```python
class MetropolisHastings(Sampler):
    """Random walk Metropolis-Hastings sampler."""

    def __init__(
        self,
        proposal_scale: float | dict[str, float] = 1.0,
    ):
        """
        Parameters
        ----------
        proposal_scale : float or dict
            Standard deviation of Gaussian proposal.
            If dict, specifies per-parameter scales.
        """
```

#### AdaptiveMetropolis

```python
class AdaptiveMetropolis(Sampler):
    """
    Adaptive Metropolis with covariance tuning.

    During warmup, adapts proposal covariance based on sample history.
    Uses the 2.38²/d scaling factor (optimal for Gaussian targets).
    """

    def __init__(
        self,
        initial_scale: float = 1.0,
        target_acceptance: float = 0.234,
    ):
        """
        Parameters
        ----------
        initial_scale : float
            Initial proposal scale before adaptation
        target_acceptance : float
            Target acceptance rate (0.234 optimal for Gaussians)
        """
```

#### HamiltonianMonteCarlo (v0.4+)

```python
class HamiltonianMonteCarlo(Sampler):
    """
    Hamiltonian Monte Carlo sampler.

    Requires gradients of log_prob. If not provided, uses finite differences.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        num_leapfrog_steps: int = 10,
        grad_log_prob: Callable | None = None,
    ):
        """
        Parameters
        ----------
        step_size : float
            Leapfrog integrator step size
        num_leapfrog_steps : int
            Number of leapfrog steps per proposal
        grad_log_prob : Callable, optional
            Gradient function. If None, uses finite differences.
        """
```

---

### Diagnostics

```python
def effective_sample_size(samples: np.ndarray) -> float:
    """
    Compute effective sample size accounting for autocorrelation.
    """

def r_hat(chains: np.ndarray) -> float:
    """
    Compute Gelman-Rubin R-hat diagnostic.

    Values > 1.01 indicate non-convergence.
    """

def summary(result: InferenceResult) -> dict:
    """
    Compute summary statistics for all parameters.

    Returns dict with:
    - mean, std per parameter
    - percentiles (5%, 50%, 95%)
    - ess (effective sample size)
    - r_hat (if multiple chains)
    """
```

---

### Error Handling

```python
class minibayesError(Exception):
    """Base exception for minibayes."""

class SamplingError(minibayesError):
    """Raised when sampling fails."""

class ConvergenceWarning(UserWarning):
    """Raised when diagnostics suggest non-convergence."""

class NumericalError(minibayesError):
    """Raised on numerical issues (NaN, Inf in log_prob)."""

class ModelSpecError(minibayesError):
    """Raised when model is mis-specified."""
```

---

## Development Roadmap

### v0.1 — "It samples"
- [ ] Core distributions: Normal, HalfNormal, Exponential, Beta, Gamma, Uniform
- [ ] Transforms: Identity, Log, Logit
- [ ] MetropolisHastings sampler
- [ ] Direct log_prob interface
- [ ] Basic InferenceResult

### v0.2 — "It's usable"
- [ ] Model class with priors + likelihood
- [ ] Automatic transforms from distribution support
- [ ] AdaptiveMetropolis sampler
- [ ] Multiple chains
- [ ] save/load to npz

### v0.3 — "It's pleasant"
- [ ] Diagnostics: ESS, R-hat
- [ ] summary() method
- [ ] JSON export
- [ ] StudentT, LogNormal, Bernoulli, Poisson distributions
- [ ] Bounded parameter transforms

### v0.4 — "It's fast"
- [ ] HMC sampler (finite difference gradients)
- [ ] Numba JIT for hot loops (optional dependency)
- [ ] Warmup adaptation for HMC

### v1.0 — "It's production-ready"
- [ ] Comprehensive numerical testing
- [ ] Detailed error messages
- [ ] Documentation
- [ ] Benchmarks vs NumPyro

### Future (v2.0+)
- Hierarchical priors (dependent prior structure)
- NUTS sampler
- User-provided gradients
- Rust core with PyO3 bindings
- Vector parameters

---

## Testing Strategy

### Test Cases

1. **Normal-Normal**: Analytical posterior known — verify sampler matches
2. **Beta-Binomial**: Analytical posterior known
3. **Linear regression**: Compare to scipy OLS point estimates
4. **Transform correctness**: Verify Jacobians are correct
5. **Convergence**: Verify R-hat < 1.01 on well-specified models

### Test Structure

```
tests/
├── test_distributions.py   # log_prob against scipy.stats
├── test_transforms.py      # forward/inverse roundtrip, Jacobians
├── test_model.py           # Model class methods
├── test_samplers.py        # Samplers on known posteriors
├── test_inference.py       # End-to-end mb.sample()
├── test_diagnostics.py     # ESS, R-hat formulas
├── test_export.py          # Save/load roundtrip
└── benchmarks/
    └── bench_samplers.py   # Performance comparison
```

---

## References

### Code to Study
- [MiniBM](https://github.com/facebookresearch/beanmachine/blob/main/minibm/minibm.py) — ~100 line PPL
- [BlackJAX](https://github.com/blackjax-devs/blackjax/tree/main/blackjax/mcmc) — Clean functional samplers
- [NumPyro](https://github.com/pyro-ppl/numpyro/tree/master/numpyro/infer) — Production-quality MCMC
- [minimc](https://github.com/ColCarroll/minimc) — Educational HMC

### Papers
- Andrieu & Thoms (2008) "A tutorial on adaptive MCMC"
- Hoffman & Gelman (2014) "The No-U-Turn Sampler"

### Books
- Gelman et al. "Bayesian Data Analysis" (BDA3)
- McElreath "Statistical Rethinking"
