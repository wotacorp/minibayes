# CLAUDE.md - Python Package Development

## Project Overview

A Python package with minimal dependencies (primarily numpy) called minibayes where overall spaces are described in minibayes-spec.md.

## Tech Stack

- **Language**: Python 3.11+
- **Core dependency**: numpy
- **Package manager**: u
- **Build backend**: setuptools or hatchling
- **Config**: pyproject.toml only (no setup.py, setup.cfg, requirements.txt)

## Project Structure

```
my_package/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── ...
├── notebooks/       # jupyter notebooks for exploration
└── tests/
    └── ...
```

Use src-layout for proper import isolation during development.

## Commands

```bash
# Install in dev mode
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Build package
uv build
```

## Code Standards

### Type Hints
- Required on all function signatures
- Use `numpy.typing` for array types: `NDArray[np.float64]`
- Avoid `Any` unless absolutely necessary

### Docstrings
- NumPy-style docstrings for all public functions/classes
- Include Parameters, Returns, Raises, Examples sections

### Style
- PEP 8, 88 char line length (ruff default)
- snake_case for functions/variables, PascalCase for classes
- Use f-strings for formatting
- Prefer comprehensions over loops where readable
- Use context managers for resources

### NumPy Specifics
- Prefer vectorized operations over loops
- Use broadcasting where applicable
- Document array shapes in docstrings: `shape (n, m)`
- Specify dtypes explicitly when it matters

### Strict mypy with NumPy

This project uses strict mypy (`disallow_any_expr=true`). NumPy operations often return generic types that mypy flags as `Any`. Use these patterns:

```python
# Pattern 1: Annotate all intermediate array variables
def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
    arr: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    result: NDArray[np.float64] = np.log(arr)
    return result

# Pattern 2: Use np.reciprocal() instead of 1/array (avoids Any)
denom: NDArray[np.float64] = 1 + np.exp(-arr)
result: NDArray[np.float64] = np.reciprocal(denom)  # Not: 1 / denom

# Pattern 3: Wrap scalar numpy results in float()
log_width: float = float(np.log(self._width))

# Pattern 4: Break complex expressions into typed intermediates
# Bad: np.log(x - low) + np.log(high - x) - np.log(width)
# Good:
log_low: NDArray[np.float64] = np.log(arr - self.low)
log_high: NDArray[np.float64] = np.log(self.high - arr)
log_width: float = float(np.log(self._width))
result: NDArray[np.float64] = log_low + log_high - log_width
```

## Example

```
def log_sum_exp(x: NDArray[np.float64]) -> float:
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    float
        log(sum(exp(x)))
    """
    x = np.asarray(x)
    max_val: float = float(np.max(x))
    if np.isinf(max_val) and max_val < 0:
        return float("-inf")
    shifted: NDArray[np.float64] = x - max_val
    exp_shifted: NDArray[np.float64] = np.exp(shifted)
    sum_exp: float = float(np.sum(exp_shifted))
    return max_val + float(np.log(sum_exp))
```

## Verification

Before committing:
1. `ruff check --fix` passes
2. `ruff format` applied
3. `mypy` passes (strict mode)
4. `pytest` passes

## Don'ts

- No mutable default arguments
- No bare `except:`
- No wildcard imports
- No hardcoded secrets (use .env)
- No commented-out code in commits

## Implementation Progress

**Update this section periodically as implementation proceeds.**

| Component | Status | Notes |
|-----------|--------|-------|
| **v0.1 Foundation** | | |
| `utils/numerical.py` | ✓ Done | `ensure_rng`, `check_finite`, `log_sum_exp` |
| `transforms/` | ✓ Done | Identity, Log, Logit, Affine + `default_transform()` |
| `distributions/` | ✓ Done | Normal, HalfNormal, Exponential, Gamma, Beta, Uniform |
| **v0.2 Core** | | |
| `model.py` | ✓ Done | Model class with transforms, Jacobian |
| `samplers/mh.py` | ✗ Skeleton | MetropolisHastings not implemented |
| `samplers/adaptive.py` | ✗ Skeleton | AdaptiveMetropolis not implemented |
| `results.py` | ◐ Partial | Dataclass done; methods skeleton |
| **v0.3 Interface** | | |
| `inference.py` | ✗ Skeleton | `sample()` entry point not implemented |
| `diagnostics.py` | ✗ Skeleton | ESS, R-hat not implemented |
| `utils/export.py` | ✗ Skeleton | save/load not implemented |

**Next steps:** Implement MH sampler, then Adaptive MH.
