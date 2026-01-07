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
