# minibayes

A lightweight library for Bayesian inference designed for production deployment on resource-constrained environments.

## Installation

```bash
pip install minibayes
```

## Development Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/theoradusz/minibayes.git
cd minibayes

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Development Commands

```bash
# Run tests
uv run pytest

# Type checking
uv run mypy src

# Linting and formatting
uv run ruff check src tests
uv run ruff format src tests
```
