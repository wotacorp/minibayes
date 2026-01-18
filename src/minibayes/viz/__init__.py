"""
Visualization module for minibayes.

This module requires matplotlib to be installed.
Install with: pip install minibayes[viz]
"""

from __future__ import annotations

try:
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError as e:
    raise ImportError("minibayes.viz requires matplotlib.\nInstall with: pip install minibayes[viz]") from e

from minibayes.viz.plots import (
    plot_autocorr,
    plot_compare,
    plot_density,
    plot_distribution,
    plot_forest,
    plot_pair,
    plot_predictive,
    plot_samples,
    summary_table,
)
from minibayes.viz.style import CHAIN_COLORS, PALETTE, apply_style, style

__all__ = [
    # Plot functions
    "plot_density",
    "plot_samples",
    "plot_forest",
    "plot_pair",
    "plot_predictive",
    "plot_autocorr",
    "plot_compare",
    "plot_distribution",
    "summary_table",
    # Style
    "style",
    "apply_style",
    "PALETTE",
    "CHAIN_COLORS",
]
