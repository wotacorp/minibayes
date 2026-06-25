# Copyright 2026 WOTA CORP.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    plot_ppc,
    plot_predictive,
    plot_prior_posterior,
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
    "plot_prior_posterior",
    "plot_ppc",
    "summary_table",
    # Style
    "style",
    "apply_style",
    "PALETTE",
    "CHAIN_COLORS",
]
