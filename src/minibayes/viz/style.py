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

"""Minimalist pastel style for minibayes visualizations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

# Pastel color palette - diverse, distinct colors
PALETTE: dict[str, str] = {
    "blue": "#88B4D4",  # Pale blue
    "terracotta": "#C97B63",  # Terracotta
    "sage": "#8FB996",  # Sage green
    "pink": "#E8B4BC",  # Pastel pink
    "lavender": "#A89CC8",  # Lavender
    "mustard": "#D4B86A",  # Muted gold
    "sand": "#C9BDA8",  # Warm sand
    "gray": "#8E9AAB",  # Cool slate
}

# Multi-chain colors (10 distinct pastels)
CHAIN_COLORS: list[str] = [
    "#88B4D4",  # Pale blue
    "#C97B63",  # Terracotta
    "#8FB996",  # Sage green
    "#E8B4BC",  # Pastel pink
    "#A89CC8",  # Lavender
    "#D4B86A",  # Muted gold
    "#6BAAAA",  # Teal
    "#E9C46A",  # Warm yellow
    "#C9A0DC",  # Orchid
    "#7DAFA5",  # Seafoam
]

# Style parameters for rcParams (typed as object to avoid Any)
STYLE_PARAMS: dict[str, object] = {
    # Figure
    "figure.facecolor": "#FFFFFF",
    "figure.dpi": 150,
    "figure.figsize": (8, 5),
    # Axes
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",
    "axes.labelcolor": "#4A4A4A",
    "axes.titlecolor": "#4A4A4A",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Grid
    "grid.color": "#F0F0F0",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    # Lines
    "lines.linewidth": 1.2,
    "lines.solid_capstyle": "round",
    # Ticks
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Legend
    "legend.frameon": False,
    "legend.fontsize": 9,
    # Font
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
}


@contextmanager
def style() -> Generator[None, None, None]:
    """
    Apply minibayes style temporarily.

    Use as a context manager to apply the minibayes pastel style
    to plots within the block.

    Examples
    --------
    >>> with style():
    ...     plt.plot(x, y)
    ...     plt.show()
    """
    import matplotlib.pyplot as plt

    original: dict[str, object] = {}
    for key in STYLE_PARAMS:
        original[key] = plt.rcParams[key]
    try:
        for key, val in STYLE_PARAMS.items():
            plt.rcParams[key] = val
        yield
    finally:
        for key, val in original.items():
            plt.rcParams[key] = val


def apply_style() -> None:
    """
    Apply minibayes style globally.

    Call this once at the start of your script to apply the style
    to all subsequent plots.
    """
    import matplotlib.pyplot as plt

    for key, val in STYLE_PARAMS.items():
        plt.rcParams[key] = val
