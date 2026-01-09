"""Visualization functions for minibayes."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from minibayes.viz._utils import extract_samples, flatten_samples
from minibayes.viz.style import CHAIN_COLORS, PALETTE, STYLE_PARAMS

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from minibayes.results import InferenceResult


def _apply_style_to_fig(fig: Figure) -> None:
    """Apply style settings to a figure."""
    fig.set_facecolor(str(STYLE_PARAMS.get("figure.facecolor", "#FFFFFF")))
    dpi_val = STYLE_PARAMS.get("figure.dpi", 150)
    fig.set_dpi(int(dpi_val) if isinstance(dpi_val, (int, float)) else 150)


def _style_axes(ax: Axes) -> None:
    """Apply style settings to axes."""
    ax.set_facecolor(str(STYLE_PARAMS.get("axes.facecolor", "#FFFFFF")))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#4A4A4A", labelsize=9)
    ax.grid(True, color="#F0F0F0", linewidth=0.6, alpha=1.0)
    ax.set_axisbelow(True)


def plot_density(
    data: InferenceResult | dict[str, NDArray[np.float64]] | NDArray[np.float64],
    params: list[str] | None = None,
    ax: Axes | None = None,
    bins: int = 30,
    show_mean: bool = True,
) -> Figure:
    """
    Plot posterior density as histogram.

    Parameters
    ----------
    data : InferenceResult, dict, or ndarray
        MCMC samples. Shape (num_chains, num_samples) or 1D.
    params : list[str], optional
        Parameters to plot. None = all.
    ax : Axes, optional
        Existing axes. If None, creates new figure.
    bins : int
        Number of histogram bins.
    show_mean : bool
        Show vertical line at mean.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    samples = extract_samples(data, params)
    param_names = list(samples.keys())
    n_params = len(param_names)

    # Create figure if needed
    if ax is None:
        fig, axes_arr = plt.subplots(1, n_params, figsize=(4 * n_params, 3), squeeze=False)
        axes_flat: NDArray[np.object_] = axes_arr.flatten()
        axes_list: list[Axes] = [cast("Axes", a) for a in axes_flat]
        fig_out: Figure = cast("Figure", fig)
    else:
        fig_out = cast("Figure", ax.figure)
        axes_list = [ax]

    _apply_style_to_fig(fig_out)

    for i, name in enumerate(param_names):
        ax_i = axes_list[i] if i < len(axes_list) else axes_list[0]
        _style_axes(ax_i)
        arr = samples[name]
        n_chains = arr.shape[0]

        # Plot histogram for each chain
        for chain_idx in range(n_chains):
            chain_data: NDArray[np.float64] = arr[chain_idx, :]
            color = CHAIN_COLORS[chain_idx % len(CHAIN_COLORS)]
            alpha = 0.5 if n_chains > 1 else 0.7

            ax_i.hist(
                chain_data,
                bins=bins,
                density=True,
                alpha=alpha,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

        # Show mean
        if show_mean:
            flat = flatten_samples(arr)
            mean_val: float = cast("float", np.mean(flat))
            ax_i.axvline(mean_val, color=PALETTE["gray"], linestyle="--", linewidth=1.2, label=f"mean={mean_val:.3f}")

        ax_i.set_xlabel(name, fontsize=10, color="#4A4A4A")
        ax_i.set_ylabel("Density", fontsize=10, color="#4A4A4A")
        ax_i.set_title(f"Posterior: {name}", fontsize=11, color="#4A4A4A")
        if show_mean:
            ax_i.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    return fig_out


def plot_samples(
    data: InferenceResult | dict[str, NDArray[np.float64]] | NDArray[np.float64],
    params: list[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """
    Plot samples over iteration (convergence diagnostic).

    Parameters
    ----------
    data : InferenceResult, dict, or ndarray
        MCMC samples. Shape (num_chains, num_samples) or 1D.
    params : list[str], optional
        Parameters to plot. None = all.
    ax : Axes, optional
        Existing axes. If None, creates new figure.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    samples = extract_samples(data, params)
    param_names = list(samples.keys())
    n_params = len(param_names)

    # Create figure if needed
    if ax is None:
        fig, axes_arr = plt.subplots(n_params, 1, figsize=(8, 2.5 * n_params), squeeze=False)
        axes_flat: NDArray[np.object_] = axes_arr.flatten()
        axes_list: list[Axes] = [cast("Axes", a) for a in axes_flat]
        fig_out: Figure = cast("Figure", fig)
    else:
        fig_out = cast("Figure", ax.figure)
        axes_list = [ax]

    _apply_style_to_fig(fig_out)

    for i, name in enumerate(param_names):
        ax_i = axes_list[i] if i < len(axes_list) else axes_list[0]
        _style_axes(ax_i)
        arr = samples[name]
        n_chains = arr.shape[0]

        # Plot each chain
        for chain_idx in range(n_chains):
            chain_data: NDArray[np.float64] = arr[chain_idx, :]
            color = CHAIN_COLORS[chain_idx % len(CHAIN_COLORS)]
            label = f"Chain {chain_idx + 1}" if n_chains > 1 else None

            ax_i.plot(chain_data, alpha=0.7, linewidth=0.5, color=color, label=label)

        ax_i.set_xlabel("Iteration", fontsize=10, color="#4A4A4A")
        ax_i.set_ylabel(name, fontsize=10, color="#4A4A4A")
        ax_i.set_title(f"Samples: {name}", fontsize=11, color="#4A4A4A")
        if n_chains > 1:
            ax_i.legend(fontsize=8, loc="upper right", frameon=False)

    plt.tight_layout()
    return fig_out


def plot_forest(
    data: InferenceResult | dict[str, NDArray[np.float64]],
    params: list[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """
    Plot parameter distributions as horizontal box plots.

    Parameters
    ----------
    data : InferenceResult or dict of samples
        MCMC samples.
    params : list[str], optional
        Parameters to plot. None = all.
    ax : Axes, optional
        Existing axes. If None, creates new figure.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    samples = extract_samples(data, params)
    param_names = list(samples.keys())
    n_params = len(param_names)

    # Create figure if needed
    if ax is None:
        fig, ax_plot = plt.subplots(figsize=(7, max(2.5, 0.8 * n_params)))
        ax_plot = cast("Axes", ax_plot)
        fig_out: Figure = cast("Figure", fig)
    else:
        fig_out = cast("Figure", ax.figure)
        ax_plot = ax

    _apply_style_to_fig(fig_out)
    _style_axes(ax_plot)

    # Prepare data for boxplot - flatten all chains
    box_data: list[NDArray[np.float64]] = []
    for name in param_names:
        flat: NDArray[np.float64] = samples[name].flatten()
        box_data.append(flat)

    # Create horizontal box plots (no outliers)
    bp = ax_plot.boxplot(
        box_data,
        orientation="horizontal",
        patch_artist=True,
        tick_labels=param_names,
        widths=0.6,
        showfliers=False,
    )

    # Style the box plots with pastel colors
    for box in bp["boxes"]:
        box.set_facecolor(PALETTE["blue"])
        box.set_alpha(0.6)
        box.set_edgecolor(PALETTE["blue"])

    for whisker in bp["whiskers"]:
        whisker.set_color(PALETTE["gray"])
        whisker.set_linewidth(1.2)

    for cap in bp["caps"]:
        cap.set_color(PALETTE["gray"])
        cap.set_linewidth(1.2)

    for median in bp["medians"]:
        median.set_color(PALETTE["terracotta"])
        median.set_linewidth(2)

    # Create legend outside plot
    legend_elements = [
        Patch(facecolor=PALETTE["blue"], alpha=0.6, edgecolor=PALETTE["blue"], label="IQR (25-75%)"),
        Line2D([0], [0], color=PALETTE["terracotta"], linewidth=2, label="Median"),
    ]
    ax_plot.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)

    ax_plot.set_xlabel("Value", fontsize=10, color="#4A4A4A")
    ax_plot.set_title("Parameter Distributions", fontsize=11, color="#4A4A4A")

    plt.tight_layout(rect=(0, 0, 0.85, 1))
    return fig_out


def plot_predictive(
    x: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    x_obs: NDArray[np.float64] | None = None,
    y_obs: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
    credible_interval: float = 0.9,
) -> Figure:
    """
    Plot predictions with uncertainty bands.

    Parameters
    ----------
    x : ndarray
        X values for predictions. Shape (n_points,).
    y_pred : ndarray
        Predicted values. Shape (n_samples, n_points).
    x_obs : ndarray, optional
        X values for observed data. If None but y_obs provided, uses x.
    y_obs : ndarray, optional
        Observed y values to overlay.
    ax : Axes, optional
        Existing axes. If None, creates new figure.
    credible_interval : float
        Width of credible interval (0 to 1).

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    # Compute percentiles
    lower_pct = (1 - credible_interval) / 2 * 100
    upper_pct = (1 + credible_interval) / 2 * 100

    y_lower: NDArray[np.float64] = np.percentile(y_pred, lower_pct, axis=0)
    y_median: NDArray[np.float64] = np.percentile(y_pred, 50, axis=0)
    y_upper: NDArray[np.float64] = np.percentile(y_pred, upper_pct, axis=0)

    # Create figure if needed
    if ax is None:
        fig, ax_plot = plt.subplots(figsize=(8, 5))
        ax_plot = cast("Axes", ax_plot)
        fig_out: Figure = cast("Figure", fig)
    else:
        fig_out = cast("Figure", ax.figure)
        ax_plot = ax

    _apply_style_to_fig(fig_out)
    _style_axes(ax_plot)

    # Plot CI band
    ax_plot.fill_between(
        x,
        y_lower,
        y_upper,
        alpha=0.3,
        color=PALETTE["blue"],
        label=f"{int(credible_interval * 100)}% CI",
    )

    # Plot median line
    ax_plot.plot(x, y_median, color=PALETTE["blue"], linewidth=1.5, label="Median")

    # Plot observed data if provided
    if y_obs is not None:
        x_scatter = x_obs if x_obs is not None else x
        ax_plot.scatter(x_scatter, y_obs, color=PALETTE["terracotta"], s=30, alpha=0.8, label="Observed", zorder=3)

    ax_plot.set_xlabel("x", fontsize=10, color="#4A4A4A")
    ax_plot.set_ylabel("y", fontsize=10, color="#4A4A4A")
    ax_plot.set_title("Posterior Predictive", fontsize=11, color="#4A4A4A")
    ax_plot.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    return fig_out


def plot_autocorr(
    data: InferenceResult | dict[str, NDArray[np.float64]] | NDArray[np.float64],
    params: list[str] | None = None,
    ax: Axes | None = None,
    max_lag: int = 50,
) -> Figure:
    """
    Plot autocorrelation by lag.

    Parameters
    ----------
    data : InferenceResult, dict, or ndarray
        MCMC samples. Shape (num_chains, num_samples) or 1D.
    params : list[str], optional
        Parameters to plot. None = all.
    ax : Axes, optional
        Existing axes. If None, creates new figure.
    max_lag : int
        Maximum lag to show.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    samples = extract_samples(data, params)
    param_names = list(samples.keys())
    n_params = len(param_names)

    # Create figure if needed
    if ax is None:
        fig, axes_arr = plt.subplots(1, n_params, figsize=(4 * n_params, 3), squeeze=False)
        axes_flat: NDArray[np.object_] = axes_arr.flatten()
        axes_list: list[Axes] = [cast("Axes", a) for a in axes_flat]
        fig_out: Figure = cast("Figure", fig)
    else:
        fig_out = cast("Figure", ax.figure)
        axes_list = [ax]

    _apply_style_to_fig(fig_out)

    for i, name in enumerate(param_names):
        ax_i = axes_list[i] if i < len(axes_list) else axes_list[0]
        _style_axes(ax_i)
        arr = samples[name]

        # Use first chain for autocorrelation
        chain_data: NDArray[np.float64] = arr[0, :]

        # Compute autocorrelation
        acf = _compute_autocorr(chain_data, max_lag)
        lags: NDArray[np.int64] = np.arange(len(acf), dtype=np.int64)

        ax_i.bar(lags, acf, color=PALETTE["blue"], alpha=0.7, width=0.8)

        ax_i.axhline(0, color=PALETTE["gray"], linewidth=0.8)
        ax_i.set_xlabel("Lag", fontsize=10, color="#4A4A4A")
        ax_i.set_ylabel("Autocorrelation", fontsize=10, color="#4A4A4A")
        ax_i.set_title(f"Autocorrelation: {name}", fontsize=11, color="#4A4A4A")
        ax_i.set_ylim(-0.2, 1.0)

    plt.tight_layout()
    return fig_out


def _compute_autocorr(samples: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """Compute autocorrelation for a 1D sample array."""
    n = len(samples)
    max_lag = min(max_lag, n - 1)

    mean: float = cast("float", np.mean(samples))
    centered: NDArray[np.float64] = samples - mean
    var: float = cast("float", np.var(samples))

    if var <= 0:
        return np.zeros(max_lag + 1, dtype=np.float64)

    acf: list[float] = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            cov_val: float = cast("float", np.mean(centered[:-lag] * centered[lag:]))
            acf.append(cov_val / var)

    return np.asarray(acf, dtype=np.float64)


def summary_table(
    data: InferenceResult | dict[str, NDArray[np.float64]] | dict[str, dict[str, float]],
    params: list[str] | None = None,
    percentiles: list[int] | None = None,
) -> str:
    """
    Generate formatted summary table.

    Parameters
    ----------
    data : InferenceResult, dict of samples, or dict of summary stats
        MCMC samples or pre-computed summary statistics.
    params : list[str], optional
        Parameters to include. None = all.
    percentiles : list[int], optional
        Percentiles to show. Default: [5, 50, 95].

    Returns
    -------
    str
        Formatted ASCII table.

    Examples
    --------
    >>> print(summary_table(result))
    """
    from minibayes.results import InferenceResult

    if percentiles is None:
        percentiles = [5, 50, 95]

    # Compute summary if needed
    if isinstance(data, InferenceResult):
        summary = data.summary(percentiles=percentiles, params=params)
    elif isinstance(data, dict):
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            summary: dict[str, dict[str, float]] = data  # type: ignore[no-redef]
            if params is not None:
                summary = {k: v for k, v in summary.items() if k in params}
        else:
            samples_dict: dict[str, NDArray[np.float64]] = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    samples_dict[k] = v
            samples = extract_samples(samples_dict, params)
            from minibayes.diagnostics import summary as compute_summary

            summary = compute_summary(samples, percentiles)
    else:
        raise TypeError(f"Expected InferenceResult or dict, got {type(data)}")

    # Build table
    pct_headers = [f"{p}%" for p in percentiles]
    headers = ["param", "mean", "std"] + pct_headers + ["ess", "r_hat"]

    # Calculate column widths
    widths = [len(h) for h in headers]
    rows: list[list[str]] = []

    for name, stats in summary.items():
        row = [
            name,
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
        ]
        for p in percentiles:
            key = f"{p}%"
            row.append(f"{stats.get(key, 0.0):.4f}")
        row.append(f"{stats.get('ess', 0.0):.1f}")
        r_hat_val = stats.get("r_hat", float("nan"))
        row.append(f"{r_hat_val:.3f}" if not np.isnan(r_hat_val) else "n/a")
        rows.append(row)

        # Update widths
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    # Format table
    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * w for w in widths)

    lines = [
        fmt_row(headers),
        separator,
    ]
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)
