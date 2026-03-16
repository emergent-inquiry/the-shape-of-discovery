"""Shared visualization style and helper functions.

All figures: 300 DPI, figsize=(12, 7), colorblind-friendly palette.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.utils import FIGURES_DIR, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011, Nature Methods)
# ---------------------------------------------------------------------------

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

PALETTE_LIST = list(PALETTE.values())

# CPC section colors (8 sections A-H)
CPC_COLORS = {
    "A": PALETTE["blue"],       # Human Necessities
    "B": PALETTE["orange"],     # Operations; Transporting
    "C": PALETTE["green"],      # Chemistry; Metallurgy
    "D": PALETTE["yellow"],     # Textiles; Paper
    "E": PALETTE["red"],        # Fixed Constructions
    "F": PALETTE["purple"],     # Mechanical Engineering
    "G": PALETTE["cyan"],       # Physics
    "H": PALETTE["black"],      # Electricity
}

CPC_LABELS = {
    "A": "Human Necessities",
    "B": "Operations & Transport",
    "C": "Chemistry & Metallurgy",
    "D": "Textiles & Paper",
    "E": "Fixed Constructions",
    "F": "Mech. Engineering",
    "G": "Physics",
    "H": "Electricity",
}


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def set_style() -> None:
    """Apply the project's standard matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    name: str,
    subdir: Optional[str] = None,
) -> Path:
    """Save a figure to the figures directory.

    Args:
        fig: Matplotlib Figure object.
        name: Filename without extension.
        subdir: Optional subdirectory under figures/.

    Returns:
        Path to the saved figure.
    """
    dest = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    dest.mkdir(parents=True, exist_ok=True)
    fp = dest / f"{name}.png"
    fig.savefig(fp)
    plt.close(fig)
    logger.info("Saved figure: %s", fp)
    return fp


# ---------------------------------------------------------------------------
# Common axis formatters
# ---------------------------------------------------------------------------

def year_axis(ax: plt.Axes, start: int = 1980, end: int = 2023) -> None:
    """Format x-axis for year-based time series.

    Args:
        ax: Matplotlib Axes to format.
        start: First year.
        end: Last year.
    """
    ax.set_xlim(start - 0.5, end + 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))


def log_log_axes(ax: plt.Axes) -> None:
    """Set both axes to log scale with clean formatting.

    Args:
        ax: Matplotlib Axes to format.
    """
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)


# ---------------------------------------------------------------------------
# Reusable plot components
# ---------------------------------------------------------------------------

def time_series_plot(
    years: np.ndarray,
    values: np.ndarray,
    label: str,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot a single time series with standard styling.

    Args:
        years: Array of year values.
        values: Array of metric values.
        label: Legend label.
        ax: Existing axes (creates new figure if None).
        color: Line color.
        ylabel: Y-axis label.
        title: Plot title.

    Returns:
        The Axes object.
    """
    if ax is None:
        set_style()
        _, ax = plt.subplots()

    ax.plot(years, values, label=label, color=color or PALETTE["blue"], linewidth=2)
    year_axis(ax)

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def confidence_band(
    ax: plt.Axes,
    years: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: str = PALETTE["blue"],
    alpha: float = 0.15,
    label: Optional[str] = None,
) -> None:
    """Add a confidence/credible interval band to a time series.

    Args:
        ax: Axes to plot on.
        years: X values.
        lower: Lower bound.
        upper: Upper bound.
        color: Fill color.
        alpha: Fill transparency.
        label: Legend label for the band.
    """
    ax.fill_between(years, lower, upper, color=color, alpha=alpha, label=label)
