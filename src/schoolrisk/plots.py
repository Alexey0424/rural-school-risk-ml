"""Shared figure styling and the report figures.

Color assignments follow a small set of rules. Risk classes always wear the
same three validated status hues with direct labels. Magnitudes wear a single
blue ramp. Signed effects wear a blue to red diverging ramp with a neutral
midpoint. Text never wears a series color.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .config import FIGURES_DIR, HAZARDS, RISK_LEVELS

INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

#: Fixed status hues for the three risk classes (CVD validated as a set).
RISK_COLORS = {"Low": "#006300", "Medium": "#eda100", "High": "#d03b3b"}

#: Fixed order categorical palette for everything that is not a risk class.
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#2a78d6", "#f0efec", "#e34948"]
)


def apply_style() -> None:
    """Apply the shared matplotlib style. Call once per session."""
    mpl.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.labelcolor": INK_SOFT,
            "axes.labelsize": 9.5,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9.5,
        }
    )


def save_figure(fig: plt.Figure, name: str, figures_dir: Path | None = None) -> Path:
    out = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def class_balance_figure(datasets: dict[str, pd.DataFrame]) -> plt.Figure:
    """Small multiples of class counts per hazard, one bar per risk level."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.9 * len(datasets), 3.0))
    axes = np.atleast_1d(axes)
    for ax, (hazard, df) in zip(axes, datasets.items()):
        counts = (
            df["risk_level"].value_counts().reindex(list(RISK_LEVELS)).fillna(0)
        )
        bars = ax.bar(
            counts.index,
            counts.values,
            width=0.62,
            color=[RISK_COLORS[c] for c in counts.index],
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(counts.values) * 0.02,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK_SOFT,
            )
        ax.set_title(f"{HAZARDS[hazard].display}  (n = {len(df)})")
        ax.set_ylim(0, max(counts.values) * 1.18)
        ax.tick_params(axis="x", length=0)
        ax.set_yticks([])
        for spine in ("left",):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Risk level distribution per hazard", y=1.04, fontsize=12,
        fontweight="semibold", color=INK,
    )
    fig.tight_layout()
    return fig


def cv_comparison_figure(
    tables: dict[str, pd.DataFrame], metric: str = "f1_macro"
) -> plt.Figure:
    """Small multiples of the cross validated model comparison per hazard.

    A single blue carries the magnitude; the winning model is emphasized with
    the darkest ramp step and a direct value label on every bar.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    for ax, (hazard, table) in zip(axes.ravel(), tables.items()):
        data = table.sort_values(metric, ascending=True)
        best_key = data.iloc[-1]["key"]
        colors = ["#104281" if k == best_key else "#5598e7" for k in data["key"]]
        bars = ax.barh(
            data["model"],
            data[metric],
            xerr=data[f"{metric}_std"],
            error_kw={"ecolor": BASELINE, "elinewidth": 1.1, "capsize": 0},
            color=colors,
            height=0.62,
            edgecolor=SURFACE,
            linewidth=1.0,
        )
        for bar, value in zip(bars, data[metric]):
            ax.text(
                min(value + 0.012, 1.11),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=8,
                color=INK_SOFT,
            )
        ax.set_title(HAZARDS[hazard].display)
        ax.set_xlim(0, 1.13)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "Ten fold cross validated macro F1 on the training partition",
        y=1.0, fontsize=12.5, fontweight="semibold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def confusion_grid_figure(results: dict[str, dict]) -> plt.Figure:
    """Confusion matrices for every hazard on the holdout partition.

    Shading encodes the row normalized share on the blue ramp; the annotation
    shows the raw count so the figure stays readable without the color.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(3.1 * len(results), 3.4))
    axes = np.atleast_1d(axes)
    labels = list(RISK_LEVELS)
    for ax, (hazard, res) in zip(axes, results.items()):
        cm = np.asarray(res["confusion_matrix"], dtype=float)
        row_share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(row_share, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ink = SURFACE if row_share[i, j] > 0.55 else INK
                ax.text(
                    j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=10, color=ink,
                )
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * 3)
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("Expert rating")
        acc = res["metrics"]["accuracy"]
        ax.set_title(f"{HAZARDS[hazard].display}  (accuracy {acc:.2f})")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Holdout confusion matrices", y=1.06, fontsize=12.5,
        fontweight="semibold", color=INK,
    )
    fig.tight_layout()
    return fig


def risk_distribution_figure(
    distributions: dict[str, pd.Series], title: str, subtitle: str | None = None
) -> plt.Figure:
    """Horizontal 100 percent stacked bars of predicted risk shares per hazard."""
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(distributions) + 1.6))
    names = [HAZARDS[h].display for h in distributions]
    left = np.zeros(len(distributions))
    for level in RISK_LEVELS:
        shares = np.array(
            [d.get(level, 0) / d.sum() * 100 for d in distributions.values()]
        )
        ax.barh(
            names, shares, left=left, color=RISK_COLORS[level], label=level,
            height=0.58, edgecolor=SURFACE, linewidth=1.5,
        )
        for y, (x0, w) in enumerate(zip(left, shares)):
            if w >= 6:
                ink = SURFACE if level != "Medium" else INK
                ax.text(
                    x0 + w / 2, y, f"{w:.0f}%", ha="center", va="center",
                    fontsize=8.5, color=ink,
                )
        left += shares
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of buildings (%)")
    ax.tick_params(axis="y", length=0)
    ax.legend(title="Predicted risk", loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncols=3)
    ax.set_title(title, pad=12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK_SOFT)
    fig.tight_layout()
    return fig
