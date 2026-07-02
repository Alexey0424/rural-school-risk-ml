"""Render the methodology overview figure used in the README."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schoolrisk.plots import BASELINE, INK, INK_SOFT, SURFACE, apply_style, save_figure

STEPS = [
    ("Expert rated survey", "rural schools, 4 hazards,\n3 risk levels"),
    ("Severity ordered\nencoding", "ordinal categories,\nno scaling"),
    ("Model tournament", "11 algorithms,\nten fold stratified CV"),
    ("Randomized tuning", "tree ensemble finalists,\n40 draws each"),
    ("Holdout audit", "untouched 20 percent,\nmacro metrics"),
    ("Explain and deploy", "SHAP attributions,\napp and scenarios"),
]

ACCENT = "#2a78d6"


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(13.2, 2.9))
    ax.set_xlim(0, len(STEPS))
    ax.set_ylim(0, 1)
    ax.axis("off")

    width, height, y0 = 0.86, 0.72, 0.12
    for i, (title, caption) in enumerate(STEPS):
        x0 = i + (1 - width) / 2
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0), width, height,
                boxstyle="round,pad=0.012,rounding_size=0.035",
                facecolor=SURFACE, edgecolor=BASELINE, linewidth=1.1,
            )
        )
        ax.text(x0 + width / 2, y0 + height - 0.055, f"{i + 1}",
                fontsize=9.5, fontweight="bold", color=ACCENT,
                ha="center", va="top")
        ax.text(x0 + width / 2, y0 + height - 0.175, title, fontsize=9.6,
                fontweight="semibold", color=INK, ha="center", va="top",
                linespacing=1.25)
        ax.text(x0 + width / 2, y0 + 0.145, caption, fontsize=7.8,
                color=INK_SOFT, ha="center", va="center", linespacing=1.35)
        if i < len(STEPS) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (i + 1 - (1 - width) / 2 + 0.008, y0 + height / 2),
                    (i + 1 + (1 - width) / 2 - 0.008, y0 + height / 2),
                    arrowstyle="-|>", mutation_scale=13,
                    color=ACCENT, linewidth=1.4,
                )
            )
    save_figure(fig, "pipeline_overview")
    print("wrote reports/figures/pipeline_overview.png")


if __name__ == "__main__":
    main()
