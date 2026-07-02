"""SHAP based interpretability for the fitted hazard pipelines.

The central artifact is the class contribution table: the mean absolute SHAP
value of every feature toward every risk class on the holdout partition,
normalized so the grand total is one hundred percent. It answers the question
the study cares about most: which attributes carry the specialist judgment
that the model reproduces.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .config import HAZARDS, RISK_LEVELS
from .plots import INK, INK_SOFT, RISK_COLORS, SURFACE, DIV_CMAP


def shap_values_by_class(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Compute SHAP values on encoded features.

    Returns an array shaped (samples, features, classes), the encoded feature
    frame, and the class order used by the model.
    """
    encoder = pipeline.named_steps["encoder"]
    model = pipeline.named_steps["model"]
    X_encoded = encoder.transform(X)
    classes = [str(c) for c in model.classes_]

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_encoded)
    except Exception:
        background = shap.sample(X_encoded, min(60, len(X_encoded)), random_state=0)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        values = explainer.shap_values(X_encoded, silent=True)

    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    if values.ndim != 3:
        raise ValueError(f"Unexpected SHAP output with shape {values.shape}")
    return values, X_encoded, classes


def class_contribution_table(
    values: np.ndarray, feature_names: list[str], classes: list[str]
) -> pd.DataFrame:
    """Mean absolute SHAP per feature and class, normalized to a 100% total."""
    mean_abs = np.abs(values).mean(axis=0)
    normalized = mean_abs / mean_abs.sum() * 100.0
    table = pd.DataFrame(normalized, index=feature_names, columns=classes)
    table = table[[c for c in RISK_LEVELS if c in table.columns]]
    table["Total"] = table.sum(axis=1)
    return table.sort_values("Total", ascending=False)


def explain_hazard(pipeline: Pipeline, X: pd.DataFrame, hazard: str) -> dict:
    """Full SHAP bundle for one hazard: raw values plus the contribution table."""
    schema = HAZARDS[hazard]
    values, X_encoded, classes = shap_values_by_class(pipeline, X)
    table = class_contribution_table(values, schema.feature_names, classes)
    return {
        "values": values,
        "X_encoded": X_encoded,
        "classes": classes,
        "table": table,
    }


def contribution_figure(table: pd.DataFrame, hazard: str) -> plt.Figure:
    """Stacked horizontal bars of the normalized SHAP contribution per class.

    Each bar totals one feature's share of the model's decision signal; the
    stack splits that share across the three risk classes with the fixed
    status hues. Totals are labeled at the bar ends and segments above a small
    threshold carry their own percentage.
    """
    schema = HAZARDS[hazard]
    data = table.drop(columns=["Total"]).copy()
    data.index = [schema.display_name(f) for f in data.index]
    totals = data.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(data) + 1.7))
    left = np.zeros(len(data))
    for level in [c for c in RISK_LEVELS if c in data.columns]:
        widths = data[level].values
        ax.barh(
            data.index, widths, left=left, color=RISK_COLORS[level],
            label=level, height=0.62, edgecolor=SURFACE, linewidth=1.2,
        )
        for y, (x0, w) in enumerate(zip(left, widths)):
            if w >= 2.6:
                ink = SURFACE if level != "Medium" else INK
                ax.text(x0 + w / 2, y, f"{w:.1f}", ha="center", va="center",
                        fontsize=7.8, color=ink)
        left += widths

    for y, total in enumerate(totals):
        ax.text(total + 0.5, y, f"{total:.1f}%", va="center", ha="left",
                fontsize=8.6, color=INK_SOFT, fontweight="semibold")

    ax.invert_yaxis()
    ax.set_xlim(0, totals.max() * 1.22)
    ax.set_xlabel("Share of total decision signal (%)")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(title="Risk class", loc="lower right")
    ax.set_title(
        f"{schema.display}: what drives the specialist judgment", pad=12
    )
    fig.tight_layout()
    return fig


def beeswarm_figure(
    bundle: dict, hazard: str, target_class: str = "High"
) -> plt.Figure:
    """SHAP beeswarm for one output class.

    Point position is the signed contribution toward the target class; point
    color is the encoded attribute severity on the blue to red diverging ramp.
    """
    schema = HAZARDS[hazard]
    class_idx = bundle["classes"].index(target_class)
    display_frame = bundle["X_encoded"].copy()
    display_frame.columns = [schema.display_name(f) for f in schema.feature_names]

    fig = plt.figure(figsize=(8.0, 0.38 * len(schema.feature_names) + 1.6))
    shap.summary_plot(
        bundle["values"][:, :, class_idx],
        display_frame,
        cmap=DIV_CMAP,
        show=False,
        plot_size=None,
    )
    ax = plt.gca()
    ax.set_xlabel(f"Contribution toward the {target_class} class", fontsize=9.5,
                  color=INK_SOFT)
    ax.tick_params(labelsize=9)
    ax.set_title(
        f"{schema.display}: per building contributions ({target_class} risk)",
        fontsize=11, color=INK, fontweight="semibold", pad=12,
    )
    if fig.axes and len(fig.axes) > 1:
        cbar = fig.axes[-1]
        cbar.set_ylabel("Attribute severity (encoded)", fontsize=8.5,
                        color=INK_SOFT)
        cbar.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def local_contributions(
    pipeline: Pipeline, record: pd.DataFrame, hazard: str, target_class: str
) -> pd.DataFrame:
    """Signed SHAP contributions of one building toward one class."""
    schema = HAZARDS[hazard]
    values, _, classes = shap_values_by_class(pipeline, record)
    class_idx = classes.index(target_class)
    contrib = pd.DataFrame(
        {
            "feature": [schema.display_name(f) for f in schema.feature_names],
            "value": [record.iloc[0][f] for f in schema.feature_names],
            "shap": values[0, :, class_idx],
        }
    )
    return contrib.reindex(
        contrib["shap"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
