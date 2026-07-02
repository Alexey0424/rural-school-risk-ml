"""Scenario generation labeled by the trained models.

The generator draws random building configurations, keeps only those that a
structural specialist would accept as physically plausible, and labels them
with a trained pipeline. It powers three things: the versioned demo dataset,
the scenario study notebook, and quick what if experiments such as measuring
how the predicted risk shifts after a retrofit intervention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import HAZARDS

#: Systems that plausibly reach three or more stories in rural school stock.
_TALL_SYSTEMS = {"Reinforced concrete frame", "Confined brick masonry"}


def _is_plausible(row: dict) -> bool:
    """Realism constraints distilled from the field campaign.

    Load bearing wall systems in this building stock do not exceed two
    stories, earthen buildings are single story with a regular plan, and
    informally mixed systems never exhibit good construction quality.
    """
    system = row["structural_system"]
    stories = row.get("stories")
    if stories == "3 or more" and system not in _TALL_SYSTEMS:
        return False
    if system == "Earthen construction":
        if stories not in (None, "1"):
            return False
        if row.get("plan_shape") == "Irregular":
            return False
    if system == "Mixed or informal system" and row.get("construction_quality") == "Good":
        return False
    return True


def generate_buildings(
    hazard: str,
    n: int,
    seed: int = 0,
    max_attempts: int = 400_000,
) -> pd.DataFrame:
    """Draw n unique, plausible building configurations for one hazard."""
    schema = HAZARDS[hazard]
    rng = np.random.default_rng(seed)
    names = schema.feature_names
    options = [schema.features[f] for f in names]

    seen: set[tuple] = set()
    rows: list[tuple] = []
    attempts = 0
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        draw = tuple(opts[rng.integers(len(opts))] for opts in options)
        if draw in seen:
            continue
        record = dict(zip(names, draw))
        if not _is_plausible(record):
            continue
        seen.add(draw)
        rows.append(draw)
    if len(rows) < n:
        raise RuntimeError(
            f"{hazard}: only {len(rows)} plausible unique configurations "
            f"found after {max_attempts} draws"
        )
    return pd.DataFrame(rows, columns=names)


def label_with_model(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Attach the model predicted risk level to a configuration frame."""
    labeled = df.copy()
    labeled["risk_level"] = pipeline.predict(df)
    return labeled


def apply_intervention(df: pd.DataFrame, changes: dict[str, str]) -> pd.DataFrame:
    """Return a copy of the portfolio with a retrofit applied to every building."""
    updated = df.copy()
    for column, value in changes.items():
        if column not in updated.columns:
            raise KeyError(f"unknown feature '{column}'")
        updated[column] = value
    return updated


def risk_migration(
    before: pd.Series, after: pd.Series
) -> pd.DataFrame:
    """Cross tabulation of predicted risk before and after an intervention."""
    table = pd.crosstab(
        before.rename("Before"), after.rename("After"), dropna=False
    )
    order = ["Low", "Medium", "High"]
    return table.reindex(index=order, columns=order, fill_value=0)
