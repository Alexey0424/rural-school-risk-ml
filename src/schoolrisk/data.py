"""Dataset loading and validation.

The original field survey is confidential, so the loader resolves data in two
tiers: a local ``data/private`` folder (never versioned) holding the real
records, and the versioned ``data/demo`` folder holding a synthetic sample
with the same schema. Anyone cloning the repository can run the full pipeline
on the demo tier; the results published in the notebooks and in the README
come from the private tier.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, HAZARDS, RANDOM_SEED, TARGET, TEST_SIZE, RISK_LEVELS


def resolve_data_file(hazard: str, data_dir: Path | None = None) -> Path:
    """Return the dataset path for a hazard, preferring the private tier."""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    private = base / "private" / f"{hazard}.csv"
    demo = base / "demo" / f"{hazard}.csv"
    if private.exists():
        return private
    if demo.exists():
        return demo
    raise FileNotFoundError(
        f"No dataset found for '{hazard}'. Expected {private} or {demo}."
    )


def validate_dataset(hazard: str, df: pd.DataFrame) -> None:
    """Check that a dataframe matches the hazard schema exactly."""
    schema = HAZARDS[hazard]
    expected = schema.feature_names + [TARGET]
    if list(df.columns) != expected:
        raise ValueError(
            f"{hazard}: column mismatch.\n  found:    {list(df.columns)}\n"
            f"  expected: {expected}"
        )
    if df.isna().any().any():
        raise ValueError(f"{hazard}: dataset contains missing values")
    for feature, allowed in schema.features.items():
        unknown = set(df[feature].unique()) - set(allowed)
        if unknown:
            raise ValueError(f"{hazard}/{feature}: unknown categories {unknown}")
    unknown_labels = set(df[TARGET].unique()) - set(RISK_LEVELS)
    if unknown_labels:
        raise ValueError(f"{hazard}: unknown risk levels {unknown_labels}")


def load_dataset(
    hazard: str,
    data_dir: Path | None = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Load, validate and optionally deduplicate one hazard dataset.

    Exact duplicate records are dropped by default: with a purely categorical
    feature space, identical rows landing on both sides of the train and test
    split would leak information and inflate the holdout scores.
    """
    path = resolve_data_file(hazard, data_dir)
    df = pd.read_csv(path, dtype=str)
    validate_dataset(hazard, df)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train and test split, returning X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y, shuffle=True
    )


def class_counts(df: pd.DataFrame) -> pd.Series:
    """Class counts in the canonical Low, Medium, High order."""
    return df[TARGET].value_counts().reindex(list(RISK_LEVELS)).fillna(0).astype(int)
