import pandas as pd
import pytest

from schoolrisk.config import HAZARDS, TARGET
from schoolrisk.data import class_counts, load_dataset, validate_dataset


@pytest.mark.parametrize("hazard", list(HAZARDS))
def test_datasets_load_and_validate(hazard):
    df = load_dataset(hazard)
    assert len(df) > 50
    assert not df.duplicated().any()
    assert list(df.columns) == HAZARDS[hazard].feature_names + [TARGET]


@pytest.mark.parametrize("hazard", list(HAZARDS))
def test_class_counts_cover_all_levels(hazard):
    counts = class_counts(load_dataset(hazard))
    assert counts.sum() > 0
    assert list(counts.index) == ["Low", "Medium", "High"]


def test_validate_rejects_unknown_category():
    df = load_dataset("flood").copy()
    df.loc[0, "flood_barrier"] = "Maybe"
    with pytest.raises(ValueError, match="unknown categories"):
        validate_dataset("flood", df)


def test_validate_rejects_column_mismatch():
    df = load_dataset("flood").drop(columns=["stories"])
    with pytest.raises(ValueError, match="column mismatch"):
        validate_dataset("flood", df)


def test_validate_rejects_missing_values():
    df = load_dataset("flood").copy()
    df.loc[0, "construction_quality"] = pd.NA
    with pytest.raises(ValueError):
        validate_dataset("flood", df)
