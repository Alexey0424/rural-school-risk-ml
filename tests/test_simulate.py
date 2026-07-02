import pandas as pd
import pytest

from schoolrisk.config import HAZARDS
from schoolrisk.simulate import (
    apply_intervention,
    generate_buildings,
    risk_migration,
)


@pytest.mark.parametrize("hazard", list(HAZARDS))
def test_generated_buildings_are_unique_and_valid(hazard):
    df = generate_buildings(hazard, n=150, seed=3)
    assert len(df) == 150
    assert not df.duplicated().any()
    schema = HAZARDS[hazard]
    assert list(df.columns) == schema.feature_names
    for feature, allowed in schema.features.items():
        assert set(df[feature]) <= set(allowed)


def test_realism_constraints_hold():
    df = generate_buildings("earthquake", n=400, seed=11)
    tall = df[df["stories"] == "3 or more"]
    assert set(tall["structural_system"]) <= {
        "Reinforced concrete frame",
        "Confined brick masonry",
    }
    earthen = df[df["structural_system"] == "Earthen construction"]
    assert set(earthen["stories"]) <= {"1"}
    assert set(earthen["plan_shape"]) <= {"Regular"}
    mixed = df[df["structural_system"] == "Mixed or informal system"]
    assert "Good" not in set(mixed["construction_quality"])


def test_generation_is_deterministic():
    a = generate_buildings("flood", n=60, seed=5)
    b = generate_buildings("flood", n=60, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_apply_intervention_replaces_column():
    df = generate_buildings("flood", n=25, seed=1)
    out = apply_intervention(df, {"flood_barrier": "Yes"})
    assert (out["flood_barrier"] == "Yes").all()
    assert (df["flood_barrier"] != "Yes").any()


def test_apply_intervention_rejects_unknown_feature():
    df = generate_buildings("flood", n=5, seed=1)
    with pytest.raises(KeyError):
        apply_intervention(df, {"helipad": "Yes"})


def test_risk_migration_is_a_full_matrix():
    before = pd.Series(["High", "High", "Medium", "Low"])
    after = pd.Series(["Medium", "High", "Low", "Low"])
    table = risk_migration(before, after)
    assert list(table.index) == ["Low", "Medium", "High"]
    assert list(table.columns) == ["Low", "Medium", "High"]
    assert table.values.sum() == 4
