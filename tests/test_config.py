from schoolrisk.config import FEATURE_DISPLAY, HAZARDS, RISK_LEVELS


def test_four_hazards_registered():
    assert set(HAZARDS) == {"earthquake", "landslide", "flood", "windstorm"}


def test_risk_levels_are_fixed():
    assert RISK_LEVELS == ("Low", "Medium", "High")


def test_every_feature_has_display_name():
    for schema in HAZARDS.values():
        for feature in schema.feature_names:
            assert feature in FEATURE_DISPLAY


def test_category_lists_are_well_formed():
    for schema in HAZARDS.values():
        for feature, values in schema.features.items():
            assert len(values) >= 2, feature
            assert len(set(values)) == len(values), feature


def test_shared_features_share_vocabulary():
    """The same feature may order values differently per hazard, but the
    admissible value sets must match so the application can use one form."""
    vocab: dict[str, set[str]] = {}
    for schema in HAZARDS.values():
        for feature, values in schema.features.items():
            expected = vocab.setdefault(feature, set(values))
            assert set(values) == expected, feature
