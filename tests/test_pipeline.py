import numpy as np

from schoolrisk.data import load_dataset, split_dataset
from schoolrisk.modeling import (
    build_pipeline,
    evaluate_holdout,
)


def test_training_smoke_run_beats_chance():
    """A default random forest trained on the flood dataset must clearly beat
    chance on the holdout. Runs on whichever data tier is available."""
    df = load_dataset("flood")
    X_train, X_test, y_train, y_test = split_dataset(df)
    pipeline = build_pipeline("flood", "rf")
    pipeline.fit(X_train, y_train)
    result = evaluate_holdout(pipeline, X_test, y_test)
    assert result["metrics"]["f1_macro"] > 0.6
    assert result["confusion_matrix"].sum() == len(X_test)


def test_encoder_respects_declared_ordering():
    df = load_dataset("earthquake")
    X_train, _, y_train, _ = split_dataset(df)
    pipeline = build_pipeline("earthquake", "dtree")
    pipeline.fit(X_train, y_train)
    encoder = pipeline.named_steps["encoder"]
    sample = X_train.head(10)
    encoded = encoder.transform(sample)
    quality_order = {"Good": 0, "Moderate": 1, "Poor": 2}
    expected = sample["construction_quality"].map(quality_order).to_numpy()
    np.testing.assert_array_equal(
        encoded["construction_quality"].to_numpy().astype(int), expected
    )
