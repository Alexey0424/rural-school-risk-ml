import math

import pytest

from schoolrisk.config import HAZARDS, MODELS_DIR, RISK_LEVELS
from schoolrisk.data import load_dataset
from schoolrisk.modeling import load_card, load_model, predict_record

models_available = pytest.mark.skipif(
    not (MODELS_DIR / "earthquake_model.pkl").exists(),
    reason="trained models not present; run scripts/train_models.py",
)


@models_available
@pytest.mark.parametrize("hazard", list(HAZARDS))
def test_pipeline_predicts_valid_levels(hazard):
    pipeline = load_model(hazard)
    X = load_dataset(hazard).drop(columns=["risk_level"]).head(20)
    predictions = pipeline.predict(X)
    assert set(predictions) <= set(RISK_LEVELS)


@models_available
@pytest.mark.parametrize("hazard", list(HAZARDS))
def test_predict_record_returns_probabilities(hazard):
    pipeline = load_model(hazard)
    schema = HAZARDS[hazard]
    worst = {f: values[-1] for f, values in schema.features.items()}
    result = predict_record(pipeline, worst, hazard)
    assert result["risk_level"] in RISK_LEVELS
    assert math.isclose(sum(result["probabilities"].values()), 1.0, abs_tol=1e-6)


@models_available
@pytest.mark.parametrize("hazard", list(HAZARDS))
def test_extreme_records_rank_sensibly(hazard):
    """The all favorable building must not score more High probability than
    the all unfavorable building."""
    pipeline = load_model(hazard)
    schema = HAZARDS[hazard]
    best = {f: values[0] for f, values in schema.features.items()}
    worst = {f: values[-1] for f, values in schema.features.items()}
    p_best = predict_record(pipeline, best, hazard)["probabilities"]["High"]
    p_worst = predict_record(pipeline, worst, hazard)["probabilities"]["High"]
    assert p_worst >= p_best


@models_available
@pytest.mark.parametrize("hazard", list(HAZARDS))
def test_model_card_is_consistent(hazard):
    card = load_card(hazard)
    assert card["hazard"] == hazard
    assert card["classes"] == list(RISK_LEVELS)
    assert set(card["features"]) == set(HAZARDS[hazard].feature_names)
    assert 0.5 < card["holdout_metrics"]["f1_macro"] <= 1.0
