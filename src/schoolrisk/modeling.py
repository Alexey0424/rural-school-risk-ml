"""Model zoo, cross validated comparison, tuning, evaluation and persistence.

The design goal is a faithful and reproducible protocol shared by every
hazard: an ordinal encoder that respects the severity ordering of each
attribute, a stratified train and test split, a ten fold cross validated
comparison of candidate algorithms on the training partition, randomized
hyperparameter search for the strongest candidates, and a final evaluation on
the untouched holdout partition.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import (
    CV_FOLDS,
    HAZARDS,
    MODELS_DIR,
    RANDOM_SEED,
    RISK_LEVELS,
    TUNING_ITERATIONS,
)

MODEL_DISPLAY = {
    "baseline": "Most frequent baseline",
    "logreg": "Logistic regression",
    "knn": "K nearest neighbors",
    "nbayes": "Gaussian naive Bayes",
    "dtree": "Decision tree",
    "rf": "Random forest",
    "et": "Extra trees",
    "gboost": "Gradient boosting",
    "ada": "AdaBoost",
    "svm": "Support vector machine",
    "lgbm": "LightGBM",
}

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "mcc": make_scorer(matthews_corrcoef),
    "kappa": make_scorer(cohen_kappa_score),
    "roc_auc": "roc_auc_ovr",
}

PRIMARY_METRIC = "f1_macro"

#: Families eligible for deployment. The screening application ships exact
#: per prediction SHAP attributions through the tree explainer, so the model
#: that gets persisted is the strongest tree ensemble; the remaining families
#: still take part in the cross validated comparison.
DEPLOYABLE_MODELS = ("rf", "et", "lgbm", "dtree")


def build_encoder(hazard: str) -> OrdinalEncoder:
    """Ordinal encoder honoring the severity ordering declared in the schema."""
    schema = HAZARDS[hazard]
    encoder = OrdinalEncoder(categories=[list(c) for c in schema.categories])
    encoder.set_output(transform="pandas")
    return encoder


def make_estimator(key: str, seed: int = RANDOM_SEED):
    """Instantiate one candidate algorithm with a fixed seed."""
    estimators = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=5000, random_state=seed),
        "knn": KNeighborsClassifier(),
        "nbayes": GaussianNB(),
        "dtree": DecisionTreeClassifier(random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "et": ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "gboost": GradientBoostingClassifier(random_state=seed),
        "ada": AdaBoostClassifier(random_state=seed),
        "svm": SVC(probability=True, random_state=seed),
        "lgbm": LGBMClassifier(
            random_state=seed, verbose=-1, n_jobs=1, force_row_wise=True
        ),
    }
    return estimators[key]


def build_pipeline(hazard: str, model_key: str, seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        steps=[
            ("encoder", build_encoder(hazard)),
            ("model", make_estimator(model_key, seed)),
        ]
    )


def cv_splitter(seed: int = RANDOM_SEED, folds: int = CV_FOLDS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def compare_models(
    hazard: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_keys: list[str] | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Cross validate every candidate on the training partition.

    Returns one row per algorithm with the mean and standard deviation of each
    metric across folds, sorted by macro F1.
    """
    keys = model_keys or list(MODEL_DISPLAY)
    rows = []
    for key in keys:
        pipe = build_pipeline(hazard, key, seed)
        result = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv_splitter(seed),
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )
        row: dict[str, object] = {"key": key, "model": MODEL_DISPLAY[key]}
        for metric in SCORING:
            scores = result[f"test_{metric}"]
            row[metric] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=False)
    return table.reset_index(drop=True)


def _search_space(key: str) -> dict:
    """Randomized search distributions per algorithm, on pipeline parameter names."""
    spaces: dict[str, dict] = {
        "rf": {
            "model__n_estimators": randint(100, 800),
            "model__max_depth": [None, 4, 6, 8, 12, 16, 20],
            "model__min_samples_split": randint(2, 12),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced"],
        },
        "et": {
            "model__n_estimators": randint(100, 800),
            "model__max_depth": [None, 4, 6, 8, 12, 16, 20],
            "model__min_samples_split": randint(2, 12),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced"],
        },
        "lgbm": {
            "model__n_estimators": randint(100, 700),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__num_leaves": randint(7, 64),
            "model__max_depth": [-1, 3, 5, 8, 12],
            "model__min_child_samples": randint(2, 25),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_alpha": loguniform(1e-8, 1.0),
            "model__reg_lambda": loguniform(1e-8, 1.0),
            "model__class_weight": [None, "balanced"],
        },
        "gboost": {
            "model__n_estimators": randint(100, 500),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_depth": randint(2, 6),
            "model__subsample": uniform(0.6, 0.4),
            "model__min_samples_leaf": randint(1, 6),
        },
        "dtree": {
            "model__max_depth": [None, 3, 5, 8, 12, 16],
            "model__min_samples_split": randint(2, 12),
            "model__min_samples_leaf": randint(1, 6),
            "model__criterion": ["gini", "entropy"],
            "model__class_weight": [None, "balanced"],
        },
        "svm": {
            "model__C": loguniform(0.1, 100),
            "model__gamma": ["scale", "auto"],
            "model__class_weight": [None, "balanced"],
        },
        "logreg": {
            "model__C": loguniform(0.01, 100),
            "model__class_weight": [None, "balanced"],
        },
        "knn": {
            "model__n_neighbors": randint(3, 15),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "ada": {
            "model__n_estimators": randint(50, 500),
            "model__learning_rate": loguniform(0.01, 2.0),
        },
        "nbayes": {
            "model__var_smoothing": loguniform(1e-11, 1e-6),
        },
    }
    return spaces.get(key, {})


@dataclass
class TunedCandidate:
    key: str
    display: str
    cv_score: float
    best_params: dict
    pipeline: Pipeline


def tune_model(
    hazard: str,
    key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = TUNING_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> TunedCandidate:
    """Randomized hyperparameter search for one algorithm.

    The tuned candidate never scores below its own default configuration: the
    default is evaluated with the same folds and kept when the search cannot
    improve on it.
    """
    pipe = build_pipeline(hazard, key, seed)
    space = _search_space(key)

    default_cv = cross_validate(
        pipe, X_train, y_train, cv=cv_splitter(seed),
        scoring={PRIMARY_METRIC: SCORING[PRIMARY_METRIC]}, n_jobs=-1,
    )[f"test_{PRIMARY_METRIC}"].mean()

    if not space:
        pipe.fit(X_train, y_train)
        return TunedCandidate(key, MODEL_DISPLAY[key], default_cv, {}, pipe)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=space,
        n_iter=n_iter,
        scoring=PRIMARY_METRIC,
        cv=cv_splitter(seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    if search.best_score_ >= default_cv:
        return TunedCandidate(
            key,
            MODEL_DISPLAY[key],
            float(search.best_score_),
            {k.removeprefix("model__"): v for k, v in search.best_params_.items()},
            search.best_estimator_,
        )
    pipe.fit(X_train, y_train)
    return TunedCandidate(key, MODEL_DISPLAY[key], float(default_cv), {}, pipe)


def evaluate_holdout(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Score a fitted pipeline on the holdout partition."""
    labels = list(RISK_LEVELS)
    y_pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "kappa": cohen_kappa_score(y_test, y_pred),
        "roc_auc": roc_auc_score(
            y_test, proba, multi_class="ovr", average="macro",
            labels=pipeline.classes_,
        ),
    }
    report = pd.DataFrame(
        classification_report(
            y_test, y_pred, labels=labels, output_dict=True, zero_division=0
        )
    ).T
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
    }


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_model(
    hazard: str,
    candidate: TunedCandidate,
    holdout: dict,
    n_train: int,
    n_test: int,
    models_dir: Path | None = None,
) -> Path:
    """Persist the fitted pipeline together with a model card."""
    out = Path(models_dir) if models_dir is not None else MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"{hazard}_model.pkl"
    joblib.dump(candidate.pipeline, model_path)

    schema = HAZARDS[hazard]
    card = {
        "hazard": hazard,
        "display": schema.display,
        "model_key": candidate.key,
        "model": candidate.display,
        "cv_f1_macro": candidate.cv_score,
        "best_params": _jsonable(candidate.best_params),
        "holdout_metrics": holdout["metrics"],
        "classes": list(RISK_LEVELS),
        "features": {name: list(vals) for name, vals in schema.features.items()},
        "n_train": int(n_train),
        "n_test": int(n_test),
        "random_seed": RANDOM_SEED,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    with open(out / f"{hazard}_card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    return model_path


def load_model(hazard: str, models_dir: Path | None = None) -> Pipeline:
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    return joblib.load(base / f"{hazard}_model.pkl")


def load_card(hazard: str, models_dir: Path | None = None) -> dict:
    base = Path(models_dir) if models_dir is not None else MODELS_DIR
    with open(base / f"{hazard}_card.json", encoding="utf-8") as fh:
        return json.load(fh)


def predict_record(pipeline: Pipeline, record: dict, hazard: str) -> dict:
    """Predict one building described as a plain dictionary."""
    schema = HAZARDS[hazard]
    row = pd.DataFrame([{name: record[name] for name in schema.feature_names}])
    label = str(pipeline.predict(row)[0])
    proba = pipeline.predict_proba(row)[0]
    by_class = {str(c): float(p) for c, p in zip(pipeline.classes_, proba)}
    return {
        "risk_level": label,
        "probabilities": {level: by_class.get(level, 0.0) for level in RISK_LEVELS},
    }
