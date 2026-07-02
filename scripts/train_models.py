"""End to end training for every hazard.

For each hazard the script loads the dataset, removes exact duplicates,
splits with stratification, cross validates the full model zoo on the
training partition, tunes the strongest candidates, evaluates the winner on
the holdout partition, and persists the fitted pipeline, the model card, the
metric tables and the report figures.

Usage:
    python scripts/train_models.py                 all hazards
    python scripts/train_models.py --hazards flood windstorm
    python scripts/train_models.py --finalists 2   tune fewer candidates
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schoolrisk.config import FIGURES_DIR, HAZARDS, METRICS_DIR, TARGET
from schoolrisk.data import load_dataset, split_dataset
from schoolrisk.explain import beeswarm_figure, contribution_figure, explain_hazard
from schoolrisk.modeling import (
    DEPLOYABLE_MODELS,
    PRIMARY_METRIC,
    compare_models,
    evaluate_holdout,
    save_model,
    tune_model,
)
from schoolrisk.plots import (
    apply_style,
    class_balance_figure,
    confusion_grid_figure,
    cv_comparison_figure,
    save_figure,
)


def train_hazard(hazard: str, finalists: int) -> dict:
    print(f"\n=== {HAZARDS[hazard].display} ===")
    t0 = time.perf_counter()
    df = load_dataset(hazard)
    X_train, X_test, y_train, y_test = split_dataset(df)
    print(f"records after deduplication: {len(df)} "
          f"(train {len(X_train)}, holdout {len(X_test)})")

    comparison = compare_models(hazard, X_train, y_train)
    comparison.to_csv(METRICS_DIR / f"{hazard}_cv_comparison.csv", index=False)
    shortlist = [k for k in comparison["key"] if k in DEPLOYABLE_MODELS][:finalists]
    print("shortlist:", ", ".join(shortlist))

    candidates = [tune_model(hazard, key, X_train, y_train) for key in shortlist]
    winner = max(candidates, key=lambda c: c.cv_score)
    print(f"selected {winner.display} (cv {PRIMARY_METRIC} {winner.cv_score:.4f})")

    holdout = evaluate_holdout(winner.pipeline, X_test, y_test)
    save_model(hazard, winner, holdout, len(X_train), len(X_test))
    holdout["report"].to_csv(METRICS_DIR / f"{hazard}_holdout_report.csv")

    bundle = explain_hazard(winner.pipeline, X_test, hazard)
    bundle["table"].to_csv(METRICS_DIR / f"{hazard}_shap_contribution.csv")
    save_figure(contribution_figure(bundle["table"], hazard), f"shap_{hazard}")
    save_figure(beeswarm_figure(bundle, hazard), f"beeswarm_{hazard}")

    print(f"holdout accuracy {holdout['metrics']['accuracy']:.4f}, "
          f"macro F1 {holdout['metrics']['f1_macro']:.4f} "
          f"({time.perf_counter() - t0:.1f}s)")
    return {
        "dataset": df,
        "comparison": comparison,
        "winner": winner,
        "holdout": holdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazards", nargs="+", default=list(HAZARDS),
                        choices=list(HAZARDS))
    parser.add_argument("--finalists", type=int, default=3,
                        help="how many shortlisted models to tune per hazard")
    args = parser.parse_args()

    apply_style()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {hz: train_hazard(hz, args.finalists) for hz in args.hazards}

    if set(args.hazards) == set(HAZARDS):
        datasets = {hz: r["dataset"] for hz, r in results.items()}
        save_figure(class_balance_figure(datasets), "class_balance")
        save_figure(
            cv_comparison_figure({hz: r["comparison"] for hz, r in results.items()}),
            "cv_model_comparison",
        )
        save_figure(
            confusion_grid_figure({hz: r["holdout"] for hz, r in results.items()}),
            "confusion_matrices",
        )

        summary = {
            hz: {
                "model": r["winner"].display,
                "cv_f1_macro": round(r["winner"].cv_score, 4),
                **{k: round(v, 4) for k, v in r["holdout"]["metrics"].items()},
                "n_records": len(r["dataset"]),
                "class_counts": r["dataset"][TARGET].value_counts().to_dict(),
            }
            for hz, r in results.items()
        }
        with open(METRICS_DIR / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("\nsummary written to", METRICS_DIR / "summary.json")
        print(pd.DataFrame(summary).T[["model", "accuracy", "f1_macro", "roc_auc"]])


if __name__ == "__main__":
    main()
