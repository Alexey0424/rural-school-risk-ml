"""Build the versioned synthetic demonstration datasets.

The demo tier exists so that anyone can run the notebooks, the tests and the
application without the confidential survey. Buildings are drawn at random
under the realism constraints of the field campaign and labeled by the
trained pipelines, so no confidential rating logic is involved anywhere in
the produced files.

Usage:
    python scripts/generate_demo_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schoolrisk.config import DATA_DIR, HAZARDS, RISK_LEVELS, TARGET
from schoolrisk.modeling import load_model
from schoolrisk.simulate import generate_buildings, label_with_model

#: Unique plausible configurations to draw per hazard. Flood has a compact
#: feature space (seven attributes), so its pool stays below the roughly
#: 2400 configurations that exist under the realism constraints.
POOL_SIZES = {"earthquake": 4000, "landslide": 4000, "flood": 2000, "windstorm": 4000}
PER_CLASS = 70
SEED = 20240


def main() -> None:
    out_dir = DATA_DIR / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for hazard in HAZARDS:
        pipeline = load_model(hazard)
        pool = generate_buildings(hazard, n=POOL_SIZES[hazard], seed=SEED)
        labeled = label_with_model(pool, pipeline)

        parts = []
        for level in RISK_LEVELS:
            members = labeled[labeled[TARGET] == level]
            parts.append(members.head(PER_CLASS))
        demo = (
            pd.concat(parts)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
        demo.to_csv(out_dir / f"{hazard}.csv", index=False)
        counts = demo[TARGET].value_counts().reindex(list(RISK_LEVELS)).fillna(0)
        print(
            f"{hazard:10s} n={len(demo):3d} "
            + " ".join(f"{lvl}={int(counts[lvl])}" for lvl in RISK_LEVELS)
        )


if __name__ == "__main__":
    main()
