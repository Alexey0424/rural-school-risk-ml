# Data

## Confidentiality statement

The project was trained on a field survey of rural school buildings rated by
a structural specialist. Those records belong to a research project and are
**not distributed** with this repository. The published material is limited
to the machine learning pipeline, aggregate results, and a five row glimpse
inside the first notebook.

## Tiers

The loader in `schoolrisk.data` resolves each hazard dataset in two tiers:

| Tier | Path | Versioned | Content |
|------|------|-----------|---------|
| Private | `data/private/<hazard>.csv` | No, excluded by `.gitignore` | The real survey, local only |
| Demo | `data/demo/<hazard>.csv` | Yes | Synthetic buildings labeled by the trained models |

When the private tier is absent, everything (notebooks, tests, application,
training script) runs on the demo tier with the identical schema.

## Demo tier provenance

`scripts/generate_demo_data.py` draws random building configurations under
the realism constraints of the field campaign, labels them with the trained
pipelines, and balances the classes. The labels therefore come from the
published models, never from the confidential rating methodology.

## Schema

One CSV per hazard. All columns are categorical strings; `risk_level` is the
target with values Low, Medium and High. The admissible values of every
feature, ordered from the most favorable to the least favorable condition,
are declared in `src/schoolrisk/config.py` and enforced by
`schoolrisk.data.validate_dataset`.
