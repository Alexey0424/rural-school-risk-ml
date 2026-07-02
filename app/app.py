"""Rural School Risk Screener.

A Streamlit application that scores a rural school building against the four
trained hazard models and explains every rating with SHAP. Run it from the
repository root:

    streamlit run app/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from schoolrisk.config import FIGURES_DIR, HAZARDS, MODELS_DIR, RISK_LEVELS
from schoolrisk.explain import local_contributions
from schoolrisk.modeling import load_card, load_model, predict_record
from schoolrisk.plots import INK_SOFT, RISK_COLORS, apply_style

st.set_page_config(
    page_title="Rural School Risk Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_TEXT = {"Low": "#ffffff", "Medium": "#1a1a19", "High": "#ffffff"}

FIELD_GROUPS = {
    "Structure": [
        "structural_system",
        "construction_period",
        "stories",
        "construction_quality",
        "structural_damage",
        "plan_shape",
    ],
    "Walls and openings": ["facade_openings", "ring_beam", "wall_connections"],
    "Roof": ["roof_covering", "roof_geometry", "roof_anchorage", "roof_condition"],
    "Site protection": ["slope_retention", "retention_maintenance", "flood_barrier"],
}

st.markdown(
    """
    <style>
    .risk-badge {
        display: inline-block; padding: 0.30rem 0.9rem; border-radius: 0.4rem;
        font-weight: 600; font-size: 1.05rem; letter-spacing: 0.02em;
    }
    .proba-track {
        background: #eceae4; border-radius: 0.25rem; height: 0.55rem;
        width: 100%; margin: 0.15rem 0 0.45rem 0;
    }
    .proba-fill { height: 100%; border-radius: 0.25rem; }
    .proba-label { font-size: 0.82rem; color: #52514e; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained models")
def load_models() -> tuple[dict, dict]:
    models, cards = {}, {}
    for hazard in HAZARDS:
        models[hazard] = load_model(hazard)
        cards[hazard] = load_card(hazard)
    return models, cards


def feature_options() -> dict[str, list[str]]:
    """Union of admissible values per feature in a stable display order."""
    options: dict[str, list[str]] = {}
    for schema in HAZARDS.values():
        for feature, values in schema.features.items():
            known = options.setdefault(feature, [])
            for value in values:
                if value not in known:
                    known.append(value)
    return options


def display_name(feature: str) -> str:
    for schema in HAZARDS.values():
        if feature in schema.features:
            return schema.display_name(feature)
    return feature


def badge(level: str) -> str:
    return (
        f'<span class="risk-badge" style="background:{RISK_COLORS[level]};'
        f'color:{RISK_TEXT[level]};">{level} risk</span>'
    )


def probability_bars(probabilities: dict[str, float]) -> str:
    parts = []
    for level in RISK_LEVELS:
        share = probabilities.get(level, 0.0)
        parts.append(
            f'<div class="proba-label">{level}: {share:.0%}</div>'
            f'<div class="proba-track"><div class="proba-fill" '
            f'style="width:{share:.0%};background:{RISK_COLORS[level]};"></div></div>'
        )
    return "".join(parts)


def contribution_chart(contrib: pd.DataFrame, top: int = 6) -> plt.Figure:
    data = contrib.head(top).iloc[::-1]
    labels = [f"{f}: {v}" for f, v in zip(data["feature"], data["value"])]
    colors = ["#e34948" if s > 0 else "#2a78d6" for s in data["shap"]]
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(data) + 0.7))
    ax.barh(labels, data["shap"], color=colors, height=0.6)
    ax.axvline(0, color="#c3c2b7", linewidth=0.9)
    ax.set_xlabel("Pushes rating down        Pushes rating up", fontsize=8.5,
                  color=INK_SOFT)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    return fig


def screen_tab(models: dict, cards: dict) -> None:
    st.subheader("Describe the building")
    st.caption(
        "Fill in the survey attributes below. The four hazard models score the "
        "building independently, so one description yields four ratings."
    )
    options = feature_options()
    record: dict[str, str] = {}
    columns = st.columns(len(FIELD_GROUPS), gap="large")
    for column, (group, features) in zip(columns, FIELD_GROUPS.items()):
        with column:
            st.markdown(f"**{group}**")
            for feature in features:
                record[feature] = st.selectbox(
                    display_name(feature), options[feature], key=feature
                )

    st.divider()
    results = {
        hazard: predict_record(models[hazard], record, hazard)
        for hazard in HAZARDS
    }
    cols = st.columns(len(results), gap="large")
    for col, (hazard, result) in zip(cols, results.items()):
        schema = HAZARDS[hazard]
        with col:
            st.markdown(f"##### {schema.display}")
            st.markdown(badge(result["risk_level"]), unsafe_allow_html=True)
            st.markdown(
                probability_bars(result["probabilities"]), unsafe_allow_html=True
            )
            with st.expander("Why this rating"):
                row = pd.DataFrame([{f: record[f] for f in schema.feature_names}])
                contrib = local_contributions(
                    models[hazard], row, hazard, result["risk_level"]
                )
                st.pyplot(contribution_chart(contrib), width=560)
                st.caption(
                    "Signed SHAP contributions toward the predicted class for "
                    "this specific building."
                )


def batch_tab(models: dict) -> None:
    st.subheader("Score a portfolio from a CSV file")
    hazard = st.selectbox(
        "Hazard model",
        list(HAZARDS),
        format_func=lambda h: HAZARDS[h].display,
    )
    schema = HAZARDS[hazard]

    template = pd.DataFrame(
        [{f: values[0] for f, values in schema.features.items()}]
    )
    st.download_button(
        "Download the CSV template",
        template.to_csv(index=False).encode(),
        file_name=f"{hazard}_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload buildings", type="csv")
    if uploaded is None:
        st.info(
            "The file needs one row per building and exactly the template "
            "columns. Every cell must use one of the admissible categories."
        )
        return

    df = pd.read_csv(uploaded, dtype=str)
    problems = []
    missing = [c for c in schema.feature_names if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
    else:
        for feature, allowed in schema.features.items():
            bad = sorted(set(df[feature].dropna().unique()) - set(allowed))
            if bad:
                problems.append(f"{feature}: unknown values {bad}")
        if df[schema.feature_names].isna().any().any():
            problems.append("empty cells found")
    if problems:
        st.error("The file does not match the schema. " + " | ".join(problems))
        return

    X = df[schema.feature_names]
    scored = df.copy()
    scored["predicted_risk"] = models[hazard].predict(X)
    proba = models[hazard].predict_proba(X)
    for cls, column in zip(models[hazard].classes_, proba.T):
        scored[f"p_{cls.lower()}"] = column.round(3)

    counts = (
        scored["predicted_risk"].value_counts().reindex(list(RISK_LEVELS)).fillna(0)
    )
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.metric("Buildings scored", len(scored))
        for level in RISK_LEVELS:
            st.markdown(
                f"{badge(level)}&nbsp;&nbsp;{int(counts[level])} buildings",
                unsafe_allow_html=True,
            )
    with right:
        st.dataframe(scored, height=320, hide_index=True)
    st.download_button(
        "Download predictions",
        scored.to_csv(index=False).encode(),
        file_name=f"{hazard}_predictions.csv",
        mime="text/csv",
    )


def performance_tab(cards: dict) -> None:
    st.subheader("Holdout performance of the deployed models")
    rows = []
    for hazard, card in cards.items():
        rows.append(
            {
                "Hazard": HAZARDS[hazard].display,
                "Model": card["model"],
                "Accuracy": card["holdout_metrics"]["accuracy"],
                "Balanced accuracy": card["holdout_metrics"]["balanced_accuracy"],
                "Macro F1": card["holdout_metrics"]["f1_macro"],
                "MCC": card["holdout_metrics"]["mcc"],
                "ROC AUC": card["holdout_metrics"]["roc_auc"],
                "Training records": card["n_train"],
                "Holdout records": card["n_test"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows).round(3), hide_index=True,
    )
    st.caption(
        "Metrics computed on the stratified twenty percent holdout partition "
        "of the expert rated survey."
    )
    for figure, caption in [
        ("confusion_matrices", "Holdout confusion matrices"),
        ("cv_model_comparison", "Cross validated model comparison"),
    ]:
        path = FIGURES_DIR / f"{figure}.png"
        if path.exists():
            st.image(str(path), caption=caption)


def methodology_tab() -> None:
    st.subheader("How this tool was built")
    st.markdown(
        """
        A field campaign surveyed rural school buildings in Colombia and a
        structural specialist rated the risk of every building for four
        natural hazards: earthquake, landslide, flood and windstorm. Each
        rating uses three levels: Low, Medium and High.

        The models in this application learned to reproduce that specialist
        judgment from the survey attributes alone. For every hazard the
        pipeline compares eleven algorithms under ten fold stratified cross
        validation, tunes the strongest candidates with randomized search, and
        keeps the best performer. All features are categorical and encoded
        ordinally, from the most favorable condition to the least favorable
        one, so the models receive the severity ranking a specialist would
        use.

        SHAP values quantify how much every attribute pushes a rating, both
        globally in the Model performance tab and for each individual
        building in the screening tab.

        The original survey records are confidential and are not distributed
        with this repository. A synthetic demonstration dataset with the same
        schema allows anyone to rerun the full pipeline.
        """
    )
    st.warning(
        "This application is a screening aid for prioritization. It does not "
        "replace a structural evaluation by a qualified professional."
    )


def main() -> None:
    apply_style()
    st.title("Rural School Risk Screener")
    st.caption(
        "Machine learning screening of rural school buildings for earthquake, "
        "landslide, flood and windstorm risk, trained to reproduce expert "
        "structural judgment."
    )

    if not (MODELS_DIR / "earthquake_model.pkl").exists():
        st.error(
            "No trained models found. Run: python scripts/train_models.py"
        )
        st.stop()

    models, cards = load_models()

    with st.sidebar:
        st.header("Deployed models")
        for hazard, card in cards.items():
            st.markdown(
                f"**{HAZARDS[hazard].display}**  \n"
                f"{card['model']}  \n"
                f"Holdout macro F1: {card['holdout_metrics']['f1_macro']:.3f}"
            )
        st.divider()
        st.caption(
            "Ratings follow the judgment of the structural specialist who "
            "labeled the training survey. Confidential survey data stays out "
            "of this repository."
        )

    screen, batch, performance, methodology = st.tabs(
        ["Screen a building", "Batch scoring", "Model performance", "Methodology"]
    )
    with screen:
        screen_tab(models, cards)
    with batch:
        batch_tab(models)
    with performance:
        performance_tab(cards)
    with methodology:
        methodology_tab()


main()
