"""Central registry for the hazard schemas and experiment settings.

Every feature is categorical and ordered from the most favorable condition to
the least favorable one, following the judgment of the structural specialist
who rated the buildings. The orderings are hazard specific: the same attribute
can rank differently depending on the failure mechanism (for example, a single
story building is the favorable case under earthquake loading but the exposed
case under flooding).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 10
TUNING_ITERATIONS = 40

TARGET = "risk_level"
RISK_LEVELS = ("Low", "Medium", "High")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

#: Human readable names used in figures and in the application.
FEATURE_DISPLAY = {
    "structural_system": "Structural system",
    "construction_period": "Construction period",
    "stories": "Number of stories",
    "facade_openings": "Facade openings",
    "construction_quality": "Construction quality",
    "structural_damage": "Visible structural damage",
    "ring_beam": "Ring beam at wall tops",
    "wall_connections": "Wall to wall connections",
    "plan_shape": "Plan configuration",
    "roof_covering": "Roof covering",
    "roof_anchorage": "Roof to wall anchorage",
    "roof_condition": "Roof condition",
    "roof_geometry": "Roof geometry",
    "slope_retention": "Slope retention system",
    "retention_maintenance": "Retention system maintenance",
    "flood_barrier": "Flood protection barrier",
}

_QUALITY = ("Good", "Moderate", "Poor")
_PERIOD = ("After 2010", "2001 to 2010", "1980 to 2000", "Before 1980")
_OPENINGS = ("0 to 2", "3 to 5", "More than 5")
_ROOF_COVER = ("Clay tiles", "Fiber cement sheets", "Metal sheets")

# Structural systems ranked from the most favorable to the least favorable
# response for each hazard. Timber, for instance, performs well under ground
# shaking but is the weakest system against water and debris loads.
_SYSTEMS_EARTHQUAKE = (
    "Timber",
    "Reinforced concrete frame",
    "Confined brick masonry",
    "Concrete block masonry",
    "Partially confined brick masonry",
    "Unreinforced brick masonry",
    "Earthen construction",
    "Mixed or informal system",
)
_SYSTEMS_LANDSLIDE = (
    "Reinforced concrete frame",
    "Confined brick masonry",
    "Concrete block masonry",
    "Partially confined brick masonry",
    "Earthen construction",
    "Unreinforced brick masonry",
    "Mixed or informal system",
    "Timber",
)
_SYSTEMS_FLOOD = (
    "Reinforced concrete frame",
    "Confined brick masonry",
    "Concrete block masonry",
    "Partially confined brick masonry",
    "Unreinforced brick masonry",
    "Mixed or informal system",
    "Earthen construction",
    "Timber",
)
_SYSTEMS_WINDSTORM = (
    "Reinforced concrete frame",
    "Confined brick masonry",
    "Concrete block masonry",
    "Partially confined brick masonry",
    "Earthen construction",
    "Unreinforced brick masonry",
    "Mixed or informal system",
    "Timber",
)


@dataclass(frozen=True)
class HazardSchema:
    """Feature dictionary of one hazard.

    ``features`` maps each column name to the tuple of admissible category
    values, ordered from the most favorable to the least favorable condition.
    That ordering feeds the ordinal encoder, so the numeric representation
    seen by the models preserves the severity ranking of every attribute.
    """

    key: str
    display: str
    features: dict[str, tuple[str, ...]] = field(repr=False)

    @property
    def feature_names(self) -> list[str]:
        return list(self.features)

    @property
    def categories(self) -> list[tuple[str, ...]]:
        return [self.features[name] for name in self.features]

    def display_name(self, feature: str) -> str:
        return FEATURE_DISPLAY.get(feature, feature)


HAZARDS: dict[str, HazardSchema] = {
    "earthquake": HazardSchema(
        key="earthquake",
        display="Earthquake",
        features={
            "structural_system": _SYSTEMS_EARTHQUAKE,
            "construction_period": _PERIOD,
            "stories": ("1", "2", "3 or more"),
            "facade_openings": _OPENINGS,
            "construction_quality": _QUALITY,
            "structural_damage": ("No", "Yes"),
            "ring_beam": ("Yes", "No"),
            "wall_connections": ("Yes", "No"),
            "plan_shape": ("Regular", "Irregular"),
            "roof_covering": _ROOF_COVER,
            "roof_anchorage": _QUALITY,
            "roof_condition": _QUALITY,
        },
    ),
    "landslide": HazardSchema(
        key="landslide",
        display="Landslide",
        features={
            "structural_system": _SYSTEMS_LANDSLIDE,
            "construction_period": _PERIOD,
            "stories": ("3 or more", "2", "1"),
            "facade_openings": _OPENINGS,
            "construction_quality": _QUALITY,
            "structural_damage": ("No", "Yes"),
            "ring_beam": ("Yes", "No"),
            "wall_connections": ("Yes", "No"),
            "roof_covering": _ROOF_COVER,
            "roof_anchorage": _QUALITY,
            "slope_retention": ("Effective", "Present but ineffective", "Absent"),
            "retention_maintenance": _QUALITY,
        },
    ),
    "flood": HazardSchema(
        key="flood",
        display="Flood",
        features={
            "structural_system": _SYSTEMS_FLOOD,
            "construction_period": _PERIOD,
            "stories": ("3 or more", "2", "1"),
            "facade_openings": _OPENINGS,
            "construction_quality": _QUALITY,
            "structural_damage": ("No", "Yes"),
            "flood_barrier": ("Yes", "No"),
        },
    ),
    "windstorm": HazardSchema(
        key="windstorm",
        display="Windstorm",
        features={
            "structural_system": _SYSTEMS_WINDSTORM,
            "construction_period": _PERIOD,
            "stories": ("3 or more", "2", "1"),
            "facade_openings": _OPENINGS,
            "construction_quality": _QUALITY,
            "structural_damage": ("No", "Yes"),
            "roof_covering": _ROOF_COVER,
            "roof_anchorage": _QUALITY,
            "roof_condition": _QUALITY,
            "roof_geometry": ("Flat", "Hipped", "Gabled"),
        },
    ),
}
