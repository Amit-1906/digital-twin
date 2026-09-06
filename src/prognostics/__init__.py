"""TwinVolt Level 6 — Battery Prognostics, Degradation Trajectories, and Feature Extraction.

Provides universal, chemistry-independent, and hardware-agnostic domain contracts,
degradation trajectories, multi-dimensional feature extraction, End-of-Life (EOL)
boundary specifications, prediction horizons, and immutable prediction value objects.
"""

from src.prognostics.estimators import (
    AbstractRULEstimator,
    EstimatorConfig,
    HistoricalLinearRULEstimator,
)
from src.prognostics.features import (
    DegradationFeatureSet,
    PrognosticFeature,
    TrajectoryFeatureExtractor,
)
from src.prognostics.horizon import PrognosticHorizon
from src.prognostics.prediction import (
    EOLDefinition,
    PredictionUncertainty,
    PrognosticMethodIdentity,
    RULPredictionResult,
)
from src.prognostics.provenance import (
    PrognosticQuantity,
    PrognosticValueProvenance,
)
from src.prognostics.state import (
    PrognosticOperatingContext,
    PrognosticStateSnapshot,
)
from src.prognostics.trajectory import (
    DegradationObservation,
    DegradationTrajectory,
    TrajectoryBuilder,
)
from src.prognostics.trajectory_types import (
    FeatureQualityStatus,
    RateDomain,
    TrajectoryOrderingStatus,
    TrajectoryQuantityType,
    TrendDirection,
)
from src.prognostics.types import (
    ComparisonOperator,
    EOLCriterionType,
    HorizonDomain,
    PrognosticStatus,
    QuantityAvailability,
    TimeUnit,
    UncertaintyType,
    ValueOriginType,
)

__all__ = [
    # Level 6.1 Enums and Classifications
    "PrognosticStatus",
    "ValueOriginType",
    "QuantityAvailability",
    "EOLCriterionType",
    "ComparisonOperator",
    "HorizonDomain",
    "TimeUnit",
    "UncertaintyType",
    # Level 6.1 Value Objects
    "PrognosticValueProvenance",
    "PrognosticQuantity",
    "PrognosticHorizon",
    "PrognosticOperatingContext",
    "PrognosticStateSnapshot",
    "EOLDefinition",
    "PrognosticMethodIdentity",
    "PredictionUncertainty",
    "RULPredictionResult",
    # Level 6.2 Trajectory Enums and Types
    "TrajectoryQuantityType",
    "TrajectoryOrderingStatus",
    "RateDomain",
    "FeatureQualityStatus",
    "TrendDirection",
    # Level 6.2 Trajectory and Feature Value Objects
    "DegradationObservation",
    "DegradationTrajectory",
    "TrajectoryBuilder",
    "PrognosticFeature",
    "DegradationFeatureSet",
    "TrajectoryFeatureExtractor",
    # Level 6.3 Estimator Contracts
    "AbstractRULEstimator",
    "EstimatorConfig",
    "HistoricalLinearRULEstimator",
]
