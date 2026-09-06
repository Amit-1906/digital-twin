"""Abstract Base Contracts for Remaining Useful Life (RUL) Estimators.

Defines the core interface, capability declarations, and configuration objects
for deterministic prognostic algorithms in TwinVolt.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from src.prognostics.horizon import PrognosticHorizon
from src.prognostics.prediction import (
    EOLDefinition,
    PrognosticMethodIdentity,
    RULPredictionResult,
)
from src.prognostics.trajectory import DegradationTrajectory
from src.prognostics.trajectory_types import TrajectoryQuantityType
from src.prognostics.types import EOLCriterionType, HorizonDomain


@dataclass(frozen=True)
class EstimatorConfig:
    """Configuration options governing an RUL estimator execution.

    Attributes:
        min_observations: Minimum valid (domain, quantity) coordinate pairs required.
        slope_threshold: Minimum absolute slope threshold for non-flat trajectory validation.
        use_ols_slope: If True, uses Ordinary Least Squares linear regression across history
            as a descriptive trend slope estimator for the linear baseline. Note: OLS is purely
            a descriptive trajectory trend estimator; no empirical RUL model or machine-learning
            RUL model was implemented.
        default_domain: Optional fallback forecast horizon domain.
        prediction_cutoff_timestamp_ns: Optional cutoff timestamp in nanoseconds. When specified,
            only observations with timestamp_ns <= prediction_cutoff_timestamp_ns are evaluated,
            strictly preventing future-data leakage across temporal evaluation boundaries.
        custom_parameters: Additional algorithm-specific configuration parameters.
    """

    min_observations: int = 2
    slope_threshold: float = 1e-9
    use_ols_slope: bool = False
    default_domain: Optional[HorizonDomain] = None
    prediction_cutoff_timestamp_ns: Optional[int] = None
    custom_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_observations < 1:
            raise ValueError(f"min_observations must be >= 1, got {self.min_observations}.")
        if self.slope_threshold < 0.0:
            raise ValueError(f"slope_threshold must be non-negative, got {self.slope_threshold}.")
        if self.default_domain is not None and not isinstance(self.default_domain, HorizonDomain):
            raise TypeError(f"Expected HorizonDomain, got {type(self.default_domain).__name__}.")
        if self.prediction_cutoff_timestamp_ns is not None:
            if not isinstance(self.prediction_cutoff_timestamp_ns, int) or self.prediction_cutoff_timestamp_ns < 0:
                raise ValueError(
                    f"prediction_cutoff_timestamp_ns must be a non-negative integer, got {self.prediction_cutoff_timestamp_ns}."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serializes estimator configuration to dictionary."""
        return {
            "min_observations": self.min_observations,
            "slope_threshold": self.slope_threshold,
            "use_ols_slope": self.use_ols_slope,
            "default_domain": self.default_domain.value if self.default_domain else None,
            "prediction_cutoff_timestamp_ns": self.prediction_cutoff_timestamp_ns,
            "custom_parameters": dict(self.custom_parameters),
        }


class AbstractRULEstimator(ABC):
    """Abstract base class for deterministic, model-based RUL estimators.

    All RUL estimators must be stateless, deterministic, side-effect free,
    and operate strictly on explicitly supplied immutable trajectories and EOL definitions.
    """

    @property
    @abstractmethod
    def method_identity(self) -> PrognosticMethodIdentity:
        """Returns the formal method identity and version metadata for this estimator."""
        pass

    @property
    @abstractmethod
    def supported_quantities(self) -> Sequence[TrajectoryQuantityType]:
        """Returns the sequence of trajectory quantities supported by this estimator."""
        pass

    @property
    @abstractmethod
    def supported_eol_criteria(self) -> Sequence[EOLCriterionType]:
        """Returns the sequence of End-of-Life criteria supported by this estimator."""
        pass

    @property
    @abstractmethod
    def supported_domains(self) -> Sequence[HorizonDomain]:
        """Returns the sequence of prediction horizon domains supported by this estimator."""
        pass

    @abstractmethod
    def estimate(
        self,
        trajectory: DegradationTrajectory,
        eol_definition: EOLDefinition,
        horizon: Optional[PrognosticHorizon] = None,
        config: Optional[EstimatorConfig] = None,
    ) -> RULPredictionResult:
        """Evaluates an RUL prediction from an immutable historical degradation trajectory.

        Args:
            trajectory: Input historical degradation trajectory.
            eol_definition: Configured End-of-Life threshold criterion.
            horizon: Optional target forecast horizon specifying domain and unit.
            config: Optional estimator configuration overriding defaults.

        Returns:
            Immutable RULPredictionResult with explicit validity status and extrapolation metadata.
        """
        pass
