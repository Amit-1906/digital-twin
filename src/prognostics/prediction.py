"""Prognostic Prediction, EOL Definition, and Uncertainty Contracts.

Defines immutable value objects for Remaining Useful Life (RUL) estimation results,
method identity/versioning, configurable End-of-Life (EOL) boundary criteria,
and rigorous prediction uncertainty specifications.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional

from src.prognostics.state import PrognosticOperatingContext
from src.prognostics.types import (
    ComparisonOperator,
    EOLCriterionType,
    HorizonDomain,
    PrognosticStatus,
    UncertaintyType,
)


@dataclass(frozen=True)
class EOLDefinition:
    """Configurable End-of-Life (EOL) degradation boundary criterion."""

    criterion_id: str
    criterion_type: EOLCriterionType
    threshold_value: float
    unit: str
    comparison_operator: ComparisonOperator = ComparisonOperator.LESS_THAN_OR_EQUAL
    name: str = ""
    description: Optional[str] = None
    version: str = "1.0.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.criterion_id or not isinstance(self.criterion_id, str):
            raise ValueError("criterion_id must be a non-empty string.")
        if not isinstance(self.criterion_type, EOLCriterionType):
            raise TypeError(f"Expected EOLCriterionType, got {type(self.criterion_type).__name__}.")
        if not isinstance(self.threshold_value, (int, float)) or math.isnan(self.threshold_value) or math.isinf(self.threshold_value):
            raise ValueError(f"threshold_value must be a finite float, got {self.threshold_value}.")
        if not self.unit or not isinstance(self.unit, str):
            raise ValueError("unit must be a non-empty string.")
        if not isinstance(self.comparison_operator, ComparisonOperator):
            raise TypeError(f"Expected ComparisonOperator, got {type(self.comparison_operator).__name__}.")
        if not self.version or not isinstance(self.version, str):
            raise ValueError("version must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes EOL definition to dictionary."""
        return {
            "criterion_id": self.criterion_id,
            "criterion_type": self.criterion_type.value,
            "threshold_value": round(float(self.threshold_value), 6),
            "unit": self.unit,
            "comparison_operator": self.comparison_operator.value,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PrognosticMethodIdentity:
    """Identity, versioning, and parameterization metadata for a prognostic algorithm or model."""

    method_id: str
    method_version: str
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    algorithm_paradigm: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method_id or not isinstance(self.method_id, str):
            raise ValueError("method_id must be a non-empty string.")
        if not self.method_version or not isinstance(self.method_version, str):
            raise ValueError("method_version must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes prognostic method identity to dictionary."""
        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "algorithm_paradigm": self.algorithm_paradigm,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class PredictionUncertainty:
    """Rigorous uncertainty specification for a prognostic prediction."""

    uncertainty_type: UncertaintyType = UncertaintyType.DETERMINISTIC_NO_UNCERTAINTY
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    coverage_parameter: Optional[float] = None
    standard_deviation: Optional[float] = None
    unit: str = ""
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.uncertainty_type, UncertaintyType):
            raise TypeError(f"Expected UncertaintyType, got {type(self.uncertainty_type).__name__}.")

        for name, val in [
            ("lower_bound", self.lower_bound),
            ("upper_bound", self.upper_bound),
            ("coverage_parameter", self.coverage_parameter),
            ("standard_deviation", self.standard_deviation),
        ]:
            if val is not None:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"{name} must be a finite float, got {val}.")

        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError(
                    f"lower_bound ({self.lower_bound}) cannot exceed upper_bound ({self.upper_bound})."
                )

        if self.standard_deviation is not None and self.standard_deviation < 0.0:
            raise ValueError(f"standard_deviation must be non-negative, got {self.standard_deviation}.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes prediction uncertainty to dictionary."""
        return {
            "uncertainty_type": self.uncertainty_type.value,
            "lower_bound": round(float(self.lower_bound), 6) if self.lower_bound is not None else None,
            "upper_bound": round(float(self.upper_bound), 6) if self.upper_bound is not None else None,
            "coverage_parameter": (
                round(float(self.coverage_parameter), 4) if self.coverage_parameter is not None else None
            ),
            "standard_deviation": (
                round(float(self.standard_deviation), 6) if self.standard_deviation is not None else None
            ),
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True)
class RULPredictionResult:
    """Immutable Remaining Useful Life (RUL) prediction result container.

    Enforces strict validity invariants:
    - When status == VALID: rul_value must be a finite non-negative float.
    - When status != VALID: rul_value must be None (preventing ambiguous RUL=0 placeholders).
    """

    prediction_id: str
    prediction_timestamp_ns: int
    created_at_utc: str
    system_id: str
    status: PrognosticStatus
    rul_value: Optional[float]
    rul_unit: str
    rul_domain: HorizonDomain
    uncertainty: Optional[PredictionUncertainty] = None
    eol_definition: Optional[EOLDefinition] = None
    method_identity: Optional[PrognosticMethodIdentity] = None
    operating_context: Optional[PrognosticOperatingContext] = None
    input_state_snapshot_id: Optional[str] = None
    diagnostic_message: Optional[str] = None
    reproducibility_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prediction_id or not isinstance(self.prediction_id, str):
            raise ValueError("prediction_id must be a non-empty string.")
        if not self.system_id or not isinstance(self.system_id, str):
            raise ValueError("system_id must be a non-empty string.")
        if not isinstance(self.prediction_timestamp_ns, int) or self.prediction_timestamp_ns < 0:
            raise ValueError(
                f"prediction_timestamp_ns must be a non-negative integer, got {self.prediction_timestamp_ns}."
            )
        if not self.created_at_utc or not isinstance(self.created_at_utc, str):
            raise ValueError("created_at_utc must be a non-empty string.")
        if not isinstance(self.status, PrognosticStatus):
            raise TypeError(f"Expected PrognosticStatus, got {type(self.status).__name__}.")
        if not isinstance(self.rul_domain, HorizonDomain):
            raise TypeError(f"Expected HorizonDomain, got {type(self.rul_domain).__name__}.")
        if not self.rul_unit or not isinstance(self.rul_unit, str):
            raise ValueError("rul_unit must be a non-empty string.")

        # Invariant: VALID prediction requires valid non-negative finite float
        if self.status == PrognosticStatus.VALID:
            if self.rul_value is None:
                raise ValueError("rul_value cannot be None when status is VALID.")
            if not isinstance(self.rul_value, (int, float)) or math.isnan(self.rul_value) or math.isinf(self.rul_value):
                raise ValueError(f"rul_value must be a finite float, got {self.rul_value}.")
            if self.rul_value < 0.0:
                raise ValueError(f"rul_value must be non-negative, got {self.rul_value}.")

            # If uncertainty bounds are provided, verify ordering consistency
            if self.uncertainty is not None:
                if self.uncertainty.lower_bound is not None and self.rul_value < self.uncertainty.lower_bound:
                    raise ValueError(
                        f"rul_value ({self.rul_value}) cannot be less than uncertainty lower_bound ({self.uncertainty.lower_bound})."
                    )
                if self.uncertainty.upper_bound is not None and self.rul_value > self.uncertainty.upper_bound:
                    raise ValueError(
                        f"rul_value ({self.rul_value}) cannot exceed uncertainty upper_bound ({self.uncertainty.upper_bound})."
                    )
        else:
            # Invariant: Non-VALID prediction MUST have rul_value = None
            if self.rul_value is not None:
                raise ValueError(
                    f"rul_value must be None when status is {self.status.value} (RUL=0 must not be used as an unavailable placeholder)."
                )

        if self.uncertainty is not None and not isinstance(self.uncertainty, PredictionUncertainty):
            raise TypeError(f"Expected PredictionUncertainty, got {type(self.uncertainty).__name__}.")
        if self.eol_definition is not None and not isinstance(self.eol_definition, EOLDefinition):
            raise TypeError(f"Expected EOLDefinition, got {type(self.eol_definition).__name__}.")
        if self.method_identity is not None and not isinstance(self.method_identity, PrognosticMethodIdentity):
            raise TypeError(f"Expected PrognosticMethodIdentity, got {type(self.method_identity).__name__}.")
        if self.operating_context is not None and not isinstance(self.operating_context, PrognosticOperatingContext):
            raise TypeError(f"Expected PrognosticOperatingContext, got {type(self.operating_context).__name__}.")

    @property
    def is_valid(self) -> bool:
        """Returns True if the prediction is valid and has a non-None RUL point estimate."""
        return self.status == PrognosticStatus.VALID and self.rul_value is not None

    def to_dict(self) -> dict[str, Any]:
        """Serializes RUL prediction result to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "prediction_timestamp_ns": self.prediction_timestamp_ns,
            "created_at_utc": self.created_at_utc,
            "system_id": self.system_id,
            "status": self.status.value,
            "is_valid": self.is_valid,
            "rul_value": round(float(self.rul_value), 6) if self.rul_value is not None else None,
            "rul_unit": self.rul_unit,
            "rul_domain": self.rul_domain.value,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "eol_definition": self.eol_definition.to_dict() if self.eol_definition else None,
            "method_identity": self.method_identity.to_dict() if self.method_identity else None,
            "operating_context": self.operating_context.to_dict() if self.operating_context else None,
            "input_state_snapshot_id": self.input_state_snapshot_id,
            "diagnostic_message": self.diagnostic_message,
            "reproducibility_metadata": dict(self.reproducibility_metadata),
        }
