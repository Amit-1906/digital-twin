"""Value Provenance and Strongly-Typed Quantity Containers for Battery Prognostics.

Tracks evidentiary origin, sensor vs. estimated vs. modeled sources, calibration/quality flags,
and explicit availability status for all physical and parametric prognostic inputs.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional

from src.prognostics.types import QuantityAvailability, ValueOriginType


@dataclass(frozen=True)
class PrognosticValueProvenance:
    """Immutable evidentiary provenance record tracking the origin and metrology of a prognostic value."""

    origin_type: ValueOriginType = ValueOriginType.UNKNOWN
    source_identifier: Optional[str] = None
    timestamp_ns: Optional[int] = None
    quality_rating: Optional[str] = None
    measurement_uncertainty: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.origin_type, ValueOriginType):
            raise TypeError(f"Expected ValueOriginType, got {type(self.origin_type).__name__}.")
        if self.timestamp_ns is not None:
            if not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
                raise ValueError(f"timestamp_ns must be a non-negative integer, got {self.timestamp_ns}.")
        if self.measurement_uncertainty is not None:
            if (
                not isinstance(self.measurement_uncertainty, (int, float))
                or math.isnan(self.measurement_uncertainty)
                or math.isinf(self.measurement_uncertainty)
                or self.measurement_uncertainty < 0.0
            ):
                raise ValueError(
                    f"measurement_uncertainty must be a non-negative finite float, got {self.measurement_uncertainty}."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serializes value provenance record to dictionary."""
        return {
            "origin_type": self.origin_type.value,
            "source_identifier": self.source_identifier,
            "timestamp_ns": self.timestamp_ns,
            "quality_rating": self.quality_rating,
            "measurement_uncertainty": (
                round(float(self.measurement_uncertainty), 8)
                if self.measurement_uncertainty is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PrognosticQuantity:
    """Strongly-typed numeric prognostic scalar with explicit unit, provenance, and availability semantics."""

    value: Optional[float]
    unit: str
    availability: QuantityAvailability = QuantityAvailability.AVAILABLE
    provenance: Optional[PrognosticValueProvenance] = None
    name: str = ""
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.availability, QuantityAvailability):
            raise TypeError(f"Expected QuantityAvailability, got {type(self.availability).__name__}.")
        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string.")

        if self.availability == QuantityAvailability.AVAILABLE:
            if self.value is None:
                raise ValueError("value cannot be None when availability is AVAILABLE.")
            if not isinstance(self.value, (int, float)) or math.isnan(self.value) or math.isinf(self.value):
                raise ValueError(f"value must be a finite float when AVAILABLE, got {self.value}.")
        else:
            if self.value is not None and (math.isnan(self.value) or math.isinf(self.value)):
                raise ValueError("Non-finite values (NaN, Inf) are not permitted.")

        if self.provenance is not None and not isinstance(self.provenance, PrognosticValueProvenance):
            raise TypeError(f"Expected PrognosticValueProvenance, got {type(self.provenance).__name__}.")

    @property
    def is_available(self) -> bool:
        """Returns True if the quantity is present and valid for prognostic consumption."""
        return self.availability == QuantityAvailability.AVAILABLE and self.value is not None

    def to_dict(self) -> dict[str, Any]:
        """Serializes prognostic quantity container to dictionary."""
        return {
            "name": self.name,
            "value": round(float(self.value), 6) if self.value is not None else None,
            "unit": self.unit,
            "availability": self.availability.value,
            "is_available": self.is_available,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "description": self.description,
        }
