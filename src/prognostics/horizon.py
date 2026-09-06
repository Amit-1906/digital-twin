"""Prognostic Horizon and Projection Domain Contracts.

Defines the physical or operational domains in which Remaining Useful Life (RUL)
and degradation trajectories are projected (time, cycles, throughput, custom).
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional

from src.prognostics.types import HorizonDomain


@dataclass(frozen=True)
class PrognosticHorizon:
    """Immutable contract defining the projection span and measurement domain for RUL predictions."""

    domain: HorizonDomain
    horizon_value: float
    unit: str
    start_value: float = 0.0
    description: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.domain, HorizonDomain):
            raise TypeError(f"Expected HorizonDomain, got {type(self.domain).__name__}.")
        if not isinstance(self.horizon_value, (int, float)) or math.isnan(self.horizon_value) or math.isinf(self.horizon_value):
            raise ValueError(f"horizon_value must be a finite float, got {self.horizon_value}.")
        if self.horizon_value <= 0.0:
            raise ValueError(f"horizon_value must be strictly positive, got {self.horizon_value}.")
        if not isinstance(self.start_value, (int, float)) or math.isnan(self.start_value) or math.isinf(self.start_value):
            raise ValueError(f"start_value must be a finite float, got {self.start_value}.")
        if self.start_value < 0.0:
            raise ValueError(f"start_value must be non-negative, got {self.start_value}.")
        if not self.unit or not isinstance(self.unit, str):
            raise ValueError("unit must be a non-empty string.")

    @property
    def end_value(self) -> float:
        """Returns the absolute target projection value (start_value + horizon_value)."""
        return self.start_value + self.horizon_value

    def to_dict(self) -> dict[str, Any]:
        """Serializes prognostic horizon contract to dictionary."""
        return {
            "domain": self.domain.value,
            "horizon_value": round(float(self.horizon_value), 6),
            "unit": self.unit,
            "start_value": round(float(self.start_value), 6),
            "end_value": round(float(self.end_value), 6),
            "description": self.description,
            "metadata": dict(self.metadata),
        }
