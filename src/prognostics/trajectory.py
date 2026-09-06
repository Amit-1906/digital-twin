"""Degradation Observation, Trajectory Value Objects, and Trajectory Builder.

Defines immutable, multi-dimensional time-series representations of battery degradation
histories, supporting independent capacity, resistance, cycling, and throughput trajectories
with strict temporal ordering and availability tracking.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from src.prognostics.provenance import (
    PrognosticQuantity,
    PrognosticValueProvenance,
)
from src.prognostics.state import PrognosticOperatingContext
from src.prognostics.trajectory_types import (
    TrajectoryOrderingStatus,
    TrajectoryQuantityType,
)
from src.prognostics.types import QuantityAvailability, ValueOriginType


@dataclass(frozen=True)
class DegradationObservation:
    """Immutable multi-dimensional degradation and operating state observation at a discrete timestamp."""

    timestamp_ns: int
    created_at_utc: str

    # Capacity Quantities
    usable_capacity_ah: Optional[PrognosticQuantity] = None
    capacity_retention_fraction: Optional[PrognosticQuantity] = None

    # Resistance Quantities
    internal_resistance_r0_ohm: Optional[PrognosticQuantity] = None
    resistance_growth_ratio: Optional[PrognosticQuantity] = None

    # Cycling and Throughput Quantities
    equivalent_full_cycles: Optional[PrognosticQuantity] = None
    charge_throughput_ah: Optional[PrognosticQuantity] = None
    discharge_throughput_ah: Optional[PrognosticQuantity] = None
    total_energy_throughput_wh: Optional[PrognosticQuantity] = None

    # Calendar and Thermal Quantities
    calendar_age_seconds: Optional[PrognosticQuantity] = None
    cumulative_thermal_exposure_deg_c_hours: Optional[PrognosticQuantity] = None

    # Context and Provenance
    operating_context: Optional[PrognosticOperatingContext] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be a non-negative integer, got {self.timestamp_ns}.")
        if not self.created_at_utc or not isinstance(self.created_at_utc, str):
            raise ValueError("created_at_utc must be a non-empty string.")

        for name in [
            "usable_capacity_ah",
            "capacity_retention_fraction",
            "internal_resistance_r0_ohm",
            "resistance_growth_ratio",
            "equivalent_full_cycles",
            "charge_throughput_ah",
            "discharge_throughput_ah",
            "total_energy_throughput_wh",
            "calendar_age_seconds",
            "cumulative_thermal_exposure_deg_c_hours",
        ]:
            val = getattr(self, name)
            if val is not None and not isinstance(val, PrognosticQuantity):
                raise TypeError(f"Expected PrognosticQuantity for '{name}', got {type(val).__name__}.")

        if self.operating_context is not None and not isinstance(
            self.operating_context, PrognosticOperatingContext
        ):
            raise TypeError(
                f"Expected PrognosticOperatingContext, got {type(self.operating_context).__name__}."
            )

    def get_quantity(self, name: str) -> Optional[PrognosticQuantity]:
        """Retrieves the strongly-typed PrognosticQuantity container for a given field."""
        return getattr(self, name, None)

    def get_value(self, name: str) -> Optional[float]:
        """Retrieves the scalar float value if present and AVAILABLE, else None."""
        q = self.get_quantity(name)
        if q is not None and q.is_available:
            return q.value
        return None

    def is_quantity_available(self, name: str) -> bool:
        """Returns True if the specified field exists and is marked AVAILABLE."""
        q = self.get_quantity(name)
        return q is not None and q.is_available

    def available_quantities(self) -> list[str]:
        """Returns list of field names that are present and marked AVAILABLE in this observation."""
        fields = [
            "usable_capacity_ah",
            "capacity_retention_fraction",
            "internal_resistance_r0_ohm",
            "resistance_growth_ratio",
            "equivalent_full_cycles",
            "charge_throughput_ah",
            "discharge_throughput_ah",
            "total_energy_throughput_wh",
            "calendar_age_seconds",
            "cumulative_thermal_exposure_deg_c_hours",
        ]
        return [f for f in fields if self.is_quantity_available(f)]

    def to_dict(self) -> dict[str, Any]:
        """Serializes degradation observation to dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "created_at_utc": self.created_at_utc,
            "usable_capacity_ah": (
                self.usable_capacity_ah.to_dict() if self.usable_capacity_ah else None
            ),
            "capacity_retention_fraction": (
                self.capacity_retention_fraction.to_dict()
                if self.capacity_retention_fraction
                else None
            ),
            "internal_resistance_r0_ohm": (
                self.internal_resistance_r0_ohm.to_dict()
                if self.internal_resistance_r0_ohm
                else None
            ),
            "resistance_growth_ratio": (
                self.resistance_growth_ratio.to_dict()
                if self.resistance_growth_ratio
                else None
            ),
            "equivalent_full_cycles": (
                self.equivalent_full_cycles.to_dict()
                if self.equivalent_full_cycles
                else None
            ),
            "charge_throughput_ah": (
                self.charge_throughput_ah.to_dict()
                if self.charge_throughput_ah
                else None
            ),
            "discharge_throughput_ah": (
                self.discharge_throughput_ah.to_dict()
                if self.discharge_throughput_ah
                else None
            ),
            "total_energy_throughput_wh": (
                self.total_energy_throughput_wh.to_dict()
                if self.total_energy_throughput_wh
                else None
            ),
            "calendar_age_seconds": (
                self.calendar_age_seconds.to_dict()
                if self.calendar_age_seconds
                else None
            ),
            "cumulative_thermal_exposure_deg_c_hours": (
                self.cumulative_thermal_exposure_deg_c_hours.to_dict()
                if self.cumulative_thermal_exposure_deg_c_hours
                else None
            ),
            "available_quantities": self.available_quantities(),
            "operating_context": (
                self.operating_context.to_dict() if self.operating_context else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DegradationTrajectory:
    """Immutable sequence of historical degradation observations with temporal ordering analysis."""

    trajectory_id: str
    system_id: str
    quantity_type: TrajectoryQuantityType = TrajectoryQuantityType.MULTIVARIATE
    observations: Tuple[DegradationObservation, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trajectory_id or not isinstance(self.trajectory_id, str):
            raise ValueError("trajectory_id must be a non-empty string.")
        if not self.system_id or not isinstance(self.system_id, str):
            raise ValueError("system_id must be a non-empty string.")
        if not isinstance(self.quantity_type, TrajectoryQuantityType):
            raise TypeError(f"Expected TrajectoryQuantityType, got {type(self.quantity_type).__name__}.")

        for idx, obs in enumerate(self.observations):
            if not isinstance(obs, DegradationObservation):
                raise TypeError(
                    f"Expected DegradationObservation at index {idx}, got {type(obs).__name__}."
                )

    @property
    def total_observations(self) -> int:
        """Total number of observations in the trajectory."""
        return len(self.observations)

    @property
    def start_timestamp_ns(self) -> Optional[int]:
        """Timestamp of the first observation in nanoseconds, or None if empty."""
        return self.observations[0].timestamp_ns if self.observations else None

    @property
    def end_timestamp_ns(self) -> Optional[int]:
        """Timestamp of the last observation in nanoseconds, or None if empty."""
        return self.observations[-1].timestamp_ns if self.observations else None

    @property
    def duration_seconds(self) -> float:
        """Total elapsed observation duration in seconds."""
        if len(self.observations) < 2:
            return 0.0
        return (self.observations[-1].timestamp_ns - self.observations[0].timestamp_ns) / 1_000_000_000.0

    @property
    def ordering_status(self) -> TrajectoryOrderingStatus:
        """Evaluates timestamp monotonicity and ordering integrity across all observations."""
        n = len(self.observations)
        if n == 0:
            return TrajectoryOrderingStatus.EMPTY
        if n == 1:
            return TrajectoryOrderingStatus.SINGLE_POINT

        has_duplicate = False
        for i in range(1, n):
            t_prev = self.observations[i - 1].timestamp_ns
            t_curr = self.observations[i].timestamp_ns
            if t_curr < t_prev:
                return TrajectoryOrderingStatus.RETROGRADE_DETECTED
            if t_curr == t_prev:
                has_duplicate = True

        if has_duplicate:
            return TrajectoryOrderingStatus.MONOTONIC_WITH_DUPLICATES
        return TrajectoryOrderingStatus.STRICTLY_MONOTONIC

    def get_series(self, quantity_name: str) -> list[Tuple[int, float]]:
        """Extracts ordered (timestamp_ns, float_value) tuples for all observations where quantity is AVAILABLE."""
        series = []
        for obs in self.observations:
            val = obs.get_value(quantity_name)
            if val is not None:
                series.append((obs.timestamp_ns, val))
        return series

    def get_observations_with(self, quantity_name: str) -> list[DegradationObservation]:
        """Filters observations where the specified quantity is present and marked AVAILABLE."""
        return [obs for obs in self.observations if obs.is_quantity_available(quantity_name)]

    def to_dict(self) -> dict[str, Any]:
        """Serializes degradation trajectory to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "system_id": self.system_id,
            "quantity_type": self.quantity_type.value,
            "total_observations": self.total_observations,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "duration_seconds": round(self.duration_seconds, 6),
            "ordering_status": self.ordering_status.value,
            "observations": [obs.to_dict() for obs in self.observations],
            "metadata": dict(self.metadata),
        }


class TrajectoryBuilder:
    """Fluent builder for validating and assembling immutable DegradationTrajectory instances."""

    def __init__(
        self,
        trajectory_id: str,
        system_id: str,
        quantity_type: TrajectoryQuantityType = TrajectoryQuantityType.MULTIVARIATE,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._trajectory_id = trajectory_id
        self._system_id = system_id
        self._quantity_type = quantity_type
        self._metadata = dict(metadata or {})
        self._observations: list[DegradationObservation] = []

    def add_observation(self, observation: DegradationObservation) -> "TrajectoryBuilder":
        """Appends a single degradation observation."""
        if not isinstance(observation, DegradationObservation):
            raise TypeError(f"Expected DegradationObservation, got {type(observation).__name__}.")
        self._observations.append(observation)
        return self

    def add_observations(self, observations: Sequence[DegradationObservation]) -> "TrajectoryBuilder":
        """Appends a sequence of degradation observations."""
        for obs in observations:
            self.add_observation(obs)
        return self

    def build(self, sort_chronologically: bool = False) -> DegradationTrajectory:
        """Assembles and returns a validated, immutable DegradationTrajectory."""
        obs_list = list(self._observations)
        if sort_chronologically:
            obs_list.sort(key=lambda o: o.timestamp_ns)

        return DegradationTrajectory(
            trajectory_id=self._trajectory_id,
            system_id=self._system_id,
            quantity_type=self._quantity_type,
            observations=tuple(obs_list),
            metadata=dict(self._metadata),
        )
