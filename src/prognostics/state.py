"""Prognostic State Snapshot and Operating Context Contracts.

Defines immutable multi-dimensional battery state representations that capture all
measurable, estimated, and modeled degradation indicators available for prognostic analysis.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional

from src.prognostics.provenance import PrognosticQuantity


@dataclass(frozen=True)
class PrognosticOperatingContext:
    """Operating conditions, environmental parameters, and profile assumptions for prognostics."""

    ambient_temperature_c: Optional[float] = None
    cell_temperature_c: Optional[float] = None
    pack_current_a: Optional[float] = None
    c_rate: Optional[float] = None
    soc_fraction: Optional[float] = None
    pack_voltage_v: Optional[float] = None
    load_profile_identifier: Optional[str] = None
    duty_cycle_fraction: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in [
            ("ambient_temperature_c", self.ambient_temperature_c),
            ("cell_temperature_c", self.cell_temperature_c),
            ("pack_current_a", self.pack_current_a),
            ("c_rate", self.c_rate),
            ("soc_fraction", self.soc_fraction),
            ("pack_voltage_v", self.pack_voltage_v),
            ("duty_cycle_fraction", self.duty_cycle_fraction),
        ]:
            if val is not None:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise ValueError(f"{name} must be a finite float, got {val}.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes operating context to dictionary."""
        return {
            "ambient_temperature_c": (
                round(float(self.ambient_temperature_c), 3)
                if self.ambient_temperature_c is not None
                else None
            ),
            "cell_temperature_c": (
                round(float(self.cell_temperature_c), 3)
                if self.cell_temperature_c is not None
                else None
            ),
            "pack_current_a": (
                round(float(self.pack_current_a), 4)
                if self.pack_current_a is not None
                else None
            ),
            "c_rate": (
                round(float(self.c_rate), 4)
                if self.c_rate is not None
                else None
            ),
            "soc_fraction": (
                round(float(self.soc_fraction), 6)
                if self.soc_fraction is not None
                else None
            ),
            "pack_voltage_v": (
                round(float(self.pack_voltage_v), 4)
                if self.pack_voltage_v is not None
                else None
            ),
            "load_profile_identifier": self.load_profile_identifier,
            "duty_cycle_fraction": (
                round(float(self.duty_cycle_fraction), 4)
                if self.duty_cycle_fraction is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PrognosticStateSnapshot:
    """Immutable multi-dimensional degradation and operating state snapshot for prognostic evaluation."""

    snapshot_id: str
    system_id: str
    timestamp_ns: int
    created_at_utc: str

    # Capacity Fade Quantities
    usable_capacity_ah: Optional[PrognosticQuantity] = None
    nominal_capacity_ah: Optional[PrognosticQuantity] = None
    capacity_retention_fraction: Optional[PrognosticQuantity] = None

    # Impedance Growth Quantities
    internal_resistance_r0_ohm: Optional[PrognosticQuantity] = None
    baseline_resistance_r0_ohm: Optional[PrognosticQuantity] = None
    resistance_growth_ratio: Optional[PrognosticQuantity] = None

    # Cycling and Throughput Quantities
    equivalent_full_cycles: Optional[PrognosticQuantity] = None
    charge_throughput_ah: Optional[PrognosticQuantity] = None
    discharge_throughput_ah: Optional[PrognosticQuantity] = None
    total_energy_throughput_wh: Optional[PrognosticQuantity] = None

    # Age and Thermal Exposure Quantities
    calendar_age_seconds: Optional[PrognosticQuantity] = None
    cumulative_thermal_exposure_deg_c_hours: Optional[PrognosticQuantity] = None

    # Context and Provenance
    operating_context: Optional[PrognosticOperatingContext] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id or not isinstance(self.snapshot_id, str):
            raise ValueError("snapshot_id must be a non-empty string.")
        if not self.system_id or not isinstance(self.system_id, str):
            raise ValueError("system_id must be a non-empty string.")
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be a non-negative integer, got {self.timestamp_ns}.")
        if not self.created_at_utc or not isinstance(self.created_at_utc, str):
            raise ValueError("created_at_utc must be a non-empty string.")

        # Validate strongly-typed quantity fields
        for name in [
            "usable_capacity_ah",
            "nominal_capacity_ah",
            "capacity_retention_fraction",
            "internal_resistance_r0_ohm",
            "baseline_resistance_r0_ohm",
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
        """Retrieves a named prognostic quantity container if present."""
        return getattr(self, name, None)

    def is_quantity_available(self, name: str) -> bool:
        """Returns True if the specified quantity is present and marked AVAILABLE."""
        q = self.get_quantity(name)
        return q is not None and q.is_available

    def available_quantities(self) -> list[str]:
        """Returns list of all quantity field names that are currently available."""
        fields = [
            "usable_capacity_ah",
            "nominal_capacity_ah",
            "capacity_retention_fraction",
            "internal_resistance_r0_ohm",
            "baseline_resistance_r0_ohm",
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
        """Serializes prognostic state snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "system_id": self.system_id,
            "timestamp_ns": self.timestamp_ns,
            "created_at_utc": self.created_at_utc,
            "usable_capacity_ah": (
                self.usable_capacity_ah.to_dict() if self.usable_capacity_ah else None
            ),
            "nominal_capacity_ah": (
                self.nominal_capacity_ah.to_dict() if self.nominal_capacity_ah else None
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
            "baseline_resistance_r0_ohm": (
                self.baseline_resistance_r0_ohm.to_dict()
                if self.baseline_resistance_r0_ohm
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
