"""Configurable Operating-Context Stratification for Battery Benchmarking.

Provides multi-dimensional partitioning of benchmark evaluations across operating regimes,
SOC windows, temperature ranges, current C-rates, voltage spans, and experimental phases
without hard-coded chemistry assumptions.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional, Sequence, Tuple


class ContextDimension(str, Enum):
    """Analytical stratification dimension."""

    OVERALL = "OVERALL"
    REGIME = "REGIME"
    SOC = "SOC"
    TEMPERATURE = "TEMPERATURE"
    CURRENT = "CURRENT"
    VOLTAGE = "VOLTAGE"
    PHASE = "PHASE"
    CUSTOM = "CUSTOM"


@dataclass(frozen=True)
class NumericRangeBucket:
    """Configurable numerical range bucket for slicing benchmark evaluations."""

    name: str
    min_value: float
    max_value: float
    inclusive_min: bool = True
    inclusive_max: bool = True

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("NumericRangeBucket name must be a non-empty string.")
        if not isinstance(self.min_value, (int, float)) or math.isnan(self.min_value) or math.isinf(self.min_value):
            raise ValueError(f"min_value must be a finite float, got {self.min_value}.")
        if not isinstance(self.max_value, (int, float)) or math.isnan(self.max_value) or math.isinf(self.max_value):
            raise ValueError(f"max_value must be a finite float, got {self.max_value}.")
        if self.min_value > self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) cannot exceed max_value ({self.max_value}) for bucket '{self.name}'."
            )

    def contains(self, value: float) -> bool:
        """Determines whether a scalar value falls within this bucket's numerical range."""
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return False

        lower_ok = (value >= self.min_value) if self.inclusive_min else (value > self.min_value)
        upper_ok = (value <= self.max_value) if self.inclusive_max else (value < self.max_value)
        return lower_ok and upper_ok

    def to_dict(self) -> dict[str, Any]:
        """Serializes range bucket to a dictionary."""
        return {
            "name": self.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "inclusive_min": self.inclusive_min,
            "inclusive_max": self.inclusive_max,
        }


@dataclass(frozen=True)
class ContextSliceIdentifier:
    """Unique key identifying an individual analytical context slice."""

    dimension: ContextDimension
    bucket_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ContextDimension):
            raise TypeError(f"Expected ContextDimension, got {type(self.dimension).__name__}.")
        if not self.bucket_name or not isinstance(self.bucket_name, str):
            raise ValueError("bucket_name must be a non-empty string.")

    def to_dict(self) -> dict[str, str]:
        """Serializes context slice identifier to dictionary."""
        return {
            "dimension": self.dimension.value,
            "bucket_name": self.bucket_name,
        }


@dataclass(frozen=True)
class StratificationConfig:
    """Configurable boundaries and dimensions governing benchmark stratification."""

    soc_buckets: Tuple[NumericRangeBucket, ...] = ()
    temperature_buckets_c: Tuple[NumericRangeBucket, ...] = ()
    current_crate_buckets: Tuple[NumericRangeBucket, ...] = ()
    voltage_buckets_v: Tuple[NumericRangeBucket, ...] = ()
    custom_phase_labels: Tuple[str, ...] = ()
    nominal_capacity_ah: Optional[float] = None
    rest_current_threshold_a: float = 0.05

    def __post_init__(self) -> None:
        for b in self.soc_buckets:
            if not isinstance(b, NumericRangeBucket):
                raise TypeError(f"Expected NumericRangeBucket in soc_buckets, got {type(b).__name__}.")
        for b in self.temperature_buckets_c:
            if not isinstance(b, NumericRangeBucket):
                raise TypeError(f"Expected NumericRangeBucket in temperature_buckets_c, got {type(b).__name__}.")
        for b in self.current_crate_buckets:
            if not isinstance(b, NumericRangeBucket):
                raise TypeError(f"Expected NumericRangeBucket in current_crate_buckets, got {type(b).__name__}.")
        for b in self.voltage_buckets_v:
            if not isinstance(b, NumericRangeBucket):
                raise TypeError(f"Expected NumericRangeBucket in voltage_buckets_v, got {type(b).__name__}.")

        if self.nominal_capacity_ah is not None:
            if not isinstance(self.nominal_capacity_ah, (int, float)) or math.isnan(self.nominal_capacity_ah) or math.isinf(self.nominal_capacity_ah) or self.nominal_capacity_ah <= 0.0:
                raise ValueError(f"nominal_capacity_ah must be strictly positive, got {self.nominal_capacity_ah}.")
        if self.rest_current_threshold_a < 0.0:
            raise ValueError(f"rest_current_threshold_a must be non-negative, got {self.rest_current_threshold_a}.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes stratification config to a dictionary."""
        return {
            "soc_buckets": [b.to_dict() for b in self.soc_buckets],
            "temperature_buckets_c": [b.to_dict() for b in self.temperature_buckets_c],
            "current_crate_buckets": [b.to_dict() for b in self.current_crate_buckets],
            "voltage_buckets_v": [b.to_dict() for b in self.voltage_buckets_v],
            "custom_phase_labels": list(self.custom_phase_labels),
            "nominal_capacity_ah": self.nominal_capacity_ah,
            "rest_current_threshold_a": self.rest_current_threshold_a,
        }


@dataclass(frozen=True)
class StratifiedContext:
    """Classified multi-dimensional context for a single observation or paired evaluation sample."""

    active_slices: Tuple[ContextSliceIdentifier, ...]
    operating_regime: Optional[str] = None
    soc_bucket: Optional[str] = None
    temperature_bucket: Optional[str] = None
    current_bucket: Optional[str] = None
    voltage_bucket: Optional[str] = None
    phase: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def matches_slice(self, slice_id: ContextSliceIdentifier) -> bool:
        """Checks if this context satisfies the given slice identifier."""
        if slice_id.dimension == ContextDimension.OVERALL:
            return True
        return slice_id in self.active_slices

    def to_dict(self) -> dict[str, Any]:
        """Serializes stratified context to a dictionary."""
        return {
            "active_slices": [s.to_dict() for s in self.active_slices],
            "operating_regime": self.operating_regime,
            "soc_bucket": self.soc_bucket,
            "temperature_bucket": self.temperature_bucket,
            "current_bucket": self.current_bucket,
            "voltage_bucket": self.voltage_bucket,
            "phase": self.phase,
            "metadata": dict(self.metadata),
        }


class OperatingContextStratifier:
    """Classifies evaluation samples into multi-dimensional analytical context slices."""

    @classmethod
    def get_default_config(
        cls,
        nominal_capacity_ah: Optional[float] = None,
    ) -> StratificationConfig:
        """Returns standard generic default stratification buckets (fully chemistry-agnostic)."""
        return StratificationConfig(
            soc_buckets=(
                NumericRangeBucket("LOW_SOC", 0.0, 0.20),
                NumericRangeBucket("MID_SOC", 0.20, 0.80),
                NumericRangeBucket("HIGH_SOC", 0.80, 1.00),
            ),
            temperature_buckets_c=(
                NumericRangeBucket("COLD_AMBIENT", -50.0, 10.0),
                NumericRangeBucket("ROOM_TEMPERATURE", 10.0, 35.0),
                NumericRangeBucket("ELEVATED_TEMPERATURE", 35.0, 90.0),
            ),
            current_crate_buckets=(
                NumericRangeBucket("REST", 0.0, 0.05),
                NumericRangeBucket("LOW_RATE", 0.05, 0.50),
                NumericRangeBucket("MEDIUM_RATE", 0.50, 1.50),
                NumericRangeBucket("HIGH_RATE", 1.50, 20.0),
            ),
            nominal_capacity_ah=nominal_capacity_ah,
        )

    @classmethod
    def classify(
        cls,
        current_a: Optional[float] = None,
        voltage_v: Optional[float] = None,
        temperature_c: Optional[float] = None,
        soc_fraction: Optional[float] = None,
        operating_context_hint: Optional[str] = None,
        phase_hint: Optional[str] = None,
        config: Optional[StratificationConfig] = None,
    ) -> StratifiedContext:
        """Classifies incoming sample telemetry into active context slices.

        Args:
            current_a: Battery pack or cell current in Amperes.
            voltage_v: Terminal voltage in Volts.
            temperature_c: Battery temperature in Celsius.
            soc_fraction: State of Charge fraction [0.0, 1.0].
            operating_context_hint: Optional pre-classified regime string (e.g. "CHARGE_CC").
            phase_hint: Optional custom experimental phase tag.
            config: Configurable stratification parameters.

        Returns:
            StratifiedContext value object.
        """
        cfg = config or cls.get_default_config()
        slices: list[ContextSliceIdentifier] = []

        # Always belongs to OVERALL
        slices.append(ContextSliceIdentifier(ContextDimension.OVERALL, "ALL"))

        # 1. Operating Regime
        regime = None
        if operating_context_hint:
            regime = operating_context_hint
        elif current_a is not None and not math.isnan(current_a) and not math.isinf(current_a):
            if abs(current_a) <= cfg.rest_current_threshold_a:
                regime = "REST"
            elif current_a < -cfg.rest_current_threshold_a:
                regime = "CHARGE"
            else:
                regime = "DISCHARGE"

        if regime:
            slices.append(ContextSliceIdentifier(ContextDimension.REGIME, regime))

        # 2. SOC Slicing
        matched_soc = None
        if soc_fraction is not None and not math.isnan(soc_fraction) and not math.isinf(soc_fraction):
            for bucket in cfg.soc_buckets:
                if bucket.contains(soc_fraction):
                    matched_soc = bucket.name
                    slices.append(ContextSliceIdentifier(ContextDimension.SOC, bucket.name))
                    break

        # 3. Temperature Slicing
        matched_temp = None
        if temperature_c is not None and not math.isnan(temperature_c) and not math.isinf(temperature_c):
            for bucket in cfg.temperature_buckets_c:
                if bucket.contains(temperature_c):
                    matched_temp = bucket.name
                    slices.append(ContextSliceIdentifier(ContextDimension.TEMPERATURE, bucket.name))
                    break

        # 4. Current / C-rate Slicing
        matched_curr = None
        if current_a is not None and not math.isnan(current_a) and not math.isinf(current_a):
            crate_val = abs(current_a) / cfg.nominal_capacity_ah if cfg.nominal_capacity_ah else abs(current_a)
            for bucket in cfg.current_crate_buckets:
                if bucket.contains(crate_val):
                    matched_curr = bucket.name
                    slices.append(ContextSliceIdentifier(ContextDimension.CURRENT, bucket.name))
                    break

        # 5. Voltage Slicing
        matched_volt = None
        if voltage_v is not None and not math.isnan(voltage_v) and not math.isinf(voltage_v):
            for bucket in cfg.voltage_buckets_v:
                if bucket.contains(voltage_v):
                    matched_volt = bucket.name
                    slices.append(ContextSliceIdentifier(ContextDimension.VOLTAGE, bucket.name))
                    break

        # 6. Custom Phase Slicing
        matched_phase = None
        if phase_hint:
            matched_phase = phase_hint
            slices.append(ContextSliceIdentifier(ContextDimension.PHASE, phase_hint))

        return StratifiedContext(
            active_slices=tuple(slices),
            operating_regime=regime,
            soc_bucket=matched_soc,
            temperature_bucket=matched_temp,
            current_bucket=matched_curr,
            voltage_bucket=matched_volt,
            phase=matched_phase,
        )
