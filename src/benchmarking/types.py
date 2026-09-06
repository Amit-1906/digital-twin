"""Metrological Data Types, Provenance Contracts, and Measurement Containers for Battery Benchmarking.

Defines the core, battery-agnostic, and hardware-agnostic value objects for tracking reference
measurements, measurement uncertainty, orthogonal provenance classification, and experimental events.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional, Sequence, Tuple


class ReferenceSourceType(str, Enum):
    """Physical or synthetic origin of reference observations."""

    LABORATORY_METROLOGY = "LABORATORY_METROLOGY"           # Calibrated metrology cyclers, precision shunts, environmental chambers
    CALIBRATED_TESTBENCH = "CALIBRATED_TESTBENCH"           # Dedicated HIL testbench, external calibrated DAQ
    FIELD_BMS_TELEMETRY = "FIELD_BMS_TELEMETRY"             # Raw on-pack physical sensors
    BMS_FIRMWARE = "BMS_FIRMWARE"                           # Microcontroller firmware algorithms/estimates
    PHYSICS_SIMULATION_MODEL = "PHYSICS_SIMULATION_MODEL"   # High-order electrochemical/thermal physics simulation
    SYNTHETIC_MATHEMATICAL_GENERATOR = "SYNTHETIC_MATHEMATICAL_GENERATOR" # Analytical profiles, synthetic noise generators
    OTHER = "OTHER"                                         # Custom / unlisted measurement source


class CalibrationStatus(str, Enum):
    """Metrological calibration status of the measurement source."""

    CERTIFIED_CALIBRATED = "CERTIFIED_CALIBRATED" # Valid ISO/IEC 17025 or factory calibration certificate
    USER_CALIBRATED = "USER_CALIBRATED"           # Multi-point calibration performed by operator
    FACTORY_NOMINAL = "FACTORY_NOMINAL"           # Nominal unverified factory sensor spec
    UNCALIBRATED = "UNCALIBRATED"                 # Raw uncalibrated sensor/signal
    UNKNOWN = "UNKNOWN"                           # Calibration history not recorded


class EvidenceRole(str, Enum):
    """Formal evidentiary role of a signal in benchmarking and validation."""

    PHYSICAL_REFERENCE = "PHYSICAL_REFERENCE"         # Physical ground truth from calibrated instrumentation
    FIELD_OBSERVATION = "FIELD_OBSERVATION"           # Field operating telemetry for operational tracking
    BMS_ESTIMATE_BASELINE = "BMS_ESTIMATE_BASELINE"   # BMS firmware estimate for cross-comparison (unvalidated)
    SYNTHETIC_REFERENCE = "SYNTHETIC_REFERENCE"       # Mathematical simulation for algorithmic verification
    TWIN_ESTIMATE = "TWIN_ESTIMATE"                   # Evaluated TwinVolt state observer output
    MODEL_OUTPUT = "MODEL_OUTPUT"                     # Evaluated TwinVolt physics/ECM model output


class BenchmarkSignalType(str, Enum):
    """Physical or derived domain signal classification."""

    VOLTAGE = "VOLTAGE"
    CURRENT = "CURRENT"
    TEMPERATURE = "TEMPERATURE"
    SOC = "SOC"
    SOH = "SOH"
    INTERNAL_RESISTANCE = "INTERNAL_RESISTANCE"
    POWER = "POWER"
    ENERGY = "ENERGY"
    CAPACITY = "CAPACITY"
    POLARIZATION_VOLTAGE = "POLARIZATION_VOLTAGE"
    HEAT_GENERATION_RATE = "HEAT_GENERATION_RATE"
    CUSTOM = "CUSTOM"


class BenchmarkSemanticClass(str, Enum):
    """Semantic evaluation class of a benchmarking comparison."""

    PHYSICAL_REFERENCE_VALIDATION = "PHYSICAL_REFERENCE_VALIDATION" # TwinVolt vs. Calibrated Physical Reference
    FIELD_BMS_TRACKING = "FIELD_BMS_TRACKING"                       # TwinVolt vs. Field Raw BMS Sensors
    BMS_ESTIMATE_CROSS_CHECK = "BMS_ESTIMATE_CROSS_CHECK"           # TwinVolt vs. BMS Firmware Estimates (Unvalidated)
    ALGORITHMIC_SYNTHETIC_VERIFICATION = "ALGORITHMIC_SYNTHETIC_VERIFICATION" # TwinVolt vs. Synthetic Physics Models


class ReferenceDataQuality(str, Enum):
    """Data quality and validity status of reference measurements and frames."""

    VALID = "VALID"
    MISSING = "MISSING"
    INVALID = "INVALID"
    UNAVAILABLE = "UNAVAILABLE"
    INSUFFICIENT_REFERENCE = "INSUFFICIENT_REFERENCE"
    DATA_QUALITY_FAILED = "DATA_QUALITY_FAILED"


@dataclass(frozen=True)
class ReferenceProvenance:
    """Immutable provenance record tracking measurement origin, calibration, and evidentiary role.

    Follows the three-way orthogonal classification model separating source type, calibration
    quality, and evidentiary purpose.
    """

    provenance_id: str
    source_type: ReferenceSourceType
    calibration_status: CalibrationStatus
    evidence_role: EvidenceRole
    measurement_method: str
    instrument_identifier: str = ""
    calibration_expiry_date: Optional[str] = None
    facility_or_environment: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provenance_id or not isinstance(self.provenance_id, str):
            raise ValueError("provenance_id must be a non-empty string.")
        if not isinstance(self.source_type, ReferenceSourceType):
            raise TypeError(f"Expected ReferenceSourceType, got {type(self.source_type).__name__}.")
        if not isinstance(self.calibration_status, CalibrationStatus):
            raise TypeError(f"Expected CalibrationStatus, got {type(self.calibration_status).__name__}.")
        if not isinstance(self.evidence_role, EvidenceRole):
            raise TypeError(f"Expected EvidenceRole, got {type(self.evidence_role).__name__}.")
        if not self.measurement_method or not isinstance(self.measurement_method, str):
            raise ValueError("measurement_method must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes reference provenance to a deterministic dictionary."""
        return {
            "provenance_id": self.provenance_id,
            "source_type": self.source_type.value,
            "calibration_status": self.calibration_status.value,
            "evidence_role": self.evidence_role.value,
            "measurement_method": self.measurement_method,
            "instrument_identifier": self.instrument_identifier,
            "calibration_expiry_date": self.calibration_expiry_date,
            "facility_or_environment": self.facility_or_environment,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReferenceMeasurement:
    """Single reference observation carrying measurement value, unit, and quantified uncertainty.

    Represents a physical or synthetic ground truth point with optional absolute or relative
    measurement uncertainty, timestamp preservation, and provenance tracking.
    """

    value: float
    unit: str
    timestamp_ns: int
    uncertainty_absolute: Optional[float] = None
    uncertainty_relative_fraction: Optional[float] = None
    provenance: Optional[ReferenceProvenance] = None
    signal_type: BenchmarkSignalType = BenchmarkSignalType.CUSTOM
    quality_status: ReferenceDataQuality = ReferenceDataQuality.VALID
    observed_at_ns: Optional[int] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or math.isnan(self.value) or math.isinf(self.value):
            raise ValueError(f"Measurement value must be a finite float, got {self.value}.")
        if not self.unit or not isinstance(self.unit, str):
            raise ValueError("Measurement unit must be a non-empty string.")
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be a non-negative integer, got {self.timestamp_ns}.")
        if self.observed_at_ns is not None:
            if not isinstance(self.observed_at_ns, int) or self.observed_at_ns < 0:
                raise ValueError(f"observed_at_ns must be a non-negative integer, got {self.observed_at_ns}.")

        if self.uncertainty_absolute is not None:
            if not isinstance(self.uncertainty_absolute, (int, float)) or math.isnan(self.uncertainty_absolute) or math.isinf(self.uncertainty_absolute):
                raise ValueError(f"uncertainty_absolute must be a finite float, got {self.uncertainty_absolute}.")
            if self.uncertainty_absolute < 0.0:
                raise ValueError(f"uncertainty_absolute must be non-negative, got {self.uncertainty_absolute}.")

        if self.uncertainty_relative_fraction is not None:
            if not isinstance(self.uncertainty_relative_fraction, (int, float)) or math.isnan(self.uncertainty_relative_fraction) or math.isinf(self.uncertainty_relative_fraction):
                raise ValueError(f"uncertainty_relative_fraction must be a finite float, got {self.uncertainty_relative_fraction}.")
            if self.uncertainty_relative_fraction < 0.0:
                raise ValueError(f"uncertainty_relative_fraction must be non-negative, got {self.uncertainty_relative_fraction}.")

        if not isinstance(self.signal_type, BenchmarkSignalType):
            raise TypeError(f"Expected BenchmarkSignalType, got {type(self.signal_type).__name__}.")
        if not isinstance(self.quality_status, ReferenceDataQuality):
            raise TypeError(f"Expected ReferenceDataQuality, got {type(self.quality_status).__name__}.")

    def effective_uncertainty(self) -> float:
        """Computes the effective absolute standard uncertainty bound in measurement units.

        If absolute uncertainty is provided, it is returned directly. If relative fraction is
        provided, it is multiplied by the magnitude of the measured value. If both are provided,
        the maximum is returned. If neither is specified, returns 0.0.
        """
        abs_u = self.uncertainty_absolute if self.uncertainty_absolute is not None else None
        rel_u = (abs(self.value * self.uncertainty_relative_fraction)
                 if self.uncertainty_relative_fraction is not None else None)

        if abs_u is not None and rel_u is not None:
            return max(abs_u, rel_u)
        if abs_u is not None:
            return abs_u
        if rel_u is not None:
            return rel_u
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serializes reference measurement to a deterministic dictionary."""
        return {
            "value": self.value,
            "unit": self.unit,
            "timestamp_ns": self.timestamp_ns,
            "uncertainty_absolute": self.uncertainty_absolute,
            "uncertainty_relative_fraction": self.uncertainty_relative_fraction,
            "effective_uncertainty": self.effective_uncertainty(),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "signal_type": self.signal_type.value,
            "quality_status": self.quality_status.value,
            "observed_at_ns": self.observed_at_ns,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReferenceSignalChannel:
    """Metadata specification for a reference signal channel in an experimental dataset."""

    signal_name: str
    signal_type: BenchmarkSignalType
    unit: str
    provenance: ReferenceProvenance
    nominal_sampling_rate_hz: Optional[float] = None
    nominal_uncertainty_absolute: Optional[float] = None
    nominal_uncertainty_relative_fraction: Optional[float] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signal_name or not isinstance(self.signal_name, str):
            raise ValueError("signal_name must be a non-empty string.")
        if not isinstance(self.signal_type, BenchmarkSignalType):
            raise TypeError(f"Expected BenchmarkSignalType, got {type(self.signal_type).__name__}.")
        if not self.unit or not isinstance(self.unit, str):
            raise ValueError("unit must be a non-empty string.")
        if not isinstance(self.provenance, ReferenceProvenance):
            raise TypeError(f"Expected ReferenceProvenance, got {type(self.provenance).__name__}.")
        if self.nominal_sampling_rate_hz is not None:
            if not isinstance(self.nominal_sampling_rate_hz, (int, float)) or math.isnan(self.nominal_sampling_rate_hz) or math.isinf(self.nominal_sampling_rate_hz) or self.nominal_sampling_rate_hz <= 0.0:
                raise ValueError(f"nominal_sampling_rate_hz must be strictly positive, got {self.nominal_sampling_rate_hz}.")
        if self.nominal_uncertainty_absolute is not None:
            if not isinstance(self.nominal_uncertainty_absolute, (int, float)) or math.isnan(self.nominal_uncertainty_absolute) or math.isinf(self.nominal_uncertainty_absolute) or self.nominal_uncertainty_absolute < 0.0:
                raise ValueError(f"nominal_uncertainty_absolute must be non-negative, got {self.nominal_uncertainty_absolute}.")
        if self.nominal_uncertainty_relative_fraction is not None:
            if not isinstance(self.nominal_uncertainty_relative_fraction, (int, float)) or math.isnan(self.nominal_uncertainty_relative_fraction) or math.isinf(self.nominal_uncertainty_relative_fraction) or self.nominal_uncertainty_relative_fraction < 0.0:
                raise ValueError(f"nominal_uncertainty_relative_fraction must be non-negative, got {self.nominal_uncertainty_relative_fraction}.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes reference signal channel metadata to a dictionary."""
        return {
            "signal_name": self.signal_name,
            "signal_type": self.signal_type.value,
            "unit": self.unit,
            "provenance": self.provenance.to_dict(),
            "nominal_sampling_rate_hz": self.nominal_sampling_rate_hz,
            "nominal_uncertainty_absolute": self.nominal_uncertainty_absolute,
            "nominal_uncertainty_relative_fraction": self.nominal_uncertainty_relative_fraction,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExperimentalGroundTruthEvent:
    """Documented physical condition or known fault injected during a controlled experiment."""

    event_id: str
    start_timestamp_ns: int
    end_timestamp_ns: Optional[int] = None
    event_type: str = ""
    target_component: str = ""
    injected_magnitude: Optional[float] = None
    injected_unit: Optional[str] = None
    expected_hypothesis_id: Optional[str] = None
    description: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not isinstance(self.event_id, str):
            raise ValueError("event_id must be a non-empty string.")
        if not isinstance(self.start_timestamp_ns, int) or self.start_timestamp_ns < 0:
            raise ValueError(f"start_timestamp_ns must be a non-negative integer, got {self.start_timestamp_ns}.")
        if self.end_timestamp_ns is not None:
            if not isinstance(self.end_timestamp_ns, int) or self.end_timestamp_ns < self.start_timestamp_ns:
                raise ValueError(
                    f"end_timestamp_ns ({self.end_timestamp_ns}) cannot be before "
                    f"start_timestamp_ns ({self.start_timestamp_ns})."
                )
        if not self.event_type or not isinstance(self.event_type, str):
            raise ValueError("event_type must be a non-empty string.")
        if self.injected_magnitude is not None:
            if not isinstance(self.injected_magnitude, (int, float)) or math.isnan(self.injected_magnitude) or math.isinf(self.injected_magnitude):
                raise ValueError(f"injected_magnitude must be a finite float, got {self.injected_magnitude}.")

    @property
    def duration_seconds(self) -> Optional[float]:
        """Returns the event duration in fractional seconds, or None if ongoing."""
        if self.end_timestamp_ns is not None:
            return (self.end_timestamp_ns - self.start_timestamp_ns) / 1_000_000_000.0
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serializes ground truth event to a dictionary."""
        return {
            "event_id": self.event_id,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "duration_seconds": self.duration_seconds,
            "event_type": self.event_type,
            "target_component": self.target_component,
            "injected_magnitude": self.injected_magnitude,
            "injected_unit": self.injected_unit,
            "expected_hypothesis_id": self.expected_hypothesis_id,
            "description": self.description,
            "metadata": dict(self.metadata),
        }
