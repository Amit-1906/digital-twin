"""Dataset Abstraction, Observation Frames, and Builders for Battery Benchmarking.

Provides immutable, self-describing, and battery-agnostic dataset containers representing
multi-channel laboratory and experimental time-series observations, ground-truth events,
and dataset metadata.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from src.benchmarking.types import (
    BenchmarkSignalType,
    ExperimentalGroundTruthEvent,
    ReferenceDataQuality,
    ReferenceMeasurement,
    ReferenceSignalChannel,
)
from src.domain.battery.value_objects import (
    BatteryIdentification,
    CellConfiguration,
    PackConfiguration,
)


@dataclass(frozen=True)
class ReferenceObservationFrame:
    """A single synchronized time frame containing reference measurements, field observations, and BMS estimates.

    Represents one discrete temporal sample in an experimental dataset.
    """

    timestamp_ns: int
    observed_at_ns: Optional[int] = None
    sequence_number: Optional[int] = None
    reference_signals: Mapping[str, ReferenceMeasurement] = field(default_factory=dict)
    field_signals: Mapping[str, float] = field(default_factory=dict)
    bms_estimates: Mapping[str, float] = field(default_factory=dict)
    operating_context_hint: Optional[str] = None
    quality_status: ReferenceDataQuality = ReferenceDataQuality.VALID
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be a non-negative integer, got {self.timestamp_ns}.")
        if self.observed_at_ns is not None:
            if not isinstance(self.observed_at_ns, int) or self.observed_at_ns < 0:
                raise ValueError(f"observed_at_ns must be a non-negative integer, got {self.observed_at_ns}.")
        if self.sequence_number is not None:
            if not isinstance(self.sequence_number, int) or self.sequence_number < 0:
                raise ValueError(f"sequence_number must be a non-negative integer, got {self.sequence_number}.")

        # Validate reference signals map
        for sig_name, meas in self.reference_signals.items():
            if not isinstance(sig_name, str) or not sig_name:
                raise ValueError("Reference signal key must be a non-empty string.")
            if not isinstance(meas, ReferenceMeasurement):
                raise TypeError(f"Expected ReferenceMeasurement for '{sig_name}', got {type(meas).__name__}.")

        # Validate field signals map
        for sig_name, val in self.field_signals.items():
            if not isinstance(sig_name, str) or not sig_name:
                raise ValueError("Field signal key must be a non-empty string.")
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Field signal '{sig_name}' must be a finite float, got {val}.")

        # Validate BMS estimates map
        for sig_name, val in self.bms_estimates.items():
            if not isinstance(sig_name, str) or not sig_name:
                raise ValueError("BMS estimate key must be a non-empty string.")
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"BMS estimate '{sig_name}' must be a finite float, got {val}.")

        if not isinstance(self.quality_status, ReferenceDataQuality):
            raise TypeError(f"Expected ReferenceDataQuality, got {type(self.quality_status).__name__}.")

    @property
    def timestamp_seconds(self) -> float:
        """Returns the frame timestamp in fractional seconds."""
        return self.timestamp_ns / 1_000_000_000.0

    def get_reference(self, signal_name: str) -> Optional[ReferenceMeasurement]:
        """Retrieves a ReferenceMeasurement for the given canonical signal name."""
        return self.reference_signals.get(signal_name)

    def get_reference_value(self, signal_name: str) -> Optional[float]:
        """Retrieves the scalar reference value for the given canonical signal name."""
        meas = self.reference_signals.get(signal_name)
        return meas.value if meas is not None else None

    def get_field_value(self, signal_name: str) -> Optional[float]:
        """Retrieves the scalar field observation for the given signal name."""
        return self.field_signals.get(signal_name)

    def get_bms_estimate(self, signal_name: str) -> Optional[float]:
        """Retrieves the scalar unvalidated BMS firmware estimate for the given signal name."""
        return self.bms_estimates.get(signal_name)

    def to_dict(self) -> dict[str, Any]:
        """Serializes observation frame to a deterministic dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "observed_at_ns": self.observed_at_ns,
            "sequence_number": self.sequence_number,
            "reference_signals": {k: v.to_dict() for k, v in self.reference_signals.items()},
            "field_signals": dict(self.field_signals),
            "bms_estimates": dict(self.bms_estimates),
            "operating_context_hint": self.operating_context_hint,
            "quality_status": self.quality_status.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata specification identifying provenance, facility, and versioning for a dataset."""

    dataset_id: str
    dataset_name: str
    version: str
    created_at_utc: str
    facility_or_provider: str = ""
    operator_name: str = ""
    test_bench_identifier: str = ""
    schema_version: str = "1.0.0"
    description: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id or not isinstance(self.dataset_id, str):
            raise ValueError("dataset_id must be a non-empty string.")
        if not self.dataset_name or not isinstance(self.dataset_name, str):
            raise ValueError("dataset_name must be a non-empty string.")
        if not self.version or not isinstance(self.version, str):
            raise ValueError("version must be a non-empty string.")
        if not self.created_at_utc or not isinstance(self.created_at_utc, str):
            raise ValueError("created_at_utc must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes dataset metadata to a dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "created_at_utc": self.created_at_utc,
            "facility_or_provider": self.facility_or_provider,
            "operator_name": self.operator_name,
            "test_bench_identifier": self.test_bench_identifier,
            "schema_version": self.schema_version,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExperimentalDataset:
    """Immutable, self-describing reference dataset container.

    Encapsulates time-series reference observations, optional battery hardware configuration,
    channel specifications, and ground-truth experimental event markers.
    """

    metadata: DatasetMetadata
    battery_identification: Optional[BatteryIdentification] = None
    pack_configuration: Optional[PackConfiguration] = None
    cell_configuration: Optional[CellConfiguration] = None
    reference_channels: Tuple[ReferenceSignalChannel, ...] = field(default_factory=tuple)
    frames: Tuple[ReferenceObservationFrame, ...] = field(default_factory=tuple)
    ground_truth_events: Tuple[ExperimentalGroundTruthEvent, ...] = field(default_factory=tuple)
    require_monotonic_timestamps: bool = True
    allow_duplicate_timestamps: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, DatasetMetadata):
            raise TypeError(f"Expected DatasetMetadata, got {type(self.metadata).__name__}.")

        # Validate channel items
        for ch in self.reference_channels:
            if not isinstance(ch, ReferenceSignalChannel):
                raise TypeError(f"Expected ReferenceSignalChannel, got {type(ch).__name__}.")

        # Validate ground truth events
        for ev in self.ground_truth_events:
            if not isinstance(ev, ExperimentalGroundTruthEvent):
                raise TypeError(f"Expected ExperimentalGroundTruthEvent, got {type(ev).__name__}.")

        # Validate frame sequence and timestamps
        if self.frames:
            prev_ts = -1
            for idx, f in enumerate(self.frames):
                if not isinstance(f, ReferenceObservationFrame):
                    raise TypeError(f"Expected ReferenceObservationFrame at index {idx}, got {type(f).__name__}.")

                curr_ts = f.timestamp_ns
                if idx > 0 and self.require_monotonic_timestamps:
                    if curr_ts < prev_ts:
                        raise ValueError(
                            f"Non-monotonic timestamp detected at frame {idx}: "
                            f"{curr_ts} < {prev_ts}."
                        )
                    if curr_ts == prev_ts and not self.allow_duplicate_timestamps:
                        raise ValueError(
                            f"Duplicate timestamp detected at frame {idx}: "
                            f"timestamp_ns={curr_ts}."
                        )
                prev_ts = curr_ts

    @property
    def dataset_id(self) -> str:
        """Unique dataset identifier."""
        return self.metadata.dataset_id

    @property
    def total_frames(self) -> int:
        """Total count of observation frames in the dataset."""
        return len(self.frames)

    @property
    def start_timestamp_ns(self) -> Optional[int]:
        """First frame timestamp in nanoseconds, or None if empty."""
        return self.frames[0].timestamp_ns if self.frames else None

    @property
    def end_timestamp_ns(self) -> Optional[int]:
        """Last frame timestamp in nanoseconds, or None if empty."""
        return self.frames[-1].timestamp_ns if self.frames else None

    @property
    def duration_seconds(self) -> float:
        """Total duration spanned by the dataset frames in fractional seconds."""
        if len(self.frames) < 2:
            return 0.0
        return (self.frames[-1].timestamp_ns - self.frames[0].timestamp_ns) / 1_000_000_000.0

    @property
    def available_signals(self) -> Tuple[str, ...]:
        """Tuple of distinct canonical signal names present in reference channels or frames."""
        sig_set: set[str] = {ch.signal_name for ch in self.reference_channels}
        for f in self.frames:
            sig_set.update(f.reference_signals.keys())
            sig_set.update(f.field_signals.keys())
            sig_set.update(f.bms_estimates.keys())
        return tuple(sorted(sig_set))

    @property
    def ground_truth_event_count(self) -> int:
        """Number of documented experimental ground truth events."""
        return len(self.ground_truth_events)

    def get_frame(self, index: int) -> ReferenceObservationFrame:
        """Retrieves an observation frame by zero-based index."""
        if not (0 <= index < len(self.frames)):
            raise IndexError(f"Frame index {index} out of range [0, {len(self.frames) - 1}].")
        return self.frames[index]

    def query_time_range(
        self,
        start_time_ns: Optional[int] = None,
        end_time_ns: Optional[int] = None,
    ) -> "ExperimentalDataset":
        """Returns a new immutable ExperimentalDataset sliced to the specified time window."""
        filtered_frames: list[ReferenceObservationFrame] = []
        for f in self.frames:
            if start_time_ns is not None and f.timestamp_ns < start_time_ns:
                continue
            if end_time_ns is not None and f.timestamp_ns > end_time_ns:
                continue
            filtered_frames.append(f)

        # Filter events that overlap with window
        filtered_events: list[ExperimentalGroundTruthEvent] = []
        for ev in self.ground_truth_events:
            ev_start = ev.start_timestamp_ns
            ev_end = ev.end_timestamp_ns or ev_start
            if start_time_ns is not None and ev_end < start_time_ns:
                continue
            if end_time_ns is not None and ev_start > end_time_ns:
                continue
            filtered_events.append(ev)

        return ExperimentalDataset(
            metadata=self.metadata,
            battery_identification=self.battery_identification,
            pack_configuration=self.pack_configuration,
            cell_configuration=self.cell_configuration,
            reference_channels=self.reference_channels,
            frames=tuple(filtered_frames),
            ground_truth_events=tuple(filtered_events),
            require_monotonic_timestamps=self.require_monotonic_timestamps,
            allow_duplicate_timestamps=self.allow_duplicate_timestamps,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes dataset to a deterministic dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "battery_identification": (
                self.battery_identification.to_dict()
                if self.battery_identification and hasattr(self.battery_identification, "to_dict")
                else None
            ),
            "pack_configuration": (
                self.pack_configuration.to_dict()
                if self.pack_configuration and hasattr(self.pack_configuration, "to_dict")
                else None
            ),
            "cell_configuration": (
                self.cell_configuration.to_dict()
                if self.cell_configuration and hasattr(self.cell_configuration, "to_dict")
                else None
            ),
            "reference_channels": [ch.to_dict() for ch in self.reference_channels],
            "total_frames": self.total_frames,
            "duration_seconds": self.duration_seconds,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "ground_truth_events": [ev.to_dict() for ev in self.ground_truth_events],
            "available_signals": list(self.available_signals),
        }


class DatasetBuilder:
    """Fluent, memory-safe builder for constructing immutable ExperimentalDataset instances."""

    def __init__(self, metadata: DatasetMetadata) -> None:
        if not isinstance(metadata, DatasetMetadata):
            raise TypeError(f"Expected DatasetMetadata, got {type(metadata).__name__}.")
        self._metadata = metadata
        self._battery_identification: Optional[BatteryIdentification] = None
        self._pack_configuration: Optional[PackConfiguration] = None
        self._cell_configuration: Optional[CellConfiguration] = None
        self._reference_channels: list[ReferenceSignalChannel] = []
        self._frames: list[ReferenceObservationFrame] = []
        self._ground_truth_events: list[ExperimentalGroundTruthEvent] = []
        self._require_monotonic_timestamps: bool = True
        self._allow_duplicate_timestamps: bool = False

    def set_battery_configuration(
        self,
        identification: Optional[BatteryIdentification] = None,
        pack_config: Optional[PackConfiguration] = None,
        cell_config: Optional[CellConfiguration] = None,
    ) -> "DatasetBuilder":
        """Attaches universal battery domain configurations to the dataset."""
        self._battery_identification = identification
        self._pack_configuration = pack_config
        self._cell_configuration = cell_config
        return self

    def add_reference_channel(self, channel: ReferenceSignalChannel) -> "DatasetBuilder":
        """Adds a reference channel specification."""
        if not isinstance(channel, ReferenceSignalChannel):
            raise TypeError(f"Expected ReferenceSignalChannel, got {type(channel).__name__}.")
        self._reference_channels.append(channel)
        return self

    def add_frame(self, frame: ReferenceObservationFrame) -> "DatasetBuilder":
        """Appends a single observation frame."""
        if not isinstance(frame, ReferenceObservationFrame):
            raise TypeError(f"Expected ReferenceObservationFrame, got {type(frame).__name__}.")
        self._frames.append(frame)
        return self

    def add_frames(self, frames: Sequence[ReferenceObservationFrame]) -> "DatasetBuilder":
        """Appends a sequence of observation frames."""
        for f in frames:
            self.add_frame(f)
        return self

    def add_ground_truth_event(self, event: ExperimentalGroundTruthEvent) -> "DatasetBuilder":
        """Adds an experimental ground-truth event."""
        if not isinstance(event, ExperimentalGroundTruthEvent):
            raise TypeError(f"Expected ExperimentalGroundTruthEvent, got {type(event).__name__}.")
        self._ground_truth_events.append(event)
        return self

    def set_timestamp_policy(
        self,
        require_monotonic: bool = True,
        allow_duplicates: bool = False,
    ) -> "DatasetBuilder":
        """Configures timestamp ordering and duplicate tolerance for dataset validation."""
        self._require_monotonic_timestamps = bool(require_monotonic)
        self._allow_duplicate_timestamps = bool(allow_duplicates)
        return self

    def build(self) -> ExperimentalDataset:
        """Constructs and validates the final immutable ExperimentalDataset."""
        return ExperimentalDataset(
            metadata=self._metadata,
            battery_identification=self._battery_identification,
            pack_configuration=self._pack_configuration,
            cell_configuration=self._cell_configuration,
            reference_channels=tuple(self._reference_channels),
            frames=tuple(self._frames),
            ground_truth_events=tuple(self._ground_truth_events),
            require_monotonic_timestamps=self._require_monotonic_timestamps,
            allow_duplicate_timestamps=self._allow_duplicate_timestamps,
        )
