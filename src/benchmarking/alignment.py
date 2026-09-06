"""Deterministic Temporal Alignment Engine for Battery Benchmarking.

Provides causal streaming and non-causal batch temporal alignment policies, time-series pairing,
tolerance window enforcement, gap protection, and non-artificial interpolation guardrails.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from src.benchmarking.dataset import (
    ExperimentalDataset,
    ReferenceObservationFrame,
)
from src.benchmarking.types import (
    ReferenceMeasurement,
    ReferenceProvenance,
)


class AlignmentMode(str, Enum):
    """Execution mode of temporal alignment."""

    CAUSAL_STREAMING = "CAUSAL_STREAMING"   # Online/real-time: strictly zero look-ahead
    NON_CAUSAL_BATCH = "NON_CAUSAL_BATCH"   # Offline historical dataset evaluation


class InterpolationPolicy(str, Enum):
    """Interpolation method applied between valid reference samples."""

    NONE = "NONE"                           # No interpolation; exact or bounded nearest-neighbor matching only
    SAMPLE_AND_HOLD = "SAMPLE_AND_HOLD"     # Zero-order hold from nearest past sample
    LINEAR_PIECEWISE = "LINEAR_PIECEWISE"   # Linear interpolation between strictly adjacent valid samples


class ResamplingPolicy(str, Enum):
    """Resampling and pairing search policy."""

    EXACT_OR_DROP = "EXACT_OR_DROP"                 # Pair only on exact timestamp match
    CAUSAL_NEAREST_PAST = "CAUSAL_NEAREST_PAST"     # Match nearest past sample within tolerance
    BIDIRECTIONAL_NEAREST = "BIDIRECTIONAL_NEAREST" # Match nearest sample (past or future) within tolerance


class AlignmentStatus(str, Enum):
    """Detailed pairing status for an individual signal observation."""

    EXACT_MATCH = "EXACT_MATCH"
    NEAREST_PAST_MATCH = "NEAREST_PAST_MATCH"
    NEAREST_FUTURE_MATCH = "NEAREST_FUTURE_MATCH"
    SAMPLE_AND_HOLD = "SAMPLE_AND_HOLD"
    INTERPOLATED_LINEAR = "INTERPOLATED_LINEAR"
    OUTSIDE_TOLERANCE = "OUTSIDE_TOLERANCE"
    DATA_GAPPED = "DATA_GAPPED"
    MISSING_REFERENCE = "MISSING_REFERENCE"
    RETROGRADE_REJECTED = "RETROGRADE_REJECTED"
    DUPLICATE_REJECTED = "DUPLICATE_REJECTED"


class ObservationProvenanceType(str, Enum):
    """Evidentiary derivation type of the reference value used in a paired evaluation."""

    DIRECT_OBSERVATION = "DIRECT_OBSERVATION" # Directly measured reference point
    HELD_VALUE = "HELD_VALUE"                 # Zero-order held value from past observation
    INTERPOLATED_VALUE = "INTERPOLATED_VALUE" # Mathematically interpolated value
    UNAVAILABLE = "UNAVAILABLE"               # No reference available


@dataclass(frozen=True)
class TemporalAlignmentPolicy:
    """Configurable rules governing temporal pairing between evaluation signals and reference data."""

    timestamp_tolerance_ns: int = 50_000_000        # Pairing tolerance window in nanoseconds (default 50 ms)
    max_allowable_gap_s: float = 2.0                # Max allowable gap in seconds before declaring DATA_GAPPED
    interpolation_policy: InterpolationPolicy = InterpolationPolicy.NONE
    resampling_policy: ResamplingPolicy = ResamplingPolicy.CAUSAL_NEAREST_PAST
    alignment_mode: AlignmentMode = AlignmentMode.CAUSAL_STREAMING
    reject_non_monotonic: bool = True               # Enforces strictly increasing or monotonic timestamps
    allow_duplicate_timestamps: bool = False        # Whether identical timestamps are allowed in evaluation series
    use_observed_at_timestamps: bool = False        # Use observed_at_ns if present on reference frames

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_tolerance_ns, int) or self.timestamp_tolerance_ns < 0:
            raise ValueError(
                f"timestamp_tolerance_ns must be a non-negative integer, got {self.timestamp_tolerance_ns}."
            )
        if not isinstance(self.max_allowable_gap_s, (int, float)) or math.isnan(self.max_allowable_gap_s) or math.isinf(self.max_allowable_gap_s) or self.max_allowable_gap_s <= 0.0:
            raise ValueError(
                f"max_allowable_gap_s must be a strictly positive finite float, got {self.max_allowable_gap_s}."
            )
        if not isinstance(self.interpolation_policy, InterpolationPolicy):
            raise TypeError(f"Expected InterpolationPolicy, got {type(self.interpolation_policy).__name__}.")
        if not isinstance(self.resampling_policy, ResamplingPolicy):
            raise TypeError(f"Expected ResamplingPolicy, got {type(self.resampling_policy).__name__}.")
        if not isinstance(self.alignment_mode, AlignmentMode):
            raise TypeError(f"Expected AlignmentMode, got {type(self.alignment_mode).__name__}.")

    @property
    def max_allowable_gap_ns(self) -> int:
        """Converts max_allowable_gap_s to integer nanoseconds."""
        return int(self.max_allowable_gap_s * 1_000_000_000)

    def to_dict(self) -> dict[str, Any]:
        """Serializes alignment policy to a dictionary."""
        return {
            "timestamp_tolerance_ns": self.timestamp_tolerance_ns,
            "max_allowable_gap_s": self.max_allowable_gap_s,
            "interpolation_policy": self.interpolation_policy.value,
            "resampling_policy": self.resampling_policy.value,
            "alignment_mode": self.alignment_mode.value,
            "reject_non_monotonic": self.reject_non_monotonic,
            "allow_duplicate_timestamps": self.allow_duplicate_timestamps,
            "use_observed_at_timestamps": self.use_observed_at_timestamps,
        }


@dataclass(frozen=True)
class AlignedSignalPair:
    """A paired evaluation and reference observation for a single signal at a specific evaluation timestamp."""

    signal_name: str
    eval_timestamp_ns: int
    eval_value: float
    ref_timestamp_ns: Optional[int] = None
    ref_value: Optional[float] = None
    ref_uncertainty: float = 0.0
    ref_uncertainty_absolute: Optional[float] = None
    ref_uncertainty_relative_fraction: Optional[float] = None
    status: AlignmentStatus = AlignmentStatus.MISSING_REFERENCE
    observation_type: ObservationProvenanceType = ObservationProvenanceType.UNAVAILABLE
    provenance: Optional[ReferenceProvenance] = None
    time_offset_ns: Optional[int] = None
    is_valid_pair: bool = False
    exclusion_reason: Optional[str] = None
    error: Optional[float] = None
    is_uncertainty_consistent: Optional[bool] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes aligned signal pair to a dictionary."""
        return {
            "signal_name": self.signal_name,
            "eval_timestamp_ns": self.eval_timestamp_ns,
            "eval_value": self.eval_value,
            "ref_timestamp_ns": self.ref_timestamp_ns,
            "ref_value": self.ref_value,
            "ref_uncertainty": self.ref_uncertainty,
            "ref_uncertainty_absolute": self.ref_uncertainty_absolute,
            "ref_uncertainty_relative_fraction": self.ref_uncertainty_relative_fraction,
            "status": self.status.value,
            "observation_type": self.observation_type.value,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "time_offset_ns": self.time_offset_ns,
            "is_valid_pair": self.is_valid_pair,
            "exclusion_reason": self.exclusion_reason,
            "error": self.error,
            "is_uncertainty_consistent": self.is_uncertainty_consistent,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AlignedFramePair:
    """A complete paired frame mapping all evaluated signals against reference signals at an evaluation timestamp."""

    eval_timestamp_ns: int
    eval_signals: Mapping[str, float]
    ref_frame: Optional[ReferenceObservationFrame] = None
    aligned_signals: Mapping[str, AlignedSignalPair] = field(default_factory=dict)
    status: AlignmentStatus = AlignmentStatus.MISSING_REFERENCE
    metadata: Mapping[str, str] = field(default_factory=dict)

    def get_signal_pair(self, signal_name: str) -> Optional[AlignedSignalPair]:
        """Retrieves the aligned pair for the specified signal name."""
        return self.aligned_signals.get(signal_name)

    def to_dict(self) -> dict[str, Any]:
        """Serializes aligned frame pair to a dictionary."""
        return {
            "eval_timestamp_ns": self.eval_timestamp_ns,
            "eval_signals": dict(self.eval_signals),
            "ref_frame_timestamp_ns": self.ref_frame.timestamp_ns if self.ref_frame else None,
            "aligned_signals": {k: v.to_dict() for k, v in self.aligned_signals.items()},
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


class TemporalAlignmentEngine:
    """Deterministic Temporal Alignment Engine.

    Executes causal streaming or non-causal batch pairing of time-series evaluation signals
    against immutable reference frames according to a strictly parameterized TemporalAlignmentPolicy.
    """

    @staticmethod
    def _extract_ref_ts(frame: ReferenceObservationFrame, policy: TemporalAlignmentPolicy) -> int:
        if policy.use_observed_at_timestamps and frame.observed_at_ns is not None:
            return frame.observed_at_ns
        return frame.timestamp_ns

    @classmethod
    def align_sample(
        cls,
        eval_timestamp_ns: int,
        eval_value: float,
        signal_name: str,
        reference_frames: Sequence[ReferenceObservationFrame],
        policy: Optional[TemporalAlignmentPolicy] = None,
    ) -> AlignedSignalPair:
        """Aligns a single evaluated scalar observation against an ordered sequence of reference frames.

        Args:
            eval_timestamp_ns: Evaluation timestamp in nanoseconds.
            eval_value: Digital twin evaluated or simulated value.
            signal_name: Canonical name of the evaluated signal.
            reference_frames: Ordered sequence of reference frames.
            policy: Configurable alignment policy (defaults to causal 50ms tolerance).

        Returns:
            AlignedSignalPair value object.
        """
        cfg = policy or TemporalAlignmentPolicy()

        if not isinstance(eval_timestamp_ns, int) or eval_timestamp_ns < 0:
            raise ValueError(f"eval_timestamp_ns must be a non-negative integer, got {eval_timestamp_ns}.")
        if not isinstance(eval_value, (int, float)) or math.isnan(eval_value) or math.isinf(eval_value):
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                status=AlignmentStatus.DATA_QUALITY_FAILED,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                is_valid_pair=False,
                exclusion_reason=f"Non-finite evaluation value: {eval_value}",
            )

        if not reference_frames:
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                status=AlignmentStatus.MISSING_REFERENCE,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                is_valid_pair=False,
                exclusion_reason="Empty reference dataset.",
            )

        # 1. Filter frames carrying the target signal
        valid_frames: list[Tuple[int, ReferenceObservationFrame, ReferenceMeasurement]] = []
        for f in reference_frames:
            meas = f.get_reference(signal_name)
            if meas is not None:
                ts = cls._extract_ref_ts(f, cfg)
                valid_frames.append((ts, f, meas))

        if not valid_frames:
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                status=AlignmentStatus.MISSING_REFERENCE,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                is_valid_pair=False,
                exclusion_reason=f"Signal '{signal_name}' not present in reference dataset.",
            )

        # 2. Alignment Mode Handling
        if cfg.alignment_mode == AlignmentMode.CAUSAL_STREAMING:
            return cls._align_causal(
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                signal_name=signal_name,
                valid_frames=valid_frames,
                policy=cfg,
            )
        else:
            return cls._align_non_causal_batch(
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                signal_name=signal_name,
                valid_frames=valid_frames,
                policy=cfg,
            )

    @classmethod
    def _align_causal(
        cls,
        eval_timestamp_ns: int,
        eval_value: float,
        signal_name: str,
        valid_frames: Sequence[Tuple[int, ReferenceObservationFrame, ReferenceMeasurement]],
        policy: TemporalAlignmentPolicy,
    ) -> AlignedSignalPair:
        """Causal streaming alignment: strictly looks only at reference frames with t_ref <= t_eval."""
        # Find candidates with ts <= eval_timestamp_ns (strictly causal)
        past_candidates = [item for item in valid_frames if item[0] <= eval_timestamp_ns]

        if not past_candidates:
            # All available reference frames are in the future; causal mode strictly rejects future reference
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                status=AlignmentStatus.OUTSIDE_TOLERANCE,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                is_valid_pair=False,
                exclusion_reason="No past reference observations available at or before evaluation timestamp.",
            )

        # Nearest past sample is the last element in past_candidates
        nearest_ts, nearest_frame, nearest_meas = past_candidates[-1]
        time_offset_ns = nearest_ts - eval_timestamp_ns  # Non-positive in causal mode (e.g. 0 or negative)
        dt_ns = abs(time_offset_ns)

        # Check gap
        if dt_ns > policy.max_allowable_gap_ns:
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                ref_timestamp_ns=nearest_ts,
                ref_value=nearest_meas.value,
                status=AlignmentStatus.DATA_GAPPED,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                time_offset_ns=time_offset_ns,
                is_valid_pair=False,
                exclusion_reason=f"Time delta {dt_ns}ns exceeds max allowable gap {policy.max_allowable_gap_ns}ns.",
            )

        # Exact match
        if dt_ns == 0:
            return cls._build_valid_pair(
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                signal_name=signal_name,
                ref_timestamp_ns=nearest_ts,
                ref_value=nearest_meas.value,
                meas=nearest_meas,
                status=AlignmentStatus.EXACT_MATCH,
                obs_type=ObservationProvenanceType.DIRECT_OBSERVATION,
                time_offset_ns=0,
            )

        # Resampling policy check
        if policy.resampling_policy == ResamplingPolicy.EXACT_OR_DROP:
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                ref_timestamp_ns=nearest_ts,
                status=AlignmentStatus.OUTSIDE_TOLERANCE,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                time_offset_ns=time_offset_ns,
                is_valid_pair=False,
                exclusion_reason="EXACT_OR_DROP policy rejected non-zero timestamp delta.",
            )

        # Check tolerance window
        if dt_ns > policy.timestamp_tolerance_ns:
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                ref_timestamp_ns=nearest_ts,
                ref_value=nearest_meas.value,
                status=AlignmentStatus.OUTSIDE_TOLERANCE,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                time_offset_ns=time_offset_ns,
                is_valid_pair=False,
                exclusion_reason=f"Time delta {dt_ns}ns exceeds tolerance {policy.timestamp_tolerance_ns}ns.",
            )

        # Valid within tolerance in causal mode
        if policy.interpolation_policy == InterpolationPolicy.SAMPLE_AND_HOLD:
            return cls._build_valid_pair(
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                signal_name=signal_name,
                ref_timestamp_ns=nearest_ts,
                ref_value=nearest_meas.value,
                meas=nearest_meas,
                status=AlignmentStatus.SAMPLE_AND_HOLD,
                obs_type=ObservationProvenanceType.HELD_VALUE,
                time_offset_ns=time_offset_ns,
            )
        else:
            return cls._build_valid_pair(
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                signal_name=signal_name,
                ref_timestamp_ns=nearest_ts,
                ref_value=nearest_meas.value,
                meas=nearest_meas,
                status=AlignmentStatus.NEAREST_PAST_MATCH,
                obs_type=ObservationProvenanceType.DIRECT_OBSERVATION,
                time_offset_ns=time_offset_ns,
            )

    @classmethod
    def _align_non_causal_batch(
        cls,
        eval_timestamp_ns: int,
        eval_value: float,
        signal_name: str,
        valid_frames: Sequence[Tuple[int, ReferenceObservationFrame, ReferenceMeasurement]],
        policy: TemporalAlignmentPolicy,
    ) -> AlignedSignalPair:
        """Non-causal batch alignment: allows bounded bidirectional search and piecewise linear interpolation."""
        # Check exact match
        for ts, frame, meas in valid_frames:
            if ts == eval_timestamp_ns:
                return cls._build_valid_pair(
                    eval_timestamp_ns=eval_timestamp_ns,
                    eval_value=eval_value,
                    signal_name=signal_name,
                    ref_timestamp_ns=ts,
                    ref_value=meas.value,
                    meas=meas,
                    status=AlignmentStatus.EXACT_MATCH,
                    obs_type=ObservationProvenanceType.DIRECT_OBSERVATION,
                    time_offset_ns=0,
                )

        # Find bracketing past and future samples
        past_items = [item for item in valid_frames if item[0] < eval_timestamp_ns]
        future_items = [item for item in valid_frames if item[0] > eval_timestamp_ns]

        past_item = past_items[-1] if past_items else None
        future_item = future_items[0] if future_items else None

        # 1. Piecewise Linear Interpolation (if both bracket samples exist)
        if policy.interpolation_policy == InterpolationPolicy.LINEAR_PIECEWISE and past_item and future_item:
            t0, f0, m0 = past_item
            t1, f1, m1 = future_item
            bracket_gap_ns = t1 - t0

            # Guard against forbidden gaps
            if bracket_gap_ns > policy.max_allowable_gap_ns:
                return AlignedSignalPair(
                    signal_name=signal_name,
                    eval_timestamp_ns=eval_timestamp_ns,
                    eval_value=eval_value,
                    status=AlignmentStatus.DATA_GAPPED,
                    observation_type=ObservationProvenanceType.UNAVAILABLE,
                    is_valid_pair=False,
                    exclusion_reason=f"Bracketing gap {bracket_gap_ns}ns exceeds max allowable gap {policy.max_allowable_gap_ns}ns; interpolation prohibited.",
                )

            # Check individual sample distances against tolerance
            dist_to_nearest = min(eval_timestamp_ns - t0, t1 - eval_timestamp_ns)
            if dist_to_nearest > policy.timestamp_tolerance_ns:
                return AlignedSignalPair(
                    signal_name=signal_name,
                    eval_timestamp_ns=eval_timestamp_ns,
                    eval_value=eval_value,
                    status=AlignmentStatus.OUTSIDE_TOLERANCE,
                    observation_type=ObservationProvenanceType.UNAVAILABLE,
                    is_valid_pair=False,
                    exclusion_reason=f"Distance to nearest reference point {dist_to_nearest}ns exceeds tolerance {policy.timestamp_tolerance_ns}ns.",
                )

            # Linear interpolation formula: y = y0 + (t - t0)/(t1 - t0) * (y1 - y0)
            fraction = (eval_timestamp_ns - t0) / float(bracket_gap_ns)
            interp_value = m0.value + fraction * (m1.value - m0.value)
            # Propagate conservative uncertainty (max of endpoints)
            interp_unc = max(m0.effective_uncertainty(), m1.effective_uncertainty())

            return cls._build_valid_pair(
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                signal_name=signal_name,
                ref_timestamp_ns=eval_timestamp_ns,
                ref_value=interp_value,
                meas=m0,  # Uses m0 provenance
                status=AlignmentStatus.INTERPOLATED_LINEAR,
                obs_type=ObservationProvenanceType.INTERPOLATED_VALUE,
                time_offset_ns=0,
                override_uncertainty=interp_unc,
            )

        # 2. Nearest Neighbor (Bidirectional or Past)
        candidates = []
        if past_item:
            candidates.append(past_item)
        if future_item and policy.resampling_policy == ResamplingPolicy.BIDIRECTIONAL_NEAREST:
            candidates.append(future_item)

        if not candidates:
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                status=AlignmentStatus.OUTSIDE_TOLERANCE,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                is_valid_pair=False,
                exclusion_reason="No reference observations within search window.",
            )

        # Pick closest candidate
        best_ts, best_frame, best_meas = min(candidates, key=lambda c: abs(c[0] - eval_timestamp_ns))
        time_offset_ns = best_ts - eval_timestamp_ns
        dt_ns = abs(time_offset_ns)

        if dt_ns > policy.max_allowable_gap_ns:
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                ref_timestamp_ns=best_ts,
                ref_value=best_meas.value,
                status=AlignmentStatus.DATA_GAPPED,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                time_offset_ns=time_offset_ns,
                is_valid_pair=False,
                exclusion_reason=f"Time delta {dt_ns}ns exceeds max allowable gap {policy.max_allowable_gap_ns}ns.",
            )

        if dt_ns > policy.timestamp_tolerance_ns:
            return AlignedSignalPair(
                signal_name=signal_name,
                eval_timestamp_ns=eval_timestamp_ns,
                eval_value=eval_value,
                ref_timestamp_ns=best_ts,
                ref_value=best_meas.value,
                status=AlignmentStatus.OUTSIDE_TOLERANCE,
                observation_type=ObservationProvenanceType.UNAVAILABLE,
                time_offset_ns=time_offset_ns,
                is_valid_pair=False,
                exclusion_reason=f"Time delta {dt_ns}ns exceeds tolerance {policy.timestamp_tolerance_ns}ns.",
            )

        status = AlignmentStatus.NEAREST_PAST_MATCH if best_ts <= eval_timestamp_ns else AlignmentStatus.NEAREST_FUTURE_MATCH
        obs_type = ObservationProvenanceType.HELD_VALUE if policy.interpolation_policy == InterpolationPolicy.SAMPLE_AND_HOLD else ObservationProvenanceType.DIRECT_OBSERVATION

        return cls._build_valid_pair(
            eval_timestamp_ns=eval_timestamp_ns,
            eval_value=eval_value,
            signal_name=signal_name,
            ref_timestamp_ns=best_ts,
            ref_value=best_meas.value,
            meas=best_meas,
            status=status,
            obs_type=obs_type,
            time_offset_ns=time_offset_ns,
        )

    @classmethod
    def _build_valid_pair(
        cls,
        eval_timestamp_ns: int,
        eval_value: float,
        signal_name: str,
        ref_timestamp_ns: int,
        ref_value: float,
        meas: ReferenceMeasurement,
        status: AlignmentStatus,
        obs_type: ObservationProvenanceType,
        time_offset_ns: int,
        override_uncertainty: Optional[float] = None,
    ) -> AlignedSignalPair:
        """Helper constructing an aligned pair value object with error and uncertainty metrics."""
        err = float(ref_value - eval_value)
        eff_unc = override_uncertainty if override_uncertainty is not None else meas.effective_uncertainty()
        is_consistent = abs(err) <= eff_unc if eff_unc > 0.0 else (abs(err) < 1e-12)

        return AlignedSignalPair(
            signal_name=signal_name,
            eval_timestamp_ns=eval_timestamp_ns,
            eval_value=eval_value,
            ref_timestamp_ns=ref_timestamp_ns,
            ref_value=ref_value,
            ref_uncertainty=eff_unc,
            ref_uncertainty_absolute=meas.uncertainty_absolute,
            ref_uncertainty_relative_fraction=meas.uncertainty_relative_fraction,
            status=status,
            observation_type=obs_type,
            provenance=meas.provenance,
            time_offset_ns=time_offset_ns,
            is_valid_pair=True,
            exclusion_reason=None,
            error=err,
            is_uncertainty_consistent=is_consistent,
            metadata=dict(meas.metadata),
        )

    @classmethod
    def align_time_series(
        cls,
        eval_series: Sequence[Tuple[int, float]],
        signal_name: str,
        reference_frames: Sequence[ReferenceObservationFrame],
        policy: Optional[TemporalAlignmentPolicy] = None,
    ) -> Tuple[AlignedSignalPair, ...]:
        """Aligns an ordered time series of (timestamp_ns, eval_value) pairs against reference frames."""
        cfg = policy or TemporalAlignmentPolicy()
        aligned_pairs: list[AlignedSignalPair] = []

        prev_ts = -1
        for idx, (ts, val) in enumerate(eval_series):
            # Enforce evaluation monotonicity
            if idx > 0 and cfg.reject_non_monotonic and ts < prev_ts:
                aligned_pairs.append(
                    AlignedSignalPair(
                        signal_name=signal_name,
                        eval_timestamp_ns=ts,
                        eval_value=val,
                        status=AlignmentStatus.RETROGRADE_REJECTED,
                        observation_type=ObservationProvenanceType.UNAVAILABLE,
                        is_valid_pair=False,
                        exclusion_reason=f"Retrograde timestamp detected at index {idx}: {ts} < {prev_ts}.",
                    )
                )
                continue

            if idx > 0 and not cfg.allow_duplicate_timestamps and ts == prev_ts:
                aligned_pairs.append(
                    AlignedSignalPair(
                        signal_name=signal_name,
                        eval_timestamp_ns=ts,
                        eval_value=val,
                        status=AlignmentStatus.DUPLICATE_REJECTED,
                        observation_type=ObservationProvenanceType.UNAVAILABLE,
                        is_valid_pair=False,
                        exclusion_reason=f"Duplicate timestamp detected at index {idx}: timestamp_ns={ts}.",
                    )
                )
                continue

            prev_ts = ts
            pair = cls.align_sample(
                eval_timestamp_ns=ts,
                eval_value=val,
                signal_name=signal_name,
                reference_frames=reference_frames,
                policy=cfg,
            )
            aligned_pairs.append(pair)

        return tuple(aligned_pairs)

    @classmethod
    def align_frames(
        cls,
        eval_frames: Sequence[Tuple[int, Mapping[str, float]]],
        reference_dataset: ExperimentalDataset,
        policy: Optional[TemporalAlignmentPolicy] = None,
    ) -> Tuple[AlignedFramePair, ...]:
        """Aligns multi-signal evaluation frames against an ExperimentalDataset."""
        cfg = policy or TemporalAlignmentPolicy()
        frame_pairs: list[AlignedFramePair] = []

        ref_frames = reference_dataset.frames

        for ts, eval_sigs in eval_frames:
            aligned_map: dict[str, AlignedSignalPair] = {}
            for sig_name, eval_val in eval_sigs.items():
                pair = cls.align_sample(
                    eval_timestamp_ns=ts,
                    eval_value=eval_val,
                    signal_name=sig_name,
                    reference_frames=ref_frames,
                    policy=cfg,
                )
                aligned_map[sig_name] = pair

            # Determine frame overall status
            if all(p.is_valid_pair for p in aligned_map.values()) and aligned_map:
                frame_status = AlignmentStatus.EXACT_MATCH if all(p.status == AlignmentStatus.EXACT_MATCH for p in aligned_map.values()) else AlignmentStatus.NEAREST_PAST_MATCH
            elif any(p.is_valid_pair for p in aligned_map.values()):
                frame_status = AlignmentStatus.NEAREST_PAST_MATCH
            else:
                frame_status = AlignmentStatus.MISSING_REFERENCE

            frame_pairs.append(
                AlignedFramePair(
                    eval_timestamp_ns=ts,
                    eval_signals=eval_sigs,
                    aligned_signals=aligned_map,
                    status=frame_status,
                )
            )

        return tuple(frame_pairs)
