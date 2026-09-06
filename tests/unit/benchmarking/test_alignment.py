"""Unit tests for Level 5.5 Phase 2 Temporal Alignment Engine and Policies."""

import unittest

from src.benchmarking.alignment import (
    AlignedSignalPair,
    AlignmentMode,
    AlignmentStatus,
    InterpolationPolicy,
    ObservationProvenanceType,
    ResamplingPolicy,
    TemporalAlignmentEngine,
    TemporalAlignmentPolicy,
)
from src.benchmarking.dataset import (
    DatasetMetadata,
    ExperimentalDataset,
    ReferenceObservationFrame,
)
from src.benchmarking.types import (
    BenchmarkSignalType,
    CalibrationStatus,
    EvidenceRole,
    ReferenceMeasurement,
    ReferenceProvenance,
    ReferenceSourceType,
)


class TestBenchmarkingAlignment(unittest.TestCase):
    """Comprehensive test suite for temporal alignment policies and pairing algorithms."""

    def setUp(self) -> None:
        self.prov = ReferenceProvenance(
            provenance_id="p1",
            source_type=ReferenceSourceType.LABORATORY_METROLOGY,
            calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
            evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
            measurement_method="FOUR_WIRE_KELVIN",
        )
        # Reference frames at 1.0s, 2.0s, 3.0s, 4.0s, 5.0s, 10.0s (a 5s gap between 5.0s and 10.0s)
        self.ref_frames = [
            ReferenceObservationFrame(
                timestamp_ns=1_000_000_000,
                reference_signals={"voltage_v": ReferenceMeasurement(value=4.0, unit="V", timestamp_ns=1_000_000_000, uncertainty_absolute=0.01, provenance=self.prov)},
            ),
            ReferenceObservationFrame(
                timestamp_ns=2_000_000_000,
                reference_signals={"voltage_v": ReferenceMeasurement(value=3.9, unit="V", timestamp_ns=2_000_000_000, uncertainty_absolute=0.01, provenance=self.prov)},
            ),
            ReferenceObservationFrame(
                timestamp_ns=3_000_000_000,
                reference_signals={"voltage_v": ReferenceMeasurement(value=3.8, unit="V", timestamp_ns=3_000_000_000, uncertainty_absolute=0.01, provenance=self.prov)},
            ),
            ReferenceObservationFrame(
                timestamp_ns=4_000_000_000,
                reference_signals={"voltage_v": ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=4_000_000_000, uncertainty_absolute=0.01, provenance=self.prov)},
            ),
            ReferenceObservationFrame(
                timestamp_ns=5_000_000_000,
                reference_signals={"voltage_v": ReferenceMeasurement(value=3.6, unit="V", timestamp_ns=5_000_000_000, uncertainty_absolute=0.01, provenance=self.prov)},
            ),
            ReferenceObservationFrame(
                timestamp_ns=10_000_000_000,
                reference_signals={"voltage_v": ReferenceMeasurement(value=3.5, unit="V", timestamp_ns=10_000_000_000, uncertainty_absolute=0.01, provenance=self.prov)},
            ),
        ]

    # 1. Exact timestamp match
    def test_exact_timestamp_match(self) -> None:
        policy = TemporalAlignmentPolicy(timestamp_tolerance_ns=50_000_000)
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=2_000_000_000,
            eval_value=3.89,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertTrue(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.EXACT_MATCH)
        self.assertEqual(pair.observation_type, ObservationProvenanceType.DIRECT_OBSERVATION)
        self.assertEqual(pair.ref_value, 3.9)
        self.assertEqual(pair.time_offset_ns, 0)
        self.assertAlmostEqual(pair.error, 0.01, places=6)
        self.assertTrue(pair.is_uncertainty_consistent)

    # 2. Nearest valid match within tolerance
    def test_nearest_valid_match_within_tolerance(self) -> None:
        # Evaluation at 2.030s (30ms after 2.0s reference), tolerance is 50ms
        policy = TemporalAlignmentPolicy(timestamp_tolerance_ns=50_000_000)
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=2_030_000_000,
            eval_value=3.9,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertTrue(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.NEAREST_PAST_MATCH)
        self.assertEqual(pair.ref_value, 3.9)
        self.assertEqual(pair.time_offset_ns, -30_000_000)

    # 3. Tolerance boundary (exact boundary)
    def test_tolerance_boundary_exact(self) -> None:
        # Exactly 50ms after 2.0s
        policy = TemporalAlignmentPolicy(timestamp_tolerance_ns=50_000_000)
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=2_050_000_000,
            eval_value=3.9,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertTrue(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.NEAREST_PAST_MATCH)

    # 4. Outside tolerance window
    def test_outside_tolerance_window(self) -> None:
        # 60ms after 2.0s (tolerance is 50ms)
        policy = TemporalAlignmentPolicy(timestamp_tolerance_ns=50_000_000)
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=2_060_000_000,
            eval_value=3.9,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertFalse(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.OUTSIDE_TOLERANCE)
        self.assertIn("exceeds tolerance", pair.exclusion_reason)

    # 5. Causal alignment and future-sample rejection
    def test_causal_mode_future_sample_rejection(self) -> None:
        # Evaluation is at 0.950s (before the first reference at 1.0s)
        policy = TemporalAlignmentPolicy(
            alignment_mode=AlignmentMode.CAUSAL_STREAMING,
            timestamp_tolerance_ns=100_000_000,
        )
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=950_000_000,
            eval_value=4.0,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        # Even though 1.0s is only 50ms in the future, causal streaming MUST NOT use future sample
        self.assertFalse(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.OUTSIDE_TOLERANCE)
        self.assertIsNone(pair.ref_timestamp_ns)

    # 6. Non-causal batch alignment
    def test_non_causal_batch_mode_future_match(self) -> None:
        # In non-causal batch mode, 0.950s can pair with future 1.0s within 100ms tolerance
        policy = TemporalAlignmentPolicy(
            alignment_mode=AlignmentMode.NON_CAUSAL_BATCH,
            resampling_policy=ResamplingPolicy.BIDIRECTIONAL_NEAREST,
            timestamp_tolerance_ns=100_000_000,
        )
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=950_000_000,
            eval_value=4.0,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertTrue(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.NEAREST_FUTURE_MATCH)
        self.assertEqual(pair.ref_timestamp_ns, 1_000_000_000)
        self.assertEqual(pair.ref_value, 4.0)

    # 7. Sample-and-hold interpolation
    def test_sample_and_hold_interpolation(self) -> None:
        policy = TemporalAlignmentPolicy(
            alignment_mode=AlignmentMode.CAUSAL_STREAMING,
            interpolation_policy=InterpolationPolicy.SAMPLE_AND_HOLD,
            timestamp_tolerance_ns=500_000_000,
        )
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=2_300_000_000,
            eval_value=3.9,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertTrue(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.SAMPLE_AND_HOLD)
        self.assertEqual(pair.observation_type, ObservationProvenanceType.HELD_VALUE)
        self.assertEqual(pair.ref_value, 3.9)

    # 8. Piecewise linear interpolation (Non-causal batch)
    def test_linear_interpolation(self) -> None:
        # t=2.5s is midway between t=2.0s (3.9V) and t=3.0s (3.8V) -> 3.85V
        policy = TemporalAlignmentPolicy(
            alignment_mode=AlignmentMode.NON_CAUSAL_BATCH,
            interpolation_policy=InterpolationPolicy.LINEAR_PIECEWISE,
            timestamp_tolerance_ns=600_000_000,
            max_allowable_gap_s=2.0,
        )
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=2_500_000_000,
            eval_value=3.85,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertTrue(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.INTERPOLATED_LINEAR)
        self.assertEqual(pair.observation_type, ObservationProvenanceType.INTERPOLATED_VALUE)
        self.assertAlmostEqual(pair.ref_value, 3.85, places=6)
        self.assertAlmostEqual(pair.error, 0.0, places=6)

    # 9. Interpolation across forbidden gap
    def test_interpolation_across_forbidden_gap_prohibited(self) -> None:
        # Between 5.0s and 10.0s, gap is 5.0s > max_allowable_gap_s (2.0s)
        policy = TemporalAlignmentPolicy(
            alignment_mode=AlignmentMode.NON_CAUSAL_BATCH,
            interpolation_policy=InterpolationPolicy.LINEAR_PIECEWISE,
            timestamp_tolerance_ns=2_000_000_000,
            max_allowable_gap_s=2.0,
        )
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=7_000_000_000,
            eval_value=3.55,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertFalse(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.DATA_GAPPED)
        self.assertIn("exceeds max allowable gap", pair.exclusion_reason)

    # 10. Missing reference signal
    def test_missing_reference_signal(self) -> None:
        pair = TemporalAlignmentEngine.align_sample(
            eval_timestamp_ns=2_000_000_000,
            eval_value=25.0,
            signal_name="temperature_c",
            reference_frames=self.ref_frames,
        )
        self.assertFalse(pair.is_valid_pair)
        self.assertEqual(pair.status, AlignmentStatus.MISSING_REFERENCE)

    # 11. Time series alignment with monotonicity and duplicate rejection
    def test_align_time_series_monotonicity_and_duplicates(self) -> None:
        eval_series = [
            (1_000_000_000, 4.0),
            (2_000_000_000, 3.9),
            (2_000_000_000, 3.9),  # Duplicate
            (1_500_000_000, 3.95), # Retrograde
            (3_000_000_000, 3.8),
        ]
        policy = TemporalAlignmentPolicy(
            reject_non_monotonic=True,
            allow_duplicate_timestamps=False,
        )
        aligned = TemporalAlignmentEngine.align_time_series(
            eval_series=eval_series,
            signal_name="voltage_v",
            reference_frames=self.ref_frames,
            policy=policy,
        )
        self.assertEqual(len(aligned), 5)
        self.assertEqual(aligned[0].status, AlignmentStatus.EXACT_MATCH)
        self.assertEqual(aligned[1].status, AlignmentStatus.EXACT_MATCH)
        self.assertEqual(aligned[2].status, AlignmentStatus.DUPLICATE_REJECTED)
        self.assertEqual(aligned[3].status, AlignmentStatus.RETROGRADE_REJECTED)
        self.assertEqual(aligned[4].status, AlignmentStatus.EXACT_MATCH)

    # 12. Frame alignment
    def test_align_frames_multi_signal(self) -> None:
        meta = DatasetMetadata(
            dataset_id="ds1",
            dataset_name="Test DS",
            version="1.0",
            created_at_utc="2026-01-01",
        )
        ds = ExperimentalDataset(
            metadata=meta,
            frames=tuple(self.ref_frames),
        )
        eval_frames = [
            (1_000_000_000, {"voltage_v": 4.01}),
            (2_000_000_000, {"voltage_v": 3.89}),
            (3_000_000_000, {"voltage_v": 3.80}),
        ]
        frame_pairs = TemporalAlignmentEngine.align_frames(
            eval_frames=eval_frames,
            reference_dataset=ds,
        )
        self.assertEqual(len(frame_pairs), 3)
        self.assertTrue(all(fp.status == AlignmentStatus.EXACT_MATCH for fp in frame_pairs))
        self.assertAlmostEqual(frame_pairs[0].get_signal_pair("voltage_v").error, -0.01, places=6)


if __name__ == "__main__":
    unittest.main()
