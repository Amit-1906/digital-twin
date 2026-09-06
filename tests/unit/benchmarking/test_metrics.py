"""Unit tests for Level 5.5 Phase 2 Metric Engine, Statistical Accumulators, and Reports."""

import math
import unittest

from src.benchmarking.alignment import (
    AlignedFramePair,
    AlignedSignalPair,
    AlignmentStatus,
    ObservationProvenanceType,
    TemporalAlignmentPolicy,
)
from src.benchmarking.dataset import ReferenceObservationFrame
from src.benchmarking.metrics import (
    BenchmarkMetricsCalculator,
    SignalMetricSummary,
    SingleMetricValue,
    StratifiedMetricReport,
    StreamingWelfordAccumulator,
)
from src.benchmarking.stratification import (
    ContextDimension,
    ContextSliceIdentifier,
    OperatingContextStratifier,
    StratificationConfig,
)
from src.benchmarking.types import (
    BenchmarkSemanticClass,
    CalibrationStatus,
    EvidenceRole,
    ReferenceMeasurement,
    ReferenceProvenance,
    ReferenceSourceType,
)


class TestBenchmarkingMetrics(unittest.TestCase):
    """Test suite for Welford statistical accumulators and evidence-qualified metric calculation."""

    def setUp(self) -> None:
        self.prov_lab = ReferenceProvenance(
            provenance_id="prov_lab_001",
            source_type=ReferenceSourceType.LABORATORY_METROLOGY,
            calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
            evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
            measurement_method="FOUR_WIRE_KELVIN",
            instrument_identifier="MACCOR_SERIES_4000",
        )
        self.prov_bms = ReferenceProvenance(
            provenance_id="prov_bms_001",
            source_type=ReferenceSourceType.BMS_FIRMWARE,
            calibration_status=CalibrationStatus.FACTORY_NOMINAL,
            evidence_role=EvidenceRole.BMS_ESTIMATE_BASELINE,
            measurement_method="COULOMB_COUNTING_LOOKUP",
        )

    # 1. Zero-error case
    def test_zero_error_metrics(self) -> None:
        pairs = [
            AlignedSignalPair(
                signal_name="voltage_v",
                eval_timestamp_ns=i * 1_000_000_000,
                eval_value=3.5 + i * 0.1,
                ref_timestamp_ns=i * 1_000_000_000,
                ref_value=3.5 + i * 0.1,
                ref_uncertainty=0.005,
                status=AlignmentStatus.EXACT_MATCH,
                observation_type=ObservationProvenanceType.DIRECT_OBSERVATION,
                provenance=self.prov_lab,
                is_valid_pair=True,
                error=0.0,
                is_uncertainty_consistent=True,
            )
            for i in range(10)
        ]
        summary = BenchmarkMetricsCalculator.compute_signal_metrics(
            pairs=pairs,
            signal_name="voltage_v",
            unit="V",
            provenance=self.prov_lab,
        )
        self.assertEqual(summary.valid_paired_samples, 10)
        self.assertEqual(summary.excluded_samples, 0)
        self.assertEqual(summary.valid_pair_coverage, 1.0)
        self.assertEqual(summary.semantic_class, BenchmarkSemanticClass.PHYSICAL_REFERENCE_VALIDATION)
        self.assertEqual(summary.evidence_role, EvidenceRole.PHYSICAL_REFERENCE)

        self.assertEqual(summary.mae.value, 0.0)
        self.assertEqual(summary.rmse.value, 0.0)
        self.assertEqual(summary.mean_bias_error.value, 0.0)
        self.assertEqual(summary.max_absolute_error.value, 0.0)
        self.assertAlmostEqual(summary.r_squared.value, 1.0, places=6)
        self.assertEqual(summary.uncertainty_consistent_fraction, 1.0)

    # 2. Known analytical error values
    def test_known_analytical_metrics(self) -> None:
        # ref = [4.0, 4.0, 4.0, 4.0], eval = [3.9, 4.2, 3.7, 4.4]
        # errors (ref - eval): [0.1, -0.2, 0.3, -0.4]
        # abs errors: [0.1, 0.2, 0.3, 0.4] -> MAE = 0.25
        # sq errors: [0.01, 0.04, 0.09, 0.16] -> MSE = 0.075 -> RMSE = sqrt(0.075) = 0.273861
        # sum errors: 0.1 - 0.2 + 0.3 - 0.4 = -0.2 -> MBE = -0.05
        # max error = 0.4
        eval_vals = [3.9, 4.2, 3.7, 4.4]
        pairs = [
            AlignedSignalPair(
                signal_name="voltage_v",
                eval_timestamp_ns=i * 1_000_000_000,
                eval_value=eval_vals[i],
                ref_timestamp_ns=i * 1_000_000_000,
                ref_value=4.0 + (i * 0.1), # Vary reference so variance > 0
                ref_uncertainty=0.15,
                status=AlignmentStatus.EXACT_MATCH,
                is_valid_pair=True,
            )
            for i in range(4)
        ]
        # ref_vals = [4.0, 4.1, 4.2, 4.3], eval_vals = [3.9, 4.2, 3.7, 4.4]
        # errors: [4.0-3.9=0.1, 4.1-4.2=-0.1, 4.2-3.7=0.5, 4.3-4.4=-0.1]
        # abs errors: [0.1, 0.1, 0.5, 0.1] -> MAE = 0.8 / 4 = 0.2
        # sq errors: [0.01, 0.01, 0.25, 0.01] = 0.28 -> MSE = 0.07 -> RMSE = sqrt(0.07) ~ 0.264575
        # errors: [0.1, -0.1, 0.5, -0.1] -> sum = 0.4 -> MBE = 0.10
        # max error = 0.5
        summary = BenchmarkMetricsCalculator.compute_signal_metrics(
            pairs=pairs,
            signal_name="voltage_v",
            unit="V",
            provenance=self.prov_lab,
        )
        self.assertEqual(summary.valid_paired_samples, 4)
        self.assertAlmostEqual(summary.mae.value, 0.20, places=6)
        self.assertAlmostEqual(summary.rmse.value, math.sqrt(0.07), places=6)
        self.assertAlmostEqual(summary.mean_bias_error.value, 0.10, places=6)
        self.assertAlmostEqual(summary.max_absolute_error.value, 0.50, places=6)
        self.assertAlmostEqual(summary.mean_reference_uncertainty, 0.15, places=6)
        # 3 out of 4 errors (0.1, 0.1, 0.1) are <= 0.15 uncertainty -> UCF = 0.75
        self.assertAlmostEqual(summary.uncertainty_consistent_fraction, 0.75, places=6)

    # 3. R² zero reference variance handling
    def test_r_squared_zero_reference_variance(self) -> None:
        # Constant reference signal (e.g. at 4.0V)
        pairs = [
            AlignedSignalPair(
                signal_name="voltage_v",
                eval_timestamp_ns=i * 1_000_000_000,
                eval_value=3.99 + (i * 0.005),
                ref_timestamp_ns=i * 1_000_000_000,
                ref_value=4.0, # Constant!
                status=AlignmentStatus.EXACT_MATCH,
                is_valid_pair=True,
            )
            for i in range(5)
        ]
        summary = BenchmarkMetricsCalculator.compute_signal_metrics(
            pairs=pairs,
            signal_name="voltage_v",
            unit="V",
        )
        self.assertEqual(summary.r_squared.status, "UNDEFINED_ZERO_REFERENCE_VARIANCE")
        self.assertIsNone(summary.r_squared.value)
        self.assertIn("Reference signal variance is zero", summary.r_squared.diagnostic_message)

    # 4. Insufficient samples for R²
    def test_r_squared_insufficient_samples(self) -> None:
        pairs = [
            AlignedSignalPair(
                signal_name="voltage_v",
                eval_timestamp_ns=1_000_000_000,
                eval_value=3.9,
                ref_timestamp_ns=1_000_000_000,
                ref_value=4.0,
                status=AlignmentStatus.EXACT_MATCH,
                is_valid_pair=True,
            )
        ]
        summary = BenchmarkMetricsCalculator.compute_signal_metrics(
            pairs=pairs,
            signal_name="voltage_v",
            unit="V",
            min_samples_for_r_squared=2,
        )
        self.assertEqual(summary.r_squared.status, "INSUFFICIENT_SAMPLES")
        self.assertIsNone(summary.r_squared.value)

    # 5. Non-finite values and exclusions
    def test_non_finite_values_exclusion_tracking(self) -> None:
        pairs = [
            AlignedSignalPair(signal_name="v", eval_timestamp_ns=1, eval_value=4.0, ref_value=4.0, is_valid_pair=True),
            AlignedSignalPair(signal_name="v", eval_timestamp_ns=2, eval_value=float("nan"), ref_value=4.0, is_valid_pair=True),
            AlignedSignalPair(signal_name="v", eval_timestamp_ns=3, eval_value=4.0, ref_value=float("inf"), is_valid_pair=True),
            AlignedSignalPair(signal_name="v", eval_timestamp_ns=4, eval_value=4.0, status=AlignmentStatus.OUTSIDE_TOLERANCE, exclusion_reason="Exceeded tolerance", is_valid_pair=False),
        ]
        summary = BenchmarkMetricsCalculator.compute_signal_metrics(
            pairs=pairs,
            signal_name="v",
            unit="V",
        )
        self.assertEqual(summary.total_eval_samples, 4)
        self.assertEqual(summary.valid_paired_samples, 1)
        self.assertEqual(summary.excluded_samples, 3)
        self.assertEqual(summary.valid_pair_coverage, 0.25)
        self.assertEqual(summary.exclusion_reasons["NON_FINITE_VALUE"], 2)
        self.assertEqual(summary.exclusion_reasons["Exceeded tolerance"], 1)

    # 6. Evidence semantic deduction
    def test_evidence_semantics_deduction(self) -> None:
        # 1. BMS firmware estimate
        sum_bms = BenchmarkMetricsCalculator.compute_signal_metrics(
            pairs=[],
            signal_name="soc",
            unit="fraction",
            provenance=self.prov_bms,
        )
        self.assertEqual(sum_bms.semantic_class, BenchmarkSemanticClass.BMS_ESTIMATE_CROSS_CHECK)
        self.assertEqual(sum_bms.evidence_role, EvidenceRole.BMS_ESTIMATE_BASELINE)

        # 2. Lab physical reference
        sum_lab = BenchmarkMetricsCalculator.compute_signal_metrics(
            pairs=[],
            signal_name="v",
            unit="V",
            provenance=self.prov_lab,
        )
        self.assertEqual(sum_lab.semantic_class, BenchmarkSemanticClass.PHYSICAL_REFERENCE_VALIDATION)
        self.assertEqual(sum_lab.evidence_role, EvidenceRole.PHYSICAL_REFERENCE)

    # 7. Stratified reporting across frame pairs
    def test_stratified_reporting(self) -> None:
        strat = OperatingContextStratifier()
        strat_cfg = StratificationConfig(
            nominal_capacity_ah=100.0,
            rest_current_threshold_a=0.05,
        )
        # Create 10 frame pairs: 5 in REST (0A), 5 in DISCHARGE (50A)
        frame_pairs = []
        for i in range(10):
            ts = i * 1_000_000_000
            curr = 0.0 if i < 5 else 50.0
            volt_ref = 3.9 - (i * 0.05)
            volt_sim = 3.9 - (i * 0.05) + 0.01

            meas = ReferenceMeasurement(value=volt_ref, unit="V", timestamp_ns=ts, provenance=self.prov_lab)
            ref_frame = ReferenceObservationFrame(timestamp_ns=ts, reference_signals={"voltage_v": meas})
            sig_pair = AlignedSignalPair(
                signal_name="voltage_v",
                eval_timestamp_ns=ts,
                eval_value=volt_sim,
                ref_timestamp_ns=ts,
                ref_value=volt_ref,
                provenance=self.prov_lab,
                is_valid_pair=True,
                status=AlignmentStatus.EXACT_MATCH,
            )
            fp = AlignedFramePair(
                eval_timestamp_ns=ts,
                eval_signals={"pack_current_a": curr, "voltage_v": volt_sim},
                ref_frame=ref_frame,
                aligned_signals={"voltage_v": sig_pair},
            )
            frame_pairs.append(fp)

        report = BenchmarkMetricsCalculator.compute_stratified_metrics(
            frame_pairs=frame_pairs,
            signal_units={"voltage_v": "V"},
            stratifier=strat,
            stratification_config=strat_cfg,
        )
        self.assertEqual(report.total_samples_evaluated, 10)
        self.assertEqual(report.overall_valid_pairs, 10)

        # Overall summary
        v_overall = report.get_overall("voltage_v")
        self.assertIsNotNone(v_overall)
        self.assertEqual(v_overall.valid_paired_samples, 10)
        self.assertAlmostEqual(v_overall.mae.value, 0.01, places=6)

        # Stratified slices
        v_rest = report.get_stratified("REGIME:REST", "voltage_v")
        self.assertIsNotNone(v_rest)
        self.assertEqual(v_rest.valid_paired_samples, 5)

        v_dis = report.get_stratified("REGIME:DISCHARGE", "voltage_v")
        self.assertIsNotNone(v_dis)
        self.assertEqual(v_dis.valid_paired_samples, 5)

        d = report.to_dict()
        self.assertIn("overall_metrics", d)
        self.assertIn("stratified_metrics", d)


if __name__ == "__main__":
    unittest.main()
