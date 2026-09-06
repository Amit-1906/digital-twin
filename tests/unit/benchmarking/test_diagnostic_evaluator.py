"""Unit tests for Level 5.5 Phase 3 Diagnostic Ground-Truth Evaluator."""

import unittest

from src.benchmarking.diagnostic_evaluator import (
    DetectionLatencySummary,
    DiagnosticEvaluationReport,
    DiagnosticGroundTruthEvaluator,
    EventDiagnosticEvaluation,
    EventMatchStatus,
    GroundTruthEventType,
    SampleConfusionMatrix,
)
from src.benchmarking.types import (
    BenchmarkSemanticClass,
    EvidenceRole,
    ExperimentalGroundTruthEvent,
)
from src.diagnostics.types import (
    DiagnosticAssessmentReport,
    DiagnosticCategory,
    DiagnosticSeverity,
    FaultLifecycleState,
    OperatingContext,
    RootCauseHypothesis,
)


class TestDiagnosticGroundTruthEvaluator(unittest.TestCase):
    """Comprehensive test suite for diagnostic ground-truth event matching, confusion matrices, and latency."""

    def setUp(self) -> None:
        self.hyp_thermal = RootCauseHypothesis(
            hypothesis_id="HYP_THERMAL_IMPAIRMENT",
            title="Thermal Dissipation Impairment",
            category=DiagnosticCategory.THERMAL,
            evidence_score=0.92,
            confidence_level="STRONG",
            required_signal_coverage=1.0,
            optional_signal_coverage=1.0,
            is_critical_eligible=True,
        )
        self.hyp_sensor = RootCauseHypothesis(
            hypothesis_id="HYP_SENSOR_DRIFT",
            title="Sensor Drift",
            category=DiagnosticCategory.SENSOR,
            evidence_score=0.88,
            confidence_level="STRONG",
            required_signal_coverage=1.0,
            optional_signal_coverage=1.0,
        )

    # 1. Exact match and positive detection latency
    def test_exact_event_match_and_positive_latency(self) -> None:
        # Ground truth fault from 10.0s to 30.0s
        event = ExperimentalGroundTruthEvent(
            event_id="ev_therm_01",
            start_timestamp_ns=10_000_000_000,
            end_timestamp_ns=30_000_000_000,
            event_type="THERMAL_DISSIPATION_IMPAIRMENT",
            expected_hypothesis_id="HYP_THERMAL_IMPAIRMENT",
        )
        # Reports at 0s-9s (NORMAL), 10s-11s (NORMAL), 12s (DIAGNOSED at t=12s -> latency 2.0s), 13s-30s (DIAGNOSED)
        reports = []
        for i in range(35):
            ts = i * 1_000_000_000
            if i < 12:
                rep = DiagnosticAssessmentReport(
                    assessment_id=f"rep_{i}",
                    system_id="twin_01",
                    timestamp_ns=ts,
                    lifecycle_state=FaultLifecycleState.NORMAL,
                    severity=DiagnosticSeverity.INFORMATIONAL,
                    operating_context=OperatingContext.DISCHARGE_CC,
                )
            elif i <= 30:
                rep = DiagnosticAssessmentReport(
                    assessment_id=f"rep_{i}",
                    system_id="twin_01",
                    timestamp_ns=ts,
                    lifecycle_state=FaultLifecycleState.DIAGNOSED,
                    severity=DiagnosticSeverity.WARNING,
                    operating_context=OperatingContext.DISCHARGE_CC,
                    primary_hypothesis=self.hyp_thermal,
                )
            else:
                rep = DiagnosticAssessmentReport(
                    assessment_id=f"rep_{i}",
                    system_id="twin_01",
                    timestamp_ns=ts,
                    lifecycle_state=FaultLifecycleState.RECOVERED,
                    severity=DiagnosticSeverity.INFORMATIONAL,
                    operating_context=OperatingContext.REST,
                )
            reports.append(rep)

        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[event],
        )

        self.assertEqual(len(eval_report.event_evaluations), 1)
        ev_eval = eval_report.event_evaluations[0]
        self.assertTrue(ev_eval.is_detected)
        self.assertEqual(ev_eval.match_status, EventMatchStatus.DETECTED_MATCH)
        self.assertEqual(ev_eval.detection_timestamp_ns, 12_000_000_000)
        self.assertAlmostEqual(ev_eval.latency_seconds, 2.0, places=4)
        self.assertEqual(ev_eval.diagnosed_hypothesis_id, "HYP_THERMAL_IMPAIRMENT")
        self.assertTrue(ev_eval.is_hypothesis_matched)

        # Latency summary
        self.assertEqual(eval_report.latency_summary.detected_events, 1)
        self.assertEqual(eval_report.latency_summary.missed_events, 0)
        self.assertAlmostEqual(eval_report.latency_summary.mean_latency_s, 2.0, places=4)

        # Sample Confusion Matrix:
        # Event window [10s, 30s]: 21 samples (10 to 30)
        # Inside window: t=10, 11 are NORMAL (FN=2), t=12..30 are DIAGNOSED (TP=19) -> TP=19, FN=2
        # Outside window: t=0..9 (TN=10), t=31..34 (TN=4) -> TN=14, FP=0
        matrix = eval_report.sample_confusion_matrix
        self.assertEqual(matrix.true_positives, 19)
        self.assertEqual(matrix.false_negatives, 2)
        self.assertEqual(matrix.true_negatives, 14)
        self.assertEqual(matrix.false_positives, 0)
        self.assertEqual(matrix.total_samples, 35)
        self.assertEqual(matrix.precision, 1.0) # 19 / (19 + 0)
        self.assertAlmostEqual(matrix.recall, 19.0 / 21.0, places=4)

    # 2. Early detection (negative latency)
    def test_early_detection_negative_latency(self) -> None:
        event = ExperimentalGroundTruthEvent(
            event_id="ev_01",
            start_timestamp_ns=20_000_000_000,
            end_timestamp_ns=30_000_000_000,
            event_type="THERMAL_FAULT",
        )
        # Diagnosis reported at t=15s (5 seconds before event start!)
        reports = [
            DiagnosticAssessmentReport(
                assessment_id="rep_15",
                system_id="twin_01",
                timestamp_ns=15_000_000_000,
                lifecycle_state=FaultLifecycleState.DIAGNOSED,
                severity=DiagnosticSeverity.WARNING,
                operating_context=OperatingContext.DISCHARGE_CC,
                primary_hypothesis=self.hyp_thermal,
            )
        ]
        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[event],
        )
        ev_eval = eval_report.event_evaluations[0]
        self.assertEqual(ev_eval.match_status, EventMatchStatus.DETECTED_EARLY)
        self.assertEqual(ev_eval.latency_seconds, -5.0)
        self.assertEqual(eval_report.latency_summary.early_detections, 1)

    # 3. Wrong hypothesis detection
    def test_detected_wrong_hypothesis(self) -> None:
        event = ExperimentalGroundTruthEvent(
            event_id="ev_01",
            start_timestamp_ns=10_000_000_000,
            end_timestamp_ns=20_000_000_000,
            event_type="THERMAL_FAULT",
            expected_hypothesis_id="HYP_THERMAL_IMPAIRMENT",
        )
        reports = [
            DiagnosticAssessmentReport(
                assessment_id="rep_12",
                system_id="twin_01",
                timestamp_ns=12_000_000_000,
                lifecycle_state=FaultLifecycleState.DIAGNOSED,
                severity=DiagnosticSeverity.WARNING,
                operating_context=OperatingContext.DISCHARGE_CC,
                primary_hypothesis=self.hyp_sensor,  # Sensor drift instead of thermal
            )
        ]
        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[event],
        )
        ev_eval = eval_report.event_evaluations[0]
        self.assertEqual(ev_eval.match_status, EventMatchStatus.DETECTED_WRONG_HYPOTHESIS)
        self.assertFalse(ev_eval.is_hypothesis_matched)

    # 4. Missed event (False negative)
    def test_missed_event(self) -> None:
        event = ExperimentalGroundTruthEvent(
            event_id="ev_01",
            start_timestamp_ns=10_000_000_000,
            end_timestamp_ns=20_000_000_000,
            event_type="THERMAL_FAULT",
        )
        reports = [
            DiagnosticAssessmentReport(
                assessment_id=f"rep_{i}",
                system_id="twin_01",
                timestamp_ns=i * 1_000_000_000,
                lifecycle_state=FaultLifecycleState.NORMAL,
                severity=DiagnosticSeverity.INFORMATIONAL,
                operating_context=OperatingContext.DISCHARGE_CC,
            )
            for i in range(10, 21)
        ]
        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[event],
        )
        self.assertEqual(eval_report.event_evaluations[0].match_status, EventMatchStatus.MISSED_EVENT)
        self.assertFalse(eval_report.event_evaluations[0].is_detected)
        self.assertEqual(eval_report.sample_confusion_matrix.false_negatives, 11)
        self.assertEqual(eval_report.sample_confusion_matrix.true_positives, 0)

    # 5. False positive outside event windows
    def test_false_positive_detection(self) -> None:
        # No ground truth event active
        reports = [
            DiagnosticAssessmentReport(
                assessment_id="rep_1",
                system_id="twin_01",
                timestamp_ns=1_000_000_000,
                lifecycle_state=FaultLifecycleState.DIAGNOSED,
                severity=DiagnosticSeverity.WARNING,
                operating_context=OperatingContext.DISCHARGE_CC,
                primary_hypothesis=self.hyp_thermal,
            )
        ]
        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[], # Nominal baseline
        )
        self.assertEqual(eval_report.sample_confusion_matrix.false_positives, 1)
        self.assertEqual(eval_report.sample_confusion_matrix.true_negatives, 0)
        self.assertEqual(eval_report.sample_confusion_matrix.precision, 0.0)

    # 6. Unobservable intervals excluded from TN/FN
    def test_unobservable_intervals_excluded_from_confusion(self) -> None:
        # Unobservable interval between 10s and 20s
        reports = [
            DiagnosticAssessmentReport(
                assessment_id=f"rep_{i}",
                system_id="twin_01",
                timestamp_ns=i * 1_000_000_000,
                lifecycle_state=FaultLifecycleState.NORMAL,
                severity=DiagnosticSeverity.INFORMATIONAL,
                operating_context=OperatingContext.DISCHARGE_CC,
            )
            for i in range(5, 25)
        ]
        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[],
            unobservable_intervals=[(10_000_000_000, 20_000_000_000)],
        )
        # Total 20 samples: 11 samples (10 to 20) are unobservable, remaining 9 samples are TN
        self.assertEqual(eval_report.sample_confusion_matrix.unknown_samples, 11)
        self.assertEqual(eval_report.sample_confusion_matrix.true_negatives, 9)
        self.assertEqual(eval_report.sample_confusion_matrix.false_negatives, 0)
        self.assertEqual(eval_report.sample_confusion_matrix.total_samples, 20)

    # 8. Zero detection latency
    def test_zero_detection_latency(self) -> None:
        event = ExperimentalGroundTruthEvent(
            event_id="ev_zero",
            start_timestamp_ns=10_000_000_000,
            end_timestamp_ns=20_000_000_000,
            event_type="INSTANT_FAULT",
            expected_hypothesis_id="HYP_THERMAL_IMPAIRMENT",
        )
        reports = [
            DiagnosticAssessmentReport(
                assessment_id="rep_zero",
                system_id="twin_01",
                timestamp_ns=10_000_000_000,
                lifecycle_state=FaultLifecycleState.DIAGNOSED,
                severity=DiagnosticSeverity.CRITICAL,
                operating_context=OperatingContext.DISCHARGE_CC,
                primary_hypothesis=self.hyp_thermal,
            )
        ]
        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[event],
        )
        ev_eval = eval_report.event_evaluations[0]
        self.assertEqual(ev_eval.match_status, EventMatchStatus.DETECTED_MATCH)
        self.assertEqual(ev_eval.latency_seconds, 0.0)
        self.assertEqual(eval_report.latency_summary.min_latency_s, 0.0)

    # 9. Overlapping ground-truth events
    def test_overlapping_events_evaluation(self) -> None:
        # Event A: 10s to 20s (Thermal)
        ev_a = ExperimentalGroundTruthEvent(
            event_id="ev_a",
            start_timestamp_ns=10_000_000_000,
            end_timestamp_ns=20_000_000_000,
            event_type="THERMAL_FAULT",
            expected_hypothesis_id="HYP_THERMAL_IMPAIRMENT",
        )
        # Event B: 15s to 30s (Sensor)
        ev_b = ExperimentalGroundTruthEvent(
            event_id="ev_b",
            start_timestamp_ns=15_000_000_000,
            end_timestamp_ns=30_000_000_000,
            event_type="SENSOR_FAULT",
            expected_hypothesis_id="HYP_SENSOR_DRIFT",
        )
        reports = [
            DiagnosticAssessmentReport(
                assessment_id="rep_10",
                system_id="twin_01",
                timestamp_ns=10_000_000_000,
                lifecycle_state=FaultLifecycleState.DIAGNOSED,
                severity=DiagnosticSeverity.WARNING,
                operating_context=OperatingContext.DISCHARGE_CC,
                primary_hypothesis=self.hyp_thermal,
            ),
            DiagnosticAssessmentReport(
                assessment_id="rep_16",
                system_id="twin_01",
                timestamp_ns=16_000_000_000,
                lifecycle_state=FaultLifecycleState.DIAGNOSED,
                severity=DiagnosticSeverity.WARNING,
                operating_context=OperatingContext.DISCHARGE_CC,
                primary_hypothesis=self.hyp_sensor,
            ),
        ]
        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[ev_a, ev_b],
        )
        self.assertEqual(len(eval_report.event_evaluations), 2)
        eval_a = eval_report.event_evaluations[0]
        eval_b = eval_report.event_evaluations[1]

        self.assertEqual(eval_a.event_id, "ev_a")
        self.assertEqual(eval_a.match_status, EventMatchStatus.DETECTED_MATCH)
        self.assertEqual(eval_a.latency_seconds, 0.0)

        self.assertEqual(eval_b.event_id, "ev_b")
        self.assertEqual(eval_b.match_status, EventMatchStatus.DETECTED_MATCH)
        self.assertAlmostEqual(eval_b.latency_seconds, 1.0, places=4)

    # 10. Data quality failed state handled as unobservable/unknown
    def test_data_quality_failed_lifecycle_excluded_from_confusion(self) -> None:
        reports = [
            DiagnosticAssessmentReport(
                assessment_id="rep_dq",
                system_id="twin_01",
                timestamp_ns=5_000_000_000,
                lifecycle_state=FaultLifecycleState.DATA_QUALITY_FAILED,
                severity=DiagnosticSeverity.WARNING,
                operating_context=OperatingContext.DISCHARGE_CC,
            )
        ]
        eval_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=reports,
            ground_truth_events=[],
        )
        self.assertEqual(eval_report.sample_confusion_matrix.unknown_samples, 1)
        self.assertEqual(eval_report.sample_confusion_matrix.true_negatives, 0)
        self.assertEqual(eval_report.sample_confusion_matrix.false_positives, 0)

    # 11. Synthetic vs Physical semantic class propagation
    def test_synthetic_vs_physical_evidence_classification(self) -> None:
        synth_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=[],
            ground_truth_events=[],
            semantic_class=BenchmarkSemanticClass.ALGORITHMIC_SYNTHETIC_VERIFICATION,
        )
        self.assertEqual(synth_report.semantic_class, BenchmarkSemanticClass.ALGORITHMIC_SYNTHETIC_VERIFICATION)
        self.assertEqual(synth_report.evidence_role, EvidenceRole.SYNTHETIC_REFERENCE)

        phys_report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=[],
            ground_truth_events=[],
            semantic_class=BenchmarkSemanticClass.PHYSICAL_REFERENCE_VALIDATION,
        )
        self.assertEqual(phys_report.semantic_class, BenchmarkSemanticClass.PHYSICAL_REFERENCE_VALIDATION)
    # 12. Ground-truth event types enum
    def test_ground_truth_event_types_enum(self) -> None:
        self.assertEqual(GroundTruthEventType.KNOWN_PHYSICAL_EVENT.value, "KNOWN_PHYSICAL_EVENT")
        self.assertEqual(GroundTruthEventType.OBSERVED_FIELD_EVENT.value, "OBSERVED_FIELD_EVENT")
        self.assertEqual(GroundTruthEventType.SYNTHETIC_INJECTED_EVENT.value, "SYNTHETIC_INJECTED_EVENT")
        self.assertEqual(GroundTruthEventType.UNKNOWN_EVENT.value, "UNKNOWN_EVENT")

    # 13. Serialization
    def test_serialization(self) -> None:
        event = ExperimentalGroundTruthEvent(
            event_id="ev_01",
            start_timestamp_ns=10_000_000_000,
            end_timestamp_ns=20_000_000_000,
            event_type="FAULT",
        )
        report = DiagnosticGroundTruthEvaluator.evaluate(
            diagnostic_reports=[],
            ground_truth_events=[event],
        )
        d = report.to_dict()
        self.assertIn("latency_summary", d)
        self.assertIn("sample_confusion_matrix", d)
        self.assertIn("event_evaluations", d)


if __name__ == "__main__":
    unittest.main()


