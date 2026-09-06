"""Unit tests for Diagnostic Domain Events (Level 5.4 Phase 4)."""

import unittest
from src.diagnostics.events import (
    BatteryDiagnosticAssessedEvent,
    CriticalAdvisoryAlertEvent,
)
from src.diagnostics.types import (
    DiagnosticAssessmentReport,
    DiagnosticCategory,
    DiagnosticSeverity,
    FaultLifecycleState,
    OperatingContext,
    RootCauseHypothesis,
)


class TestDiagnosticEvents(unittest.TestCase):
    """Test suite verifying diagnostic event instantiation, serialization, and immutability."""

    def setUp(self) -> None:
        self.report = DiagnosticAssessmentReport(
            assessment_id="diag_pack_01_1000",
            system_id="pack_01",
            timestamp_ns=1_000_000_000,
            lifecycle_state=FaultLifecycleState.DIAGNOSED,
            severity=DiagnosticSeverity.WARNING,
            operating_context=OperatingContext.REST,
            primary_hypothesis=RootCauseHypothesis(
                hypothesis_id="HYP_SENSOR_DRIFT",
                title="Sensor Drift",
                category=DiagnosticCategory.SENSOR,
                evidence_score=0.85,
                confidence_level="STRONG",
                required_signal_coverage=1.0,
                optional_signal_coverage=1.0,
            ),
            alternative_hypotheses=(),
            explanation_narrative="Sensor drift suspected.",
            recommended_operator_actions=["Inspect ADC sensor."],
        )

    def test_battery_diagnostic_assessed_event_metadata_and_payload(self) -> None:
        """BatteryDiagnosticAssessedEvent initializes source_id, timestamp_ns, and payload from report."""
        evt = BatteryDiagnosticAssessedEvent(diagnostic_report=self.report)
        self.assertEqual(evt.event_type, "twin.diagnostic")
        self.assertEqual(evt.source_id, "pack_01")
        self.assertEqual(evt.timestamp_ns, 1_000_000_000)
        self.assertIn("lifecycle_state", evt.payload)
        self.assertEqual(evt.payload["lifecycle_state"], "DIAGNOSED")
        self.assertEqual(evt.payload["severity"], "WARNING")

    def test_critical_advisory_alert_event_metadata_and_payload(self) -> None:
        """CriticalAdvisoryAlertEvent initializes source_id, timestamp_ns, and payload from report."""
        crit_report = DiagnosticAssessmentReport(
            assessment_id="diag_pack_01_crit",
            system_id="pack_01",
            timestamp_ns=2_000_000_000,
            lifecycle_state=FaultLifecycleState.DIAGNOSED_CRITICAL,
            severity=DiagnosticSeverity.CRITICAL,
            operating_context=OperatingContext.DYNAMIC_TRANSIENT,
            primary_hypothesis=RootCauseHypothesis(
                hypothesis_id="HYP_THERMAL_IMPAIRMENT",
                title="Thermal Impairment",
                category=DiagnosticCategory.THERMAL,
                evidence_score=0.95,
                confidence_level="STRONG",
                required_signal_coverage=1.0,
                optional_signal_coverage=1.0,
                is_critical_eligible=True,
            ),
            explanation_narrative="Thermal dissipation impairment detected under high load.",
            recommended_operator_actions=["Inspect cooling airflow."],
        )
        evt = CriticalAdvisoryAlertEvent(diagnostic_report=crit_report)
        self.assertEqual(evt.event_type, "twin.diagnostic.critical_advisory")
        self.assertEqual(evt.source_id, "pack_01")
        self.assertEqual(evt.timestamp_ns, 2_000_000_000)
        self.assertEqual(evt.payload["lifecycle_state"], "DIAGNOSED_CRITICAL")

    def test_event_immutability(self) -> None:
        """Diagnostic events are frozen dataclasses and reject attribute assignment."""
        evt = BatteryDiagnosticAssessedEvent(diagnostic_report=self.report)
        with self.assertRaises(Exception):
            evt.source_id = "other_id"  # type: ignore


if __name__ == "__main__":
    unittest.main()
