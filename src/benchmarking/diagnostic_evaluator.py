"""Diagnostic Ground-Truth Evaluator for Battery Benchmarking.

Evaluates Level 5.4 Explainable Diagnostic Assessment Reports against reference and experimental
ground-truth events, calculating deterministic event matching, detection latency, sample-level
confusion matrices, and explainability grounding without probabilistic claims.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from src.benchmarking.types import (
    BenchmarkSemanticClass,
    EvidenceRole,
    ExperimentalGroundTruthEvent,
)
from src.diagnostics.types import (
    DiagnosticAssessmentReport,
    FaultLifecycleState,
)


class GroundTruthEventType(str, Enum):
    """Categorical classification of experimental ground truth evidence."""

    KNOWN_PHYSICAL_EVENT = "KNOWN_PHYSICAL_EVENT"       # Physically induced/instrumented fault on real hardware
    OBSERVED_FIELD_EVENT = "OBSERVED_FIELD_EVENT"       # Logged physical event from field operation
    SYNTHETIC_INJECTED_EVENT = "SYNTHETIC_INJECTED_EVENT" # Mathematically simulated fault injection
    UNKNOWN_EVENT = "UNKNOWN_EVENT"                     # Uncharacterized or unconfirmed event


class EventMatchStatus(str, Enum):
    """Evaluation status of matching a ground-truth event against diagnostic reports."""

    DETECTED_MATCH = "DETECTED_MATCH"                   # Fault detected with matching hypothesis within event window
    DETECTED_EARLY = "DETECTED_EARLY"                   # Detection occurred before event start (negative latency)
    DETECTED_WRONG_HYPOTHESIS = "DETECTED_WRONG_HYPOTHESIS" # Detected fault, but primary hypothesis differed
    MISSED_EVENT = "MISSED_EVENT"                       # Event active, but system remained normal / undetected
    UNOBSERVABLE = "UNOBSERVABLE"                       # Event interval lacks sufficient reference data


@dataclass(frozen=True)
class EventDiagnosticEvaluation:
    """Individual ground-truth event evaluation result."""

    event_id: str
    event_type: str
    expected_hypothesis_id: Optional[str]
    start_timestamp_ns: int
    end_timestamp_ns: Optional[int]
    match_status: EventMatchStatus
    is_detected: bool
    detection_timestamp_ns: Optional[int] = None
    latency_seconds: Optional[float] = None
    diagnosed_hypothesis_id: Optional[str] = None
    is_hypothesis_matched: bool = False
    lifecycle_state_at_detection: Optional[str] = None
    severity_at_detection: Optional[str] = None
    corroborating_channels: Tuple[str, ...] = field(default_factory=tuple)
    explanation_narrative: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes event evaluation to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "expected_hypothesis_id": self.expected_hypothesis_id,
            "start_timestamp_ns": self.start_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "match_status": self.match_status.value,
            "is_detected": self.is_detected,
            "detection_timestamp_ns": self.detection_timestamp_ns,
            "latency_seconds": round(self.latency_seconds, 4) if self.latency_seconds is not None else None,
            "diagnosed_hypothesis_id": self.diagnosed_hypothesis_id,
            "is_hypothesis_matched": self.is_hypothesis_matched,
            "lifecycle_state_at_detection": self.lifecycle_state_at_detection,
            "severity_at_detection": self.severity_at_detection,
            "corroborating_channels": list(self.corroborating_channels),
            "explanation_narrative": self.explanation_narrative,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SampleConfusionMatrix:
    """Sample-level diagnostic classification confusion counts and metrics."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    unknown_samples: int = 0
    total_samples: int = 0

    @property
    def precision(self) -> Optional[float]:
        """Positive Predictive Value: TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return (self.true_positives / float(denom)) if denom > 0 else None

    @property
    def recall(self) -> Optional[float]:
        """Sensitivity / True Positive Rate: TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return (self.true_positives / float(denom)) if denom > 0 else None

    @property
    def f1_score(self) -> Optional[float]:
        """Harmonic mean of Precision and Recall."""
        p = self.precision
        r = self.recall
        if p is not None and r is not None and (p + r) > 0.0:
            return 2.0 * (p * r) / (p + r)
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serializes sample confusion matrix to dictionary."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "unknown_samples": self.unknown_samples,
            "total_samples": self.total_samples,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "f1_score": round(self.f1_score, 4) if self.f1_score is not None else None,
        }


@dataclass(frozen=True)
class DetectionLatencySummary:
    """Statistical summary of fault detection latencies across evaluated events."""

    total_events: int = 0
    detected_events: int = 0
    missed_events: int = 0
    early_detections: int = 0
    min_latency_s: Optional[float] = None
    max_latency_s: Optional[float] = None
    mean_latency_s: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes latency summary to dictionary."""
        return {
            "total_events": self.total_events,
            "detected_events": self.detected_events,
            "missed_events": self.missed_events,
            "early_detections": self.early_detections,
            "min_latency_s": round(self.min_latency_s, 4) if self.min_latency_s is not None else None,
            "max_latency_s": round(self.max_latency_s, 4) if self.max_latency_s is not None else None,
            "mean_latency_s": round(self.mean_latency_s, 4) if self.mean_latency_s is not None else None,
        }


@dataclass(frozen=True)
class DiagnosticEvaluationReport:
    """Comprehensive evaluation report summarizing diagnostic accuracy against ground truth."""

    event_evaluations: Tuple[EventDiagnosticEvaluation, ...]
    sample_confusion_matrix: SampleConfusionMatrix
    latency_summary: DetectionLatencySummary
    evaluated_reports_count: int
    semantic_class: BenchmarkSemanticClass
    evidence_role: EvidenceRole
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes diagnostic evaluation report to dictionary."""
        return {
            "evaluated_reports_count": self.evaluated_reports_count,
            "semantic_class": self.semantic_class.value,
            "evidence_role": self.evidence_role.value,
            "latency_summary": self.latency_summary.to_dict(),
            "sample_confusion_matrix": self.sample_confusion_matrix.to_dict(),
            "event_evaluations": [e.to_dict() for e in self.event_evaluations],
            "metadata": dict(self.metadata),
        }


class DiagnosticGroundTruthEvaluator:
    """Evaluates Level 5.4 Diagnostic Reports against experimental ground-truth events."""

    @classmethod
    def evaluate(
        cls,
        diagnostic_reports: Sequence[DiagnosticAssessmentReport],
        ground_truth_events: Sequence[ExperimentalGroundTruthEvent] = (),
        unobservable_intervals: Sequence[Tuple[int, int]] = (),
        semantic_class: Optional[BenchmarkSemanticClass] = None,
    ) -> DiagnosticEvaluationReport:
        """Executes event-level and sample-level diagnostic evaluation against ground truth.

        Args:
            diagnostic_reports: Ordered sequence of DiagnosticAssessmentReport instances.
            ground_truth_events: Sequence of ExperimentalGroundTruthEvent instances.
            unobservable_intervals: Sequence of (start_ts, end_ts) tuples where fault existence is unestablished.
            semantic_class: Optional semantic classification override.

        Returns:
            DiagnosticEvaluationReport value object.
        """
        sem_class = semantic_class or BenchmarkSemanticClass.PHYSICAL_REFERENCE_VALIDATION
        ev_role = (
            EvidenceRole.PHYSICAL_REFERENCE
            if sem_class == BenchmarkSemanticClass.PHYSICAL_REFERENCE_VALIDATION
            else EvidenceRole.SYNTHETIC_REFERENCE
        )

        n_reports = len(diagnostic_reports)
        if n_reports == 0:
            missed_evals = [
                EventDiagnosticEvaluation(
                    event_id=ev.event_id,
                    event_type=ev.event_type,
                    expected_hypothesis_id=ev.expected_hypothesis_id,
                    start_timestamp_ns=ev.start_timestamp_ns,
                    end_timestamp_ns=ev.end_timestamp_ns,
                    match_status=EventMatchStatus.MISSED_EVENT,
                    is_detected=False,
                )
                for ev in ground_truth_events
            ]
            return DiagnosticEvaluationReport(
                event_evaluations=tuple(missed_evals),
                sample_confusion_matrix=SampleConfusionMatrix(
                    false_negatives=len(ground_truth_events),
                    total_samples=0,
                ),
                latency_summary=DetectionLatencySummary(
                    total_events=len(ground_truth_events),
                    missed_events=len(ground_truth_events),
                ),
                evaluated_reports_count=0,
                semantic_class=sem_class,
                evidence_role=ev_role,
            )

        # 1. Event-Level Evaluation
        event_evals: list[EventDiagnosticEvaluation] = []
        valid_latencies: list[float] = []
        early_count = 0
        missed_count = 0

        for ev in ground_truth_events:
            ev_start = ev.start_timestamp_ns
            ev_end = ev.end_timestamp_ns or float("inf")

            # Check if event is in unobservable interval
            is_unobservable = any(
                (u_start <= ev_start <= u_end) for u_start, u_end in unobservable_intervals
            )
            if is_unobservable:
                event_evals.append(
                    EventDiagnosticEvaluation(
                        event_id=ev.event_id,
                        event_type=ev.event_type,
                        expected_hypothesis_id=ev.expected_hypothesis_id,
                        start_timestamp_ns=ev_start,
                        end_timestamp_ns=ev.end_timestamp_ns,
                        match_status=EventMatchStatus.UNOBSERVABLE,
                        is_detected=False,
                    )
                )
                continue

            # Candidate detection selection:
            # 1. Look for active reports during event window [ev_start, ev_end]
            in_window_reports = [
                rep for rep in diagnostic_reports
                if ev_start <= rep.timestamp_ns <= ev_end
                and rep.lifecycle_state in (
                    FaultLifecycleState.SUSPECTED,
                    FaultLifecycleState.DIAGNOSED,
                    FaultLifecycleState.DIAGNOSED_CRITICAL,
                )
            ]

            detection_report: Optional[DiagnosticAssessmentReport] = None
            if in_window_reports:
                if ev.expected_hypothesis_id is not None:
                    # Find first matching hypothesis in window
                    matching_in_window = [
                        r for r in in_window_reports
                        if r.primary_hypothesis and r.primary_hypothesis.hypothesis_id == ev.expected_hypothesis_id
                    ]
                    detection_report = matching_in_window[0] if matching_in_window else in_window_reports[0]
                else:
                    detection_report = in_window_reports[0]
            else:
                # 2. Look for early reports (before ev_start) or slightly delayed reports
                early_reports = [
                    rep for rep in diagnostic_reports
                    if rep.timestamp_ns < ev_start
                    and rep.lifecycle_state in (
                        FaultLifecycleState.SUSPECTED,
                        FaultLifecycleState.DIAGNOSED,
                        FaultLifecycleState.DIAGNOSED_CRITICAL,
                    )
                ]
                # Filter to reports matching hypothesis or most recent prior detection
                if early_reports:
                    if ev.expected_hypothesis_id is not None:
                        matching_early = [
                            r for r in early_reports
                            if r.primary_hypothesis and r.primary_hypothesis.hypothesis_id == ev.expected_hypothesis_id
                        ]
                        if matching_early:
                            detection_report = matching_early[-1]
                    if detection_report is None:
                        detection_report = early_reports[-1]

            if detection_report is None:
                missed_count += 1
                event_evals.append(
                    EventDiagnosticEvaluation(
                        event_id=ev.event_id,
                        event_type=ev.event_type,
                        expected_hypothesis_id=ev.expected_hypothesis_id,
                        start_timestamp_ns=ev_start,
                        end_timestamp_ns=ev.end_timestamp_ns,
                        match_status=EventMatchStatus.MISSED_EVENT,
                        is_detected=False,
                    )
                )
            else:
                det_ts = detection_report.timestamp_ns
                lat_s = (det_ts - ev_start) / 1_000_000_000.0
                diag_hyp_id = (
                    detection_report.primary_hypothesis.hypothesis_id
                    if detection_report.primary_hypothesis
                    else None
                )
                hyp_matched = (
                    ev.expected_hypothesis_id is None
                    or (diag_hyp_id == ev.expected_hypothesis_id)
                )

                if lat_s < 0.0:
                    early_count += 1
                    status = EventMatchStatus.DETECTED_EARLY
                elif not hyp_matched:
                    status = EventMatchStatus.DETECTED_WRONG_HYPOTHESIS
                else:
                    status = EventMatchStatus.DETECTED_MATCH
                    valid_latencies.append(lat_s)

                channels = ()
                if "corroborating_channels" in detection_report.diagnostics:
                    channels = tuple(detection_report.diagnostics["corroborating_channels"])

                event_evals.append(
                    EventDiagnosticEvaluation(
                        event_id=ev.event_id,
                        event_type=ev.event_type,
                        expected_hypothesis_id=ev.expected_hypothesis_id,
                        start_timestamp_ns=ev_start,
                        end_timestamp_ns=ev.end_timestamp_ns,
                        match_status=status,
                        is_detected=True,
                        detection_timestamp_ns=det_ts,
                        latency_seconds=lat_s,
                        diagnosed_hypothesis_id=diag_hyp_id,
                        is_hypothesis_matched=hyp_matched,
                        lifecycle_state_at_detection=detection_report.lifecycle_state.value,
                        severity_at_detection=detection_report.severity.value,
                        corroborating_channels=channels,
                        explanation_narrative=detection_report.explanation_narrative,
                    )
                )

        # Latency summary
        det_count = sum(1 for e in event_evals if e.is_detected and e.match_status == EventMatchStatus.DETECTED_MATCH)
        latency_summary = DetectionLatencySummary(
            total_events=len(ground_truth_events),
            detected_events=det_count,
            missed_events=missed_count,
            early_detections=early_count,
            min_latency_s=min(valid_latencies) if valid_latencies else None,
            max_latency_s=max(valid_latencies) if valid_latencies else None,
            mean_latency_s=(sum(valid_latencies) / len(valid_latencies)) if valid_latencies else None,
        )

        # 2. Sample-Level Confusion Matrix
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        unknown = 0

        for rep in diagnostic_reports:
            ts = rep.timestamp_ns
            st = rep.lifecycle_state

            # Check if sample lies in an unobservable region
            in_unobservable = any((u_start <= ts <= u_end) for u_start, u_end in unobservable_intervals)
            if in_unobservable or st == FaultLifecycleState.DATA_QUALITY_FAILED:
                unknown += 1
                continue

            # Check if sample lies within any known event window
            active_events = [
                ev for ev in ground_truth_events
                if ev.start_timestamp_ns <= ts <= (ev.end_timestamp_ns or float("inf"))
            ]

            is_diagnosed_fault = st in (
                FaultLifecycleState.SUSPECTED,
                FaultLifecycleState.DIAGNOSED,
                FaultLifecycleState.DIAGNOSED_CRITICAL,
            )

            if active_events:
                # Ground truth indicates a fault is active
                if is_diagnosed_fault:
                    # Check if hypothesis matches any active event
                    matched = any(
                        ev.expected_hypothesis_id is None or (
                            rep.primary_hypothesis and rep.primary_hypothesis.hypothesis_id == ev.expected_hypothesis_id
                        )
                        for ev in active_events
                    )
                    if matched:
                        tp += 1
                    else:
                        fn += 1
                else:
                    fn += 1
            else:
                # Ground truth indicates nominal operation
                if is_diagnosed_fault:
                    fp += 1
                else:
                    tn += 1

        sample_matrix = SampleConfusionMatrix(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            unknown_samples=unknown,
            total_samples=n_reports,
        )

        return DiagnosticEvaluationReport(
            event_evaluations=tuple(event_evals),
            sample_confusion_matrix=sample_matrix,
            latency_summary=latency_summary,
            evaluated_reports_count=n_reports,
            semantic_class=sem_class,
            evidence_role=ev_role,
        )
