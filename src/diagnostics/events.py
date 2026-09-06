"""Domain events for Battery Diagnostics and Explainable Intelligence."""

from dataclasses import dataclass
from typing import Optional

from src.diagnostics.types import DiagnosticAssessmentReport
from src.events.base import TwinEvent


@dataclass(frozen=True)
class BatteryDiagnosticAssessedEvent(TwinEvent):
    """Published when the diagnostic engine evaluates a step or transitions lifecycle state."""

    diagnostic_report: Optional[DiagnosticAssessmentReport] = None
    event_type: str = "twin.diagnostic"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.diagnostic_report is not None:
            if self.source_id == "system":
                object.__setattr__(self, "source_id", self.diagnostic_report.system_id)
            object.__setattr__(self, "timestamp_ns", self.diagnostic_report.timestamp_ns)
            if not self.payload:
                object.__setattr__(self, "payload", self.diagnostic_report.to_dict())


@dataclass(frozen=True)
class CriticalAdvisoryAlertEvent(TwinEvent):
    """Published when the diagnostic engine evaluates a DIAGNOSED_CRITICAL advisory condition."""

    diagnostic_report: Optional[DiagnosticAssessmentReport] = None
    event_type: str = "twin.diagnostic.critical_advisory"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.diagnostic_report is not None:
            if self.source_id == "system":
                object.__setattr__(self, "source_id", self.diagnostic_report.system_id)
            object.__setattr__(self, "timestamp_ns", self.diagnostic_report.timestamp_ns)
            if not self.payload:
                object.__setattr__(self, "payload", self.diagnostic_report.to_dict())
