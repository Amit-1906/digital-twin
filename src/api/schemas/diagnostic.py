"""Pydantic schemas for Battery Diagnostics REST and WebSocket payloads."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class DiagnosticEvidenceItemDTO(BaseModel):
    """Transport schema for individual diagnostic evidence item."""

    source_layer: str
    signal_name: str
    observed_value: Any
    expected_value: Any = None
    provenance: str = "MEASURED"
    status: str = "NO_EVIDENCE"
    weight: float = 1.0
    rationale: str = ""


class RootCauseHypothesisDTO(BaseModel):
    """Transport schema for diagnostic root-cause hypothesis evaluation."""

    hypothesis_id: str
    title: str
    category: str
    evidence_score: float
    confidence_level: str
    evidence_strength_tier: str
    required_signal_coverage: float
    optional_signal_coverage: float
    is_diagnostically_eligible: bool = True
    is_critical_eligible: bool = False
    supporting_evidence: list[DiagnosticEvidenceItemDTO] = Field(default_factory=list)
    contraindicating_evidence: list[DiagnosticEvidenceItemDTO] = Field(default_factory=list)
    untestable_confounds: list[str] = Field(default_factory=list)
    suggested_investigations: list[str] = Field(default_factory=list)


class DiagnosticAssessmentResponseDTO(BaseModel):
    """Transport schema for comprehensive diagnostic assessment report."""

    assessment_id: str
    system_id: str
    timestamp_ns: int
    lifecycle_state: str
    severity: str
    operating_context: str
    primary_hypothesis: Optional[RootCauseHypothesisDTO] = None
    alternative_hypotheses: list[RootCauseHypothesisDTO] = Field(default_factory=list)
    corroborating_signals: list[str] = Field(default_factory=list)
    explanation_narrative: str
    recommended_operator_actions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    active_anomalies_count: int = 0
    data_quality_status: str = "VALID"


class DiagnosticResetResponseDTO(BaseModel):
    """Transport schema for diagnostic subsystem reset confirmation."""

    system_id: str
    status: str = "RESET_SUCCESSFUL"
    subsystem: str = "DIAGNOSTICS"
    timestamp_ns: int
