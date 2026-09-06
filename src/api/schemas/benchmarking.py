"""Pydantic Data Transfer Objects (DTOs) for Battery Benchmarking REST API."""

from typing import Any, Mapping, Optional
from pydantic import BaseModel, Field


class BenchmarkRunRequestDTO(BaseModel):
    """Payload for initiating an isolated benchmark evaluation run against a reference dataset."""

    dataset_id: str = Field(..., description="Identifier of the registered ExperimentalDataset to evaluate against.")
    alignment_mode: Optional[str] = Field(None, description="Optional temporal alignment mode (e.g. CAUSAL_STREAMING, NON_CAUSAL_BATCH).")
    evaluate_signals: bool = Field(True, description="Whether to compute quantitative signal tracking metrics (RMSE, MAE, R^2).")
    evaluate_diagnostics: bool = Field(True, description="Whether to evaluate diagnostic reports against ground truth events.")
    target_voltage_rmse_v: Optional[float] = Field(None, ge=0.0, description="Optional maximum acceptable Voltage RMSE threshold for gating.")
    target_temp_rmse_c: Optional[float] = Field(None, ge=0.0, description="Optional maximum acceptable Temperature RMSE threshold for gating.")
    target_soc_rmse: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional maximum acceptable SOC RMSE threshold for gating.")
    min_diagnostic_precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional minimum acceptable Precision threshold for gating.")
    min_diagnostic_recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional minimum acceptable Recall threshold for gating.")


class BenchmarkDatasetSummaryDTO(BaseModel):
    """Summary representation of a registered experimental reference dataset."""

    dataset_id: str
    dataset_name: str
    version: str
    total_frames: int
    start_timestamp_ns: Optional[int] = None
    end_timestamp_ns: Optional[int] = None
    duration_seconds: Optional[float] = None
    reference_channels: list[str] = Field(default_factory=list)
    ground_truth_events_count: int = 0
    semantic_class: Optional[str] = None
    evidence_role: Optional[str] = None
    description: str = ""


class SignalMetricSummaryDTO(BaseModel):
    """Quantitative metrological tracking metrics for an individual reference signal channel."""

    signal_name: str
    unit: str
    valid_paired_samples: int
    rmse: Optional[float] = None
    mae: Optional[float] = None
    max_error: Optional[float] = None
    mean_bias_error: Optional[float] = None
    r_squared: Optional[float] = None
    nrmse: Optional[float] = None
    coverage_fraction: float = 1.0


class SampleConfusionMatrixDTO(BaseModel):
    """Sample-level diagnostic classification confusion counts and rates."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    unknown_samples: int
    total_samples: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class DetectionLatencySummaryDTO(BaseModel):
    """Statistical summary of fault detection latencies across evaluated events."""

    total_events: int
    detected_events: int
    missed_events: int
    early_detections: int
    min_latency_s: Optional[float] = None
    max_latency_s: Optional[float] = None
    mean_latency_s: Optional[float] = None


class EventDiagnosticEvaluationDTO(BaseModel):
    """Evaluation result for an individual ground-truth event."""

    event_id: str
    event_type: str
    expected_hypothesis_id: Optional[str] = None
    start_timestamp_ns: int
    end_timestamp_ns: Optional[int] = None
    match_status: str
    is_detected: bool
    detection_timestamp_ns: Optional[int] = None
    latency_seconds: Optional[float] = None
    diagnosed_hypothesis_id: Optional[str] = None
    is_hypothesis_matched: bool = False
    lifecycle_state_at_detection: Optional[str] = None
    severity_at_detection: Optional[str] = None
    corroborating_channels: list[str] = Field(default_factory=list)
    explanation_narrative: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class DiagnosticEvaluationReportDTO(BaseModel):
    """Comprehensive diagnostic evaluation report against experimental ground truth."""

    evaluated_reports_count: int
    semantic_class: str
    evidence_role: str
    sample_confusion_matrix: SampleConfusionMatrixDTO
    latency_summary: DetectionLatencySummaryDTO
    event_evaluations: list[EventDiagnosticEvaluationDTO] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResultResponseDTO(BaseModel):
    """Comprehensive, immutable benchmark evaluation result DTO preserving complete scientific metadata."""

    benchmark_id: str
    system_id: str
    dataset_id: str
    dataset_name: str
    executed_at_utc: str
    duration_seconds: float
    total_frames_evaluated: int
    is_passing: bool
    gating_status: str = "PASS"  # PASS, FAIL, NOT_EVALUABLE
    alignment_policy: dict[str, Any] = Field(default_factory=dict)
    stratification_config: dict[str, Any] = Field(default_factory=dict)
    signals: dict[str, SignalMetricSummaryDTO] = Field(default_factory=dict)
    diagnostics: Optional[DiagnosticEvaluationReportDTO] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResetResponseDTO(BaseModel):
    """Response DTO for benchmark history reset operations."""

    success: bool = True
    system_id: Optional[str] = None
    message: str
