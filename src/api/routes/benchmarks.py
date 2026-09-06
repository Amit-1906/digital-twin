"""Battery Benchmarking and Experimental Evidence API Routes."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_benchmark_service
from src.api.schemas.benchmarking import (
    BenchmarkDatasetSummaryDTO,
    BenchmarkResetResponseDTO,
    BenchmarkResultResponseDTO,
    BenchmarkRunRequestDTO,
    DetectionLatencySummaryDTO,
    DiagnosticEvaluationReportDTO,
    EventDiagnosticEvaluationDTO,
    SampleConfusionMatrixDTO,
    SignalMetricSummaryDTO,
)
from src.benchmarking.alignment import AlignmentMode, TemporalAlignmentPolicy
from src.benchmarking.engine import BenchmarkExecutionConfig, BenchmarkExecutionResult
from src.services.benchmark_service import BenchmarkService

router = APIRouter(prefix="/api/v1/benchmarks", tags=["Battery Benchmarking & Validation"])


def _serialize_benchmark_result(res: BenchmarkExecutionResult, default_system_id: str = "") -> BenchmarkResultResponseDTO:
    """Helper transforming a BenchmarkExecutionResult domain object into a comprehensive response DTO."""
    signals_map: dict[str, SignalMetricSummaryDTO] = {}
    if res.signal_metrics is not None:
        for name, sig in res.signal_metrics.overall_metrics.items():
            signals_map[name] = SignalMetricSummaryDTO(
                signal_name=sig.signal_name,
                unit=sig.unit,
                valid_paired_samples=sig.valid_paired_samples,
                rmse=sig.rmse.value,
                mae=sig.mae.value,
                max_error=sig.max_absolute_error.value,
                mean_bias_error=sig.mean_bias_error.value,
                r_squared=sig.r_squared.value,
                nrmse=sig.nrmse.value if sig.nrmse else None,
                coverage_fraction=sig.valid_pair_coverage,
            )

    diag_dto: Optional[DiagnosticEvaluationReportDTO] = None
    if res.diagnostic_metrics is not None:
        dm = res.diagnostic_metrics
        matrix_dto = SampleConfusionMatrixDTO(
            true_positives=dm.sample_confusion_matrix.true_positives,
            false_positives=dm.sample_confusion_matrix.false_positives,
            true_negatives=dm.sample_confusion_matrix.true_negatives,
            false_negatives=dm.sample_confusion_matrix.false_negatives,
            unknown_samples=dm.sample_confusion_matrix.unknown_samples,
            total_samples=dm.sample_confusion_matrix.total_samples,
            precision=dm.sample_confusion_matrix.precision,
            recall=dm.sample_confusion_matrix.recall,
            f1_score=dm.sample_confusion_matrix.f1_score,
        )
        lat_dto = DetectionLatencySummaryDTO(
            total_events=dm.latency_summary.total_events,
            detected_events=dm.latency_summary.detected_events,
            missed_events=dm.latency_summary.missed_events,
            early_detections=dm.latency_summary.early_detections,
            min_latency_s=dm.latency_summary.min_latency_s,
            max_latency_s=dm.latency_summary.max_latency_s,
            mean_latency_s=dm.latency_summary.mean_latency_s,
        )
        event_evals = [
            EventDiagnosticEvaluationDTO(
                event_id=e.event_id,
                event_type=e.event_type,
                expected_hypothesis_id=e.expected_hypothesis_id,
                start_timestamp_ns=e.start_timestamp_ns,
                end_timestamp_ns=e.end_timestamp_ns,
                match_status=e.match_status.value,
                is_detected=e.is_detected,
                detection_timestamp_ns=e.detection_timestamp_ns,
                latency_seconds=e.latency_seconds,
                diagnosed_hypothesis_id=e.diagnosed_hypothesis_id,
                is_hypothesis_matched=e.is_hypothesis_matched,
                lifecycle_state_at_detection=e.lifecycle_state_at_detection,
                severity_at_detection=e.severity_at_detection,
                corroborating_channels=list(e.corroborating_channels),
                explanation_narrative=e.explanation_narrative,
                metadata=dict(e.metadata),
            )
            for e in dm.event_evaluations
        ]
        diag_dto = DiagnosticEvaluationReportDTO(
            evaluated_reports_count=dm.evaluated_reports_count,
            semantic_class=dm.semantic_class.value,
            evidence_role=dm.evidence_role.value,
            sample_confusion_matrix=matrix_dto,
            latency_summary=lat_dto,
            event_evaluations=event_evals,
            metadata=dict(dm.metadata),
        )

    gating = "PASS" if res.is_passing else "FAIL"
    if res.total_frames_evaluated == 0:
        gating = "NOT_EVALUABLE"

    sys_id = res.metadata.get("system_id", default_system_id)

    return BenchmarkResultResponseDTO(
        benchmark_id=res.benchmark_id,
        system_id=sys_id,
        dataset_id=res.dataset_id,
        dataset_name=res.dataset_name,
        executed_at_utc=res.executed_at_utc,
        duration_seconds=res.duration_seconds,
        total_frames_evaluated=res.total_frames_evaluated,
        is_passing=res.is_passing,
        gating_status=gating,
        alignment_policy=res.alignment_policy.to_dict(),
        stratification_config=res.stratification_config.to_dict(),
        signals=signals_map,
        diagnostics=diag_dto,
        metadata=dict(res.metadata),
    )


@router.post(
    "/{system_id}/run",
    response_model=BenchmarkResultResponseDTO,
    summary="Execute Benchmark Evaluation Run",
)
async def run_benchmark(
    system_id: str,
    payload: BenchmarkRunRequestDTO,
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResultResponseDTO:
    """Executes an isolated co-simulation benchmark evaluation against a registered ExperimentalDataset."""
    align_mode = AlignmentMode.CAUSAL_STREAMING
    if payload.alignment_mode:
        try:
            align_mode = AlignmentMode(payload.alignment_mode.upper())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid alignment_mode '{payload.alignment_mode}'. Valid modes: {[m.value for m in AlignmentMode]}.",
            )

    policy = TemporalAlignmentPolicy(alignment_mode=align_mode)
    cfg = BenchmarkExecutionConfig(
        system_id=system_id,
        alignment_policy=policy,
        evaluate_signals=payload.evaluate_signals,
        evaluate_diagnostics=payload.evaluate_diagnostics,
        target_voltage_rmse_v=payload.target_voltage_rmse_v,
        target_temp_rmse_c=payload.target_temp_rmse_c,
        target_soc_rmse=payload.target_soc_rmse,
        min_diagnostic_precision=payload.min_diagnostic_precision,
        min_diagnostic_recall=payload.min_diagnostic_recall,
    )

    result = benchmark_service.run_benchmark(
        system_id=system_id,
        dataset_id=payload.dataset_id,
        config=cfg,
    )
    return _serialize_benchmark_result(result, default_system_id=system_id)


@router.get(
    "/datasets",
    response_model=list[BenchmarkDatasetSummaryDTO],
    summary="List Registered Reference Datasets",
)
async def list_datasets(
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> list[BenchmarkDatasetSummaryDTO]:
    """Retrieves metadata summaries for all registered reference and experimental datasets."""
    datasets = benchmark_service.list_datasets()
    return [BenchmarkDatasetSummaryDTO(**ds) for ds in datasets]


@router.get(
    "/datasets/{dataset_id}",
    response_model=BenchmarkDatasetSummaryDTO,
    summary="Get Dataset Metadata Summary",
)
async def get_dataset(
    dataset_id: str,
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkDatasetSummaryDTO:
    """Retrieves metadata summary for a specific registered reference dataset."""
    ds = benchmark_service.get_dataset(dataset_id)
    ch = ds.reference_channels[0] if ds.reference_channels else None
    ev_role = ch.provenance.evidence_role.value if (ch and ch.provenance) else None
    sem_class = "PHYSICAL_REFERENCE_VALIDATION" if (ch and ch.provenance and ch.provenance.evidence_role.value == "PHYSICAL_REFERENCE") else "ALGORITHMIC_SYNTHETIC_VERIFICATION"

    return BenchmarkDatasetSummaryDTO(
        dataset_id=ds.dataset_id,
        dataset_name=ds.metadata.dataset_name,
        version=ds.metadata.version,
        total_frames=len(ds.frames),
        start_timestamp_ns=ds.start_timestamp_ns,
        end_timestamp_ns=ds.end_timestamp_ns,
        duration_seconds=ds.duration_seconds,
        reference_channels=[c.signal_name for c in ds.reference_channels],
        ground_truth_events_count=len(ds.ground_truth_events),
        semantic_class=sem_class,
        evidence_role=ev_role,
        description=ds.metadata.description,
    )


@router.get(
    "/{system_id}/latest",
    response_model=BenchmarkResultResponseDTO,
    summary="Get Latest Benchmark Execution Result for System",
)
async def get_latest_benchmark(
    system_id: str,
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResultResponseDTO:
    """Retrieves the most recent benchmark evaluation result for a digital twin system."""
    result = benchmark_service.get_latest_result(system_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No benchmark results found for system '{system_id}'.",
        )
    return _serialize_benchmark_result(result, default_system_id=system_id)


@router.get(
    "/{system_id}/history",
    response_model=list[BenchmarkResultResponseDTO],
    summary="Get Bounded Benchmark History for System",
)
async def get_system_benchmark_history(
    system_id: str,
    limit: Optional[int] = Query(50, ge=1, le=100),
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> list[BenchmarkResultResponseDTO]:
    """Retrieves bounded historical benchmark results for a specific digital twin system in reverse chronological order."""
    results = benchmark_service.get_history(system_id=system_id, limit=limit)
    return [_serialize_benchmark_result(r, default_system_id=system_id) for r in results]


@router.get(
    "/results/{benchmark_id}",
    response_model=BenchmarkResultResponseDTO,
    summary="Get Benchmark Result by ID",
)
async def get_benchmark_result_by_id(
    benchmark_id: str,
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResultResponseDTO:
    """Retrieves a specific historical benchmark evaluation result by its unique benchmark_id."""
    result = benchmark_service.get_result(benchmark_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark result '{benchmark_id}' not found.",
        )
    return _serialize_benchmark_result(result)


@router.get(
    "/results",
    response_model=list[BenchmarkResultResponseDTO],
    summary="List All Historical Benchmark Results",
)
async def list_all_benchmark_results(
    limit: Optional[int] = Query(50, ge=1, le=100),
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> list[BenchmarkResultResponseDTO]:
    """Retrieves global bounded historical benchmark results across all systems."""
    results = benchmark_service.get_history(system_id=None, limit=limit)
    return [_serialize_benchmark_result(r) for r in results]


@router.post(
    "/{system_id}/reset",
    response_model=BenchmarkResetResponseDTO,
    summary="Reset Benchmark History for System",
)
async def reset_benchmark_history(
    system_id: str,
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResetResponseDTO:
    """Resets benchmark result history for a system without mutating the live digital twin."""
    benchmark_service.reset(system_id=system_id)
    return BenchmarkResetResponseDTO(
        success=True,
        system_id=system_id,
        message=f"Benchmark results history cleared for system '{system_id}'. Live digital twin state remains unaffected.",
    )
