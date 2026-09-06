"""Benchmark Execution Engine for Battery Benchmarking.

Orchestrates automated dataset replay, temporal alignment, multi-dimensional metric computation,
diagnostic ground-truth evaluation, and deterministic reporting inside isolated sandboxes.
"""

from dataclasses import dataclass, field
import datetime
import time
from typing import Any, Mapping, Optional, Sequence, Tuple
import uuid

from src.benchmarking.alignment import (
    AlignedFramePair,
    TemporalAlignmentEngine,
    TemporalAlignmentPolicy,
)
from src.benchmarking.dataset import (
    ExperimentalDataset,
    ReferenceObservationFrame,
)
from src.benchmarking.diagnostic_evaluator import (
    DiagnosticEvaluationReport,
    DiagnosticGroundTruthEvaluator,
)
from src.benchmarking.events import (
    BenchmarkExecutionCompletedEvent,
    BenchmarkExecutionStartedEvent,
)
from src.benchmarking.metrics import (
    BenchmarkMetricsCalculator,
    StratifiedMetricReport,
)
from src.benchmarking.stratification import (
    OperatingContextStratifier,
    StratificationConfig,
)
from src.benchmarking.types import (
    BenchmarkSemanticClass,
    ReferenceProvenance,
)
from src.diagnostics.types import DiagnosticAssessmentReport
from src.runtime.instance import DigitalTwinInstance
from src.runtime.synchronizer import TwinSyncOutput
from src.telemetry.snapshots import TelemetrySnapshot


@dataclass(frozen=True)
class BenchmarkExecutionConfig:
    """Configuration options governing an automated benchmark evaluation run."""

    system_id: Optional[str] = None
    alignment_policy: Optional[TemporalAlignmentPolicy] = None
    stratification_config: Optional[StratificationConfig] = None
    evaluate_diagnostics: bool = True
    evaluate_signals: bool = True
    signal_units: Mapping[str, str] = field(
        default_factory=lambda: {
            "voltage_v": "V",
            "pack_voltage_v": "V",
            "temperature_c": "degC",
            "avg_cell_temperature_c": "degC",
            "current_a": "A",
            "pack_current_a": "A",
            "soc_fraction": "fraction",
            "soh_fraction": "fraction",
        }
    )
    target_voltage_rmse_v: Optional[float] = None
    target_temp_rmse_c: Optional[float] = None
    target_soc_rmse: Optional[float] = None
    min_diagnostic_precision: Optional[float] = None
    min_diagnostic_recall: Optional[float] = None
    max_batch_samples: int = 100_000
    auto_reset_instance: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serializes benchmark execution config to dictionary."""
        return {
            "system_id": self.system_id,
            "alignment_policy": self.alignment_policy.to_dict() if self.alignment_policy else None,
            "stratification_config": self.stratification_config.to_dict() if self.stratification_config else None,
            "evaluate_diagnostics": self.evaluate_diagnostics,
            "evaluate_signals": self.evaluate_signals,
            "signal_units": dict(self.signal_units),
            "target_voltage_rmse_v": self.target_voltage_rmse_v,
            "target_temp_rmse_c": self.target_temp_rmse_c,
            "target_soc_rmse": self.target_soc_rmse,
            "min_diagnostic_precision": self.min_diagnostic_precision,
            "min_diagnostic_recall": self.min_diagnostic_recall,
            "max_batch_samples": self.max_batch_samples,
            "auto_reset_instance": self.auto_reset_instance,
        }


@dataclass(frozen=True)
class BenchmarkExecutionResult:
    """Comprehensive, immutable benchmark evaluation result container."""

    benchmark_id: str
    dataset_id: str
    dataset_name: str
    executed_at_utc: str
    duration_seconds: float
    total_frames_evaluated: int
    alignment_policy: TemporalAlignmentPolicy
    stratification_config: StratificationConfig
    signal_metrics: Optional[StratifiedMetricReport] = None
    diagnostic_metrics: Optional[DiagnosticEvaluationReport] = None
    is_passing: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes benchmark execution result to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "executed_at_utc": self.executed_at_utc,
            "duration_seconds": round(self.duration_seconds, 4),
            "total_frames_evaluated": self.total_frames_evaluated,
            "alignment_policy": self.alignment_policy.to_dict(),
            "stratification_config": self.stratification_config.to_dict(),
            "signal_metrics": self.signal_metrics.to_dict() if self.signal_metrics else None,
            "diagnostic_metrics": self.diagnostic_metrics.to_dict() if self.diagnostic_metrics else None,
            "is_passing": self.is_passing,
            "metadata": dict(self.metadata),
        }


class BenchmarkExecutionEngine:
    """Deterministic Benchmark Execution Engine.

    Executes isolated co-simulation replay against an ExperimentalDataset, aligning signals,
    calculating metrics, and evaluating diagnostic performance with zero live twin mutation.
    """

    @classmethod
    def run_benchmark(
        cls,
        instance: DigitalTwinInstance,
        dataset: ExperimentalDataset,
        config: Optional[BenchmarkExecutionConfig] = None,
        event_bus: Optional[Any] = None,
    ) -> BenchmarkExecutionResult:
        """Executes a complete isolated benchmark evaluation run.

        Args:
            instance: DigitalTwinInstance to execute benchmark against.
            dataset: Immutable ExperimentalDataset containing reference frames.
            config: Optional benchmark configuration.
            event_bus: Optional DigitalTwinEventBus for publishing lifecycle events.

        Returns:
            BenchmarkExecutionResult value object.
        """
        if not isinstance(instance, DigitalTwinInstance):
            raise TypeError(f"Expected DigitalTwinInstance, got {type(instance).__name__}.")
        if not isinstance(dataset, ExperimentalDataset):
            raise TypeError(f"Expected ExperimentalDataset, got {type(dataset).__name__}.")

        cfg = config or BenchmarkExecutionConfig()
        total_frames = len(dataset.frames)

        if total_frames > cfg.max_batch_samples:
            raise ValueError(
                f"Dataset frame count ({total_frames}) exceeds configured max_batch_samples ({cfg.max_batch_samples})."
            )

        benchmark_id = f"bench_{dataset.dataset_id}_{uuid.uuid4().hex[:8]}"
        t_start_perf = time.perf_counter()
        t_start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Publish start event
        if event_bus is not None and hasattr(event_bus, "publish"):
            event_bus.publish(
                BenchmarkExecutionStartedEvent(
                    benchmark_id=benchmark_id,
                    dataset_id=dataset.dataset_id,
                    dataset_name=dataset.metadata.dataset_name,
                    total_frames=total_frames,
                    source_id=instance.system_id,
                    timestamp_ns=dataset.start_timestamp_ns or 0,
                )
            )

        # 2. Replay Sandbox Setup
        if cfg.auto_reset_instance:
            instance.reset()
            # If dataset carries initial frame, initialize twin from initial frame
            if total_frames > 0:
                first_frame = dataset.frames[0]
                init_v = first_frame.get_reference_value("voltage_v") or first_frame.get_field_value("pack_voltage_v")
                init_t = first_frame.get_reference_value("temperature_c") or first_frame.get_field_value("avg_cell_temperature_c") or 25.0
                init_soc = first_frame.get_reference_value("soc_fraction") or first_frame.get_bms_estimate("bms_soc") or 1.0
                instance.initialize(
                    initial_soc=init_soc,
                    temperature_c=init_t,
                )

        # 3. Stepping loop
        eval_frames: list[Tuple[int, Mapping[str, float]]] = []
        diagnostic_reports: list[DiagnosticAssessmentReport] = []

        for idx, frame in enumerate(dataset.frames):
            ts = frame.timestamp_ns
            # Build input telemetry snapshot for twin step
            curr_val = frame.field_signals.get("pack_current_a", 0.0)
            amb_temp = frame.field_signals.get("ambient_temperature_c", 25.0)

            # Extract voltage observation if present in field signals
            v_field = frame.field_signals.get("pack_voltage_v") or frame.field_signals.get("voltage_v")
            t_field = frame.field_signals.get("avg_cell_temperature_c") or frame.field_signals.get("temperature_c")
            soc_field = frame.bms_estimates.get("bms_soc") or frame.field_signals.get("soc_fraction")

            snap = TelemetrySnapshot(
                snapshot_id=f"bench_snap_{idx:06d}",
                system_id=instance.system_id,
                timestamp_ns=ts,
                pack_current_a=curr_val,
                pack_voltage_v=v_field,
                avg_cell_temperature_c=t_field,
                ambient_temperature_c=amb_temp,
                soc_fraction=soc_field,
            )

            sync_out = instance.step(snap)

            # Record evaluated signals
            eval_sigs: dict[str, float] = {}
            if sync_out.model_output:
                eval_sigs["voltage_v"] = sync_out.model_output.terminal_voltage_v
                eval_sigs["pack_voltage_v"] = sync_out.model_output.terminal_voltage_v
                eval_sigs["temperature_c"] = sync_out.model_output.state.temperature_c
                eval_sigs["avg_cell_temperature_c"] = sync_out.model_output.state.temperature_c

            if sync_out.estimation_output:
                eval_sigs["soc_fraction"] = sync_out.estimation_output.state.soc_fraction
            elif sync_out.model_output:
                eval_sigs["soc_fraction"] = sync_out.model_output.state.soc_fraction

            eval_sigs["pack_current_a"] = curr_val
            eval_frames.append((ts, eval_sigs))

            if instance.latest_diagnostic_report is not None:
                diagnostic_reports.append(instance.latest_diagnostic_report)

        duration_sec = time.perf_counter() - t_start_perf

        # 4. Temporal Alignment & Quantitative Signal Metrics
        align_policy = cfg.alignment_policy or TemporalAlignmentPolicy()
        strat_cfg = cfg.stratification_config or OperatingContextStratifier.get_default_config()

        signal_report: Optional[StratifiedMetricReport] = None
        is_passing = True

        if cfg.evaluate_signals and eval_frames:
            frame_pairs = TemporalAlignmentEngine.align_frames(
                eval_frames=eval_frames,
                reference_dataset=dataset,
                policy=align_policy,
            )
            signal_report = BenchmarkMetricsCalculator.compute_stratified_metrics(
                frame_pairs=frame_pairs,
                signal_units=cfg.signal_units,
                stratification_config=strat_cfg,
                alignment_policy=align_policy,
            )

            # Check threshold pass/fail
            v_sum = signal_report.get_overall("voltage_v") or signal_report.get_overall("pack_voltage_v")
            if v_sum and cfg.target_voltage_rmse_v is not None and v_sum.rmse.value is not None:
                if v_sum.rmse.value > cfg.target_voltage_rmse_v:
                    is_passing = False

            t_sum = signal_report.get_overall("temperature_c") or signal_report.get_overall("avg_cell_temperature_c")
            if t_sum and cfg.target_temp_rmse_c is not None and t_sum.rmse.value is not None:
                if t_sum.rmse.value > cfg.target_temp_rmse_c:
                    is_passing = False

            s_sum = signal_report.get_overall("soc_fraction")
            if s_sum and cfg.target_soc_rmse is not None and s_sum.rmse.value is not None:
                if s_sum.rmse.value > cfg.target_soc_rmse:
                    is_passing = False

        # 5. Diagnostic Ground-Truth Evaluation
        diag_report: Optional[DiagnosticEvaluationReport] = None
        if cfg.evaluate_diagnostics and dataset.ground_truth_events:
            diag_report = DiagnosticGroundTruthEvaluator.evaluate(
                diagnostic_reports=diagnostic_reports,
                ground_truth_events=dataset.ground_truth_events,
            )
            # Check diagnostic thresholds
            if cfg.min_diagnostic_precision is not None:
                prec = diag_report.sample_confusion_matrix.precision
                if prec is None or prec < cfg.min_diagnostic_precision:
                    is_passing = False

            if cfg.min_diagnostic_recall is not None:
                rec = diag_report.sample_confusion_matrix.recall
                if rec is None or rec < cfg.min_diagnostic_recall:
                    is_passing = False

        # 6. Publish complete event
        key_metrics: dict[str, Any] = {}
        if signal_report:
            for s_name in ("voltage_v", "pack_voltage_v", "temperature_c", "soc_fraction"):
                s_sum = signal_report.get_overall(s_name)
                if s_sum and s_sum.rmse.value is not None:
                    key_metrics[f"{s_name}_rmse"] = s_sum.rmse.value

        if event_bus is not None and hasattr(event_bus, "publish"):
            event_bus.publish(
                BenchmarkExecutionCompletedEvent(
                    benchmark_id=benchmark_id,
                    dataset_id=dataset.dataset_id,
                    duration_seconds=duration_sec,
                    total_frames=total_frames,
                    is_passing=is_passing,
                    key_metrics=key_metrics,
                    source_id=instance.system_id,
                    timestamp_ns=dataset.end_timestamp_ns or 0,
                )
            )

        return BenchmarkExecutionResult(
            benchmark_id=benchmark_id,
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.metadata.dataset_name,
            executed_at_utc=t_start_utc,
            duration_seconds=duration_sec,
            total_frames_evaluated=total_frames,
            alignment_policy=align_policy,
            stratification_config=strat_cfg,
            signal_metrics=signal_report,
            diagnostic_metrics=diag_report,
            is_passing=is_passing,
            metadata={
                "system_id": instance.system_id,
                "dataset_version": dataset.metadata.version,
                "ground_truth_events_count": len(dataset.ground_truth_events),
            },
        )
