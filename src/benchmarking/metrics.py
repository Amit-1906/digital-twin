"""Quantitative Metric Engine and Statistical Error Accumulators for Battery Benchmarking.

Calculates evidence-qualified, uncertainty-aware statistical tracking metrics (MAE, RMSE, MBE,
Max Error, R^2, NRMSE, Uncertainty-Consistent Fraction) using bounded O(1) streaming accumulators.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from src.benchmarking.alignment import (
    AlignedFramePair,
    AlignedSignalPair,
    TemporalAlignmentPolicy,
)
from src.benchmarking.stratification import (
    ContextDimension,
    ContextSliceIdentifier,
    OperatingContextStratifier,
    StratificationConfig,
)
from src.benchmarking.types import (
    BenchmarkSemanticClass,
    EvidenceRole,
    ReferenceProvenance,
)


@dataclass(frozen=True)
class SingleMetricValue:
    """Strongly-typed container for an individual statistical metric with semantic validity status."""

    metric_name: str
    value: Optional[float]
    unit: str
    status: str = "VALID"                           # "VALID", "INSUFFICIENT_SAMPLES", "UNDEFINED_ZERO_REFERENCE_VARIANCE", "NO_PAIRED_SAMPLES"
    diagnostic_message: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.metric_name or not isinstance(self.metric_name, str):
            raise ValueError("metric_name must be a non-empty string.")
        if self.value is not None:
            if not isinstance(self.value, (int, float)) or math.isnan(self.value) or math.isinf(self.value):
                raise ValueError(f"Metric value for '{self.metric_name}' must be a finite float, got {self.value}.")

    def to_dict(self) -> dict[str, Any]:
        """Serializes single metric value to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": round(self.value, 6) if self.value is not None else None,
            "unit": self.unit,
            "status": self.status,
            "diagnostic_message": self.diagnostic_message,
        }


@dataclass(frozen=True)
class SignalMetricSummary:
    """Comprehensive statistical accuracy and error summary for a single benchmarked signal."""

    signal_name: str
    unit: str
    semantic_class: BenchmarkSemanticClass
    evidence_role: EvidenceRole
    provenance: Optional[ReferenceProvenance] = None
    total_eval_samples: int = 0
    valid_paired_samples: int = 0
    excluded_samples: int = 0
    exclusion_reasons: Mapping[str, int] = field(default_factory=dict)
    valid_pair_coverage: float = 0.0
    mae: SingleMetricValue = field(default_factory=lambda: SingleMetricValue("MAE", None, "", status="NO_PAIRED_SAMPLES"))
    rmse: SingleMetricValue = field(default_factory=lambda: SingleMetricValue("RMSE", None, "", status="NO_PAIRED_SAMPLES"))
    mean_bias_error: SingleMetricValue = field(default_factory=lambda: SingleMetricValue("MEAN_BIAS_ERROR", None, "", status="NO_PAIRED_SAMPLES"))
    max_absolute_error: SingleMetricValue = field(default_factory=lambda: SingleMetricValue("MAX_ABSOLUTE_ERROR", None, "", status="NO_PAIRED_SAMPLES"))
    r_squared: SingleMetricValue = field(default_factory=lambda: SingleMetricValue("R_SQUARED", None, "", status="NO_PAIRED_SAMPLES"))
    nrmse: Optional[SingleMetricValue] = None
    mean_reference_value: Optional[float] = None
    mean_eval_value: Optional[float] = None
    reference_min: Optional[float] = None
    reference_max: Optional[float] = None
    mean_reference_uncertainty: float = 0.0
    uncertainty_consistent_sample_count: int = 0
    uncertainty_consistent_fraction: float = 0.0
    context_slice: Optional[ContextSliceIdentifier] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes signal metric summary to dictionary."""
        return {
            "signal_name": self.signal_name,
            "unit": self.unit,
            "semantic_class": self.semantic_class.value,
            "evidence_role": self.evidence_role.value,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "total_eval_samples": self.total_eval_samples,
            "valid_paired_samples": self.valid_paired_samples,
            "excluded_samples": self.excluded_samples,
            "exclusion_reasons": dict(self.exclusion_reasons),
            "valid_pair_coverage": round(self.valid_pair_coverage, 4),
            "mae": self.mae.to_dict(),
            "rmse": self.rmse.to_dict(),
            "mean_bias_error": self.mean_bias_error.to_dict(),
            "max_absolute_error": self.max_absolute_error.to_dict(),
            "r_squared": self.r_squared.to_dict(),
            "nrmse": self.nrmse.to_dict() if self.nrmse else None,
            "mean_reference_value": round(self.mean_reference_value, 6) if self.mean_reference_value is not None else None,
            "mean_eval_value": round(self.mean_eval_value, 6) if self.mean_eval_value is not None else None,
            "reference_min": round(self.reference_min, 6) if self.reference_min is not None else None,
            "reference_max": round(self.reference_max, 6) if self.reference_max is not None else None,
            "mean_reference_uncertainty": round(self.mean_reference_uncertainty, 6),
            "uncertainty_consistent_sample_count": self.uncertainty_consistent_sample_count,
            "uncertainty_consistent_fraction": round(self.uncertainty_consistent_fraction, 4),
            "context_slice": self.context_slice.to_dict() if self.context_slice else None,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class StratifiedMetricReport:
    """Overall multi-signal and context-stratified benchmark evaluation report."""

    overall_metrics: Mapping[str, SignalMetricSummary]
    stratified_metrics: Mapping[str, Mapping[str, SignalMetricSummary]] = field(default_factory=dict)
    total_samples_evaluated: int = 0
    overall_valid_pairs: int = 0
    alignment_policy: Optional[TemporalAlignmentPolicy] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get_overall(self, signal_name: str) -> Optional[SignalMetricSummary]:
        """Retrieves overall signal metric summary."""
        return self.overall_metrics.get(signal_name)

    def get_stratified(self, slice_key: str, signal_name: str) -> Optional[SignalMetricSummary]:
        """Retrieves signal metric summary for a specific context slice."""
        slice_map = self.stratified_metrics.get(slice_key)
        return slice_map.get(signal_name) if slice_map else None

    def to_dict(self) -> dict[str, Any]:
        """Serializes stratified metric report to dictionary."""
        return {
            "total_samples_evaluated": self.total_samples_evaluated,
            "overall_valid_pairs": self.overall_valid_pairs,
            "alignment_policy": self.alignment_policy.to_dict() if self.alignment_policy else None,
            "overall_metrics": {k: v.to_dict() for k, v in self.overall_metrics.items()},
            "stratified_metrics": {
                slice_k: {sig_k: sig_v.to_dict() for sig_k, sig_v in sig_map.items()}
                for slice_k, sig_map in self.stratified_metrics.items()
            },
            "metadata": dict(self.metadata),
        }


class StreamingWelfordAccumulator:
    """Bounded O(1) memory streaming statistical accumulator for online accuracy calculations."""

    def __init__(self) -> None:
        self.count: int = 0
        self.sum_error: float = 0.0
        self.sum_abs_error: float = 0.0
        self.sum_sq_error: float = 0.0
        self.max_abs_error: float = 0.0
        self.ref_mean: float = 0.0
        self.ref_M2: float = 0.0
        self.eval_mean: float = 0.0
        self.eval_M2: float = 0.0
        self.ref_min: float = float("inf")
        self.ref_max: float = float("-inf")
        self.sum_uncertainty: float = 0.0
        self.uncertainty_consistent_count: int = 0

    def update(
        self,
        ref_val: float,
        eval_val: float,
        ref_uncertainty: float = 0.0,
    ) -> None:
        """Accumulates a single paired observation into the streaming state."""
        if math.isnan(ref_val) or math.isinf(ref_val) or math.isnan(eval_val) or math.isinf(eval_val):
            return

        self.count += 1
        err = float(ref_val - eval_val)
        abs_err = abs(err)
        sq_err = err * err

        self.sum_error += err
        self.sum_abs_error += abs_err
        self.sum_sq_error += sq_err
        if abs_err > self.max_abs_error:
            self.max_abs_error = abs_err

        # Welford's algorithm for Reference mean and M2 (sum of squared deviations)
        delta_ref = ref_val - self.ref_mean
        self.ref_mean += delta_ref / self.count
        delta2_ref = ref_val - self.ref_mean
        self.ref_M2 += delta_ref * delta2_ref

        # Welford's algorithm for Eval mean and M2
        delta_eval = eval_val - self.eval_mean
        self.eval_mean += delta_eval / self.count
        delta2_eval = eval_val - self.eval_mean
        self.eval_M2 += delta_eval * delta2_eval

        # Min/Max
        if ref_val < self.ref_min:
            self.ref_min = ref_val
        if ref_val > self.ref_max:
            self.ref_max = ref_val

        # Uncertainty
        eff_unc = max(0.0, ref_uncertainty)
        self.sum_uncertainty += eff_unc
        if abs_err <= eff_unc or (eff_unc == 0.0 and abs_err < 1e-12):
            self.uncertainty_consistent_count += 1

    def compute_metrics(
        self,
        unit: str,
        min_samples_for_r_squared: int = 2,
    ) -> dict[str, Any]:
        """Finalizes accumulated sums into strongly-typed metrics."""
        n = self.count
        if n == 0:
            return {
                "count": 0,
                "mae": SingleMetricValue("MAE", None, unit, status="NO_PAIRED_SAMPLES"),
                "rmse": SingleMetricValue("RMSE", None, unit, status="NO_PAIRED_SAMPLES"),
                "mbe": SingleMetricValue("MEAN_BIAS_ERROR", None, unit, status="NO_PAIRED_SAMPLES"),
                "max_error": SingleMetricValue("MAX_ABSOLUTE_ERROR", None, unit, status="NO_PAIRED_SAMPLES"),
                "r_squared": SingleMetricValue("R_SQUARED", None, "dimensionless", status="NO_PAIRED_SAMPLES"),
                "nrmse": None,
                "ref_mean": None,
                "eval_mean": None,
                "ref_min": None,
                "ref_max": None,
                "mean_uncertainty": 0.0,
                "ucf": 0.0,
            }

        mae_val = self.sum_abs_error / n
        rmse_val = math.sqrt(self.sum_sq_error / n)
        mbe_val = self.sum_error / n
        max_err_val = self.max_abs_error

        # R² calculation: R² = 1 - (SS_res / SS_tot)
        ss_res = self.sum_sq_error
        ss_tot = self.ref_M2

        if n < min_samples_for_r_squared:
            r2_metric = SingleMetricValue(
                "R_SQUARED",
                None,
                "dimensionless",
                status="INSUFFICIENT_SAMPLES",
                diagnostic_message=f"Sample count ({n}) < required minimum ({min_samples_for_r_squared}).",
            )
        elif ss_tot < 1e-12:
            r2_metric = SingleMetricValue(
                "R_SQUARED",
                None,
                "dimensionless",
                status="UNDEFINED_ZERO_REFERENCE_VARIANCE",
                diagnostic_message="Reference signal variance is zero (constant reference signal).",
            )
        else:
            r2_val = 1.0 - (ss_res / ss_tot)
            r2_metric = SingleMetricValue("R_SQUARED", r2_val, "dimensionless", status="VALID")

        # NRMSE (NRMSE = RMSE / Range)
        ref_range = self.ref_max - self.ref_min
        if ref_range > 1e-12:
            nrmse_metric = SingleMetricValue("NRMSE", rmse_val / ref_range, "dimensionless", status="VALID")
        else:
            nrmse_metric = None

        return {
            "count": n,
            "mae": SingleMetricValue("MAE", mae_val, unit, status="VALID"),
            "rmse": SingleMetricValue("RMSE", rmse_val, unit, status="VALID"),
            "mbe": SingleMetricValue("MEAN_BIAS_ERROR", mbe_val, unit, status="VALID"),
            "max_error": SingleMetricValue("MAX_ABSOLUTE_ERROR", max_err_val, unit, status="VALID"),
            "r_squared": r2_metric,
            "nrmse": nrmse_metric,
            "ref_mean": self.ref_mean,
            "eval_mean": self.eval_mean,
            "ref_min": self.ref_min,
            "ref_max": self.ref_max,
            "mean_uncertainty": self.sum_uncertainty / n,
            "ucf": self.uncertainty_consistent_count / float(n),
        }


class BenchmarkMetricsCalculator:
    """Calculates evidence-qualified statistical accuracy metrics from aligned pairs."""

    @classmethod
    def deduce_semantic_class(
        cls,
        provenance: Optional[ReferenceProvenance],
        explicit_class: Optional[BenchmarkSemanticClass] = None,
    ) -> Tuple[BenchmarkSemanticClass, EvidenceRole]:
        """Determines the formal benchmark semantic class and evidence role from reference provenance."""
        if explicit_class is not None:
            role = EvidenceRole.PHYSICAL_REFERENCE if explicit_class == BenchmarkSemanticClass.PHYSICAL_REFERENCE_VALIDATION else EvidenceRole.FIELD_OBSERVATION
            return explicit_class, role

        if provenance is not None:
            role = provenance.evidence_role
            if role == EvidenceRole.PHYSICAL_REFERENCE:
                return BenchmarkSemanticClass.PHYSICAL_REFERENCE_VALIDATION, role
            elif role == EvidenceRole.FIELD_OBSERVATION:
                return BenchmarkSemanticClass.FIELD_BMS_TRACKING, role
            elif role == EvidenceRole.BMS_ESTIMATE_BASELINE:
                return BenchmarkSemanticClass.BMS_ESTIMATE_CROSS_CHECK, role
            elif role == EvidenceRole.SYNTHETIC_REFERENCE:
                return BenchmarkSemanticClass.ALGORITHMIC_SYNTHETIC_VERIFICATION, role

        # Default fallback
        return BenchmarkSemanticClass.ALGORITHMIC_SYNTHETIC_VERIFICATION, EvidenceRole.SYNTHETIC_REFERENCE

    @classmethod
    def compute_signal_metrics(
        cls,
        pairs: Sequence[AlignedSignalPair],
        signal_name: str,
        unit: str,
        semantic_class: Optional[BenchmarkSemanticClass] = None,
        provenance: Optional[ReferenceProvenance] = None,
        context_slice: Optional[ContextSliceIdentifier] = None,
        min_samples_for_r_squared: int = 2,
    ) -> SignalMetricSummary:
        """Calculates statistical metrics for a collection of AlignedSignalPairs.

        Args:
            pairs: Sequence of aligned signal pairs.
            signal_name: Name of evaluated signal.
            unit: Physical measurement unit string.
            semantic_class: Optional override for semantic classification.
            provenance: Reference provenance record.
            context_slice: Optional context slice identifier if evaluating a stratified subset.
            min_samples_for_r_squared: Minimum samples needed for valid R^2.

        Returns:
            SignalMetricSummary value object.
        """
        sem_class, ev_role = cls.deduce_semantic_class(provenance, semantic_class)

        total_samples = len(pairs)
        valid_pairs: list[AlignedSignalPair] = []
        excluded_reasons: dict[str, int] = {}

        prov_cand = provenance

        for p in pairs:
            if prov_cand is None and p.provenance is not None:
                prov_cand = p.provenance
                sem_class, ev_role = cls.deduce_semantic_class(prov_cand, semantic_class)

            if p.is_valid_pair and p.ref_value is not None:
                # Numerical validity check
                if math.isnan(p.ref_value) or math.isinf(p.ref_value) or math.isnan(p.eval_value) or math.isinf(p.eval_value):
                    reason = "NON_FINITE_VALUE"
                    excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
                else:
                    valid_pairs.append(p)
            else:
                reason = p.exclusion_reason or p.status.value
                excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

        n_valid = len(valid_pairs)
        n_excluded = total_samples - n_valid
        coverage = (n_valid / float(total_samples)) if total_samples > 0 else 0.0

        # Accumulate streaming metrics
        acc = StreamingWelfordAccumulator()
        for p in valid_pairs:
            acc.update(
                ref_val=p.ref_value,  # type: ignore
                eval_val=p.eval_value,
                ref_uncertainty=p.ref_uncertainty,
            )

        res = acc.compute_metrics(unit=unit, min_samples_for_r_squared=min_samples_for_r_squared)

        return SignalMetricSummary(
            signal_name=signal_name,
            unit=unit,
            semantic_class=sem_class,
            evidence_role=ev_role,
            provenance=prov_cand,
            total_eval_samples=total_samples,
            valid_paired_samples=n_valid,
            excluded_samples=n_excluded,
            exclusion_reasons=excluded_reasons,
            valid_pair_coverage=coverage,
            mae=res["mae"],
            rmse=res["rmse"],
            mean_bias_error=res["mbe"],
            max_absolute_error=res["max_error"],
            r_squared=res["r_squared"],
            nrmse=res["nrmse"],
            mean_reference_value=res["ref_mean"],
            mean_eval_value=res["eval_mean"],
            reference_min=res["ref_min"],
            reference_max=res["ref_max"],
            mean_reference_uncertainty=res["mean_uncertainty"],
            uncertainty_consistent_sample_count=acc.uncertainty_consistent_count,
            uncertainty_consistent_fraction=res["ucf"],
            context_slice=context_slice,
        )

    @classmethod
    def compute_stratified_metrics(
        cls,
        frame_pairs: Sequence[AlignedFramePair],
        signal_units: Mapping[str, str],
        stratifier: Optional[OperatingContextStratifier] = None,
        stratification_config: Optional[StratificationConfig] = None,
        alignment_policy: Optional[TemporalAlignmentPolicy] = None,
        min_samples_for_r_squared: int = 2,
    ) -> StratifiedMetricReport:
        """Calculates overall and context-stratified metric reports across multi-signal frame pairs."""
        strat = stratifier or OperatingContextStratifier()
        cfg = stratification_config or strat.get_default_config()

        # Identify all present signals
        all_signals: set[str] = set()
        for fp in frame_pairs:
            all_signals.update(fp.aligned_signals.keys())

        # 1. Bucket aligned signal pairs per context slice
        # Key: ContextSliceIdentifier -> signal_name -> list[AlignedSignalPair]
        bucket_pairs: dict[ContextSliceIdentifier, dict[str, list[AlignedSignalPair]]] = {}
        overall_pairs: dict[str, list[AlignedSignalPair]] = {s: [] for s in all_signals}

        for fp in frame_pairs:
            # Extract inputs for context classification
            curr = fp.eval_signals.get("pack_current_a")
            volt = fp.eval_signals.get("pack_voltage_v")
            temp = fp.eval_signals.get("avg_cell_temperature_c") or fp.eval_signals.get("temperature_c")
            soc = fp.eval_signals.get("soc_fraction")

            # Check reference frame for hints if available
            ctx_hint = fp.ref_frame.operating_context_hint if fp.ref_frame else None

            ctx = strat.classify(
                current_a=curr,
                voltage_v=volt,
                temperature_c=temp,
                soc_fraction=soc,
                operating_context_hint=ctx_hint,
                config=cfg,
            )

            for sig_name, sig_pair in fp.aligned_signals.items():
                overall_pairs[sig_name].append(sig_pair)

                # Add to each active slice
                for slice_id in ctx.active_slices:
                    if slice_id.dimension == ContextDimension.OVERALL:
                        continue
                    if slice_id not in bucket_pairs:
                        bucket_pairs[slice_id] = {s: [] for s in all_signals}
                    bucket_pairs[slice_id][sig_name].append(sig_pair)

        # 2. Compute Overall Summaries
        overall_summaries: dict[str, SignalMetricSummary] = {}
        for sig_name, p_list in overall_pairs.items():
            unit = signal_units.get(sig_name, "")
            summary = cls.compute_signal_metrics(
                pairs=p_list,
                signal_name=sig_name,
                unit=unit,
                context_slice=ContextSliceIdentifier(ContextDimension.OVERALL, "ALL"),
                min_samples_for_r_squared=min_samples_for_r_squared,
            )
            overall_summaries[sig_name] = summary

        # 3. Compute Stratified Summaries
        stratified_summaries: dict[str, dict[str, SignalMetricSummary]] = {}
        for slice_id, sig_map in bucket_pairs.items():
            slice_key = f"{slice_id.dimension.value}:{slice_id.bucket_name}"
            stratified_summaries[slice_key] = {}
            for sig_name, p_list in sig_map.items():
                unit = signal_units.get(sig_name, "")
                summary = cls.compute_signal_metrics(
                    pairs=p_list,
                    signal_name=sig_name,
                    unit=unit,
                    context_slice=slice_id,
                    min_samples_for_r_squared=min_samples_for_r_squared,
                )
                stratified_summaries[slice_key][sig_name] = summary

        total_samples = len(frame_pairs)
        valid_frames = sum(1 for fp in frame_pairs if any(p.is_valid_pair for p in fp.aligned_signals.values()))

        return StratifiedMetricReport(
            overall_metrics=overall_summaries,
            stratified_metrics=stratified_summaries,
            total_samples_evaluated=total_samples,
            overall_valid_pairs=valid_frames,
            alignment_policy=alignment_policy,
        )
