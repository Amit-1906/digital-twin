"""TwinVolt Real-World Validation, Benchmarking, and Experimental Evidence Framework.

Level 5.5 introduces foundational, battery-agnostic, and hardware-agnostic contracts
for reference dataset representation, causal temporal alignment, operating-context stratification,
quantitative metric calculations, diagnostic ground-truth evaluation, and benchmark execution.
"""

from src.benchmarking.alignment import (
    AlignedFramePair,
    AlignedSignalPair,
    AlignmentMode,
    AlignmentStatus,
    InterpolationPolicy,
    ObservationProvenanceType,
    ResamplingPolicy,
    TemporalAlignmentEngine,
    TemporalAlignmentPolicy,
)
from src.benchmarking.dataset import (
    DatasetBuilder,
    DatasetMetadata,
    ExperimentalDataset,
    ReferenceObservationFrame,
)
from src.benchmarking.diagnostic_evaluator import (
    DetectionLatencySummary,
    DiagnosticEvaluationReport,
    DiagnosticGroundTruthEvaluator,
    EventDiagnosticEvaluation,
    EventMatchStatus,
    GroundTruthEventType,
    SampleConfusionMatrix,
)
from src.benchmarking.engine import (
    BenchmarkExecutionConfig,
    BenchmarkExecutionEngine,
    BenchmarkExecutionResult,
)
from src.benchmarking.events import (
    BenchmarkDataQualityWarningEvent,
    BenchmarkExecutionCompletedEvent,
    BenchmarkExecutionStartedEvent,
)
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
    NumericRangeBucket,
    OperatingContextStratifier,
    StratificationConfig,
    StratifiedContext,
)
from src.benchmarking.types import (
    BenchmarkSemanticClass,
    BenchmarkSignalType,
    CalibrationStatus,
    EvidenceRole,
    ExperimentalGroundTruthEvent,
    ReferenceDataQuality,
    ReferenceMeasurement,
    ReferenceProvenance,
    ReferenceSignalChannel,
    ReferenceSourceType,
)

__all__ = [
    # Provenance and Metrological Classification
    "ReferenceSourceType",
    "CalibrationStatus",
    "EvidenceRole",
    "ReferenceProvenance",
    # Measurement and Signal Typing
    "BenchmarkSignalType",
    "BenchmarkSemanticClass",
    "ReferenceDataQuality",
    "ReferenceMeasurement",
    "ReferenceSignalChannel",
    # Experimental Events and Ground Truth
    "ExperimentalGroundTruthEvent",
    "GroundTruthEventType",
    "EventMatchStatus",
    "EventDiagnosticEvaluation",
    "SampleConfusionMatrix",
    "DetectionLatencySummary",
    "DiagnosticEvaluationReport",
    "DiagnosticGroundTruthEvaluator",
    # Dataset Abstractions
    "DatasetMetadata",
    "ReferenceObservationFrame",
    "ExperimentalDataset",
    "DatasetBuilder",
    # Temporal Alignment
    "AlignmentMode",
    "InterpolationPolicy",
    "ResamplingPolicy",
    "AlignmentStatus",
    "ObservationProvenanceType",
    "TemporalAlignmentPolicy",
    "AlignedSignalPair",
    "AlignedFramePair",
    "TemporalAlignmentEngine",
    # Stratification
    "ContextDimension",
    "NumericRangeBucket",
    "ContextSliceIdentifier",
    "StratificationConfig",
    "StratifiedContext",
    "OperatingContextStratifier",
    # Metrics
    "SingleMetricValue",
    "SignalMetricSummary",
    "StratifiedMetricReport",
    "StreamingWelfordAccumulator",
    "BenchmarkMetricsCalculator",
    # Engine & Events
    "BenchmarkExecutionConfig",
    "BenchmarkExecutionResult",
    "BenchmarkExecutionEngine",
    "BenchmarkExecutionStartedEvent",
    "BenchmarkExecutionCompletedEvent",
    "BenchmarkDataQualityWarningEvent",
]
