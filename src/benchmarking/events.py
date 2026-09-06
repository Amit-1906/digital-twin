"""Observability and Event Bus Events for Battery Benchmarking.

Defines lifecycle events published during benchmark execution sessions.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Optional

from src.events.base import TwinEvent


@dataclass(frozen=True)
class BenchmarkExecutionStartedEvent(TwinEvent):
    """Published when an automated benchmark execution run begins."""

    benchmark_id: str = ""
    dataset_id: str = ""
    dataset_name: str = ""
    total_frames: int = 0
    event_type: str = "benchmark.execution.started"

    def to_dict(self) -> dict[str, Any]:
        """Serializes event to dictionary."""
        d = super().to_dict()
        d.update({
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "total_frames": self.total_frames,
        })
        return d


@dataclass(frozen=True)
class BenchmarkExecutionCompletedEvent(TwinEvent):
    """Published when a benchmark execution run successfully completes."""

    benchmark_id: str = ""
    dataset_id: str = ""
    duration_seconds: float = 0.0
    total_frames: int = 0
    is_passing: bool = True
    key_metrics: Mapping[str, Any] = field(default_factory=dict)
    event_type: str = "benchmark.execution.completed"

    def to_dict(self) -> dict[str, Any]:
        """Serializes event to dictionary."""
        d = super().to_dict()
        d.update({
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.dataset_id,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "is_passing": self.is_passing,
            "key_metrics": dict(self.key_metrics),
        })
        return d


@dataclass(frozen=True)
class BenchmarkDataQualityWarningEvent(TwinEvent):
    """Published when data quality or temporal gaps are detected in a reference benchmark."""

    benchmark_id: str = ""
    dataset_id: str = ""
    warning_code: str = ""
    message: str = ""
    event_type: str = "benchmark.quality.warning"

    def to_dict(self) -> dict[str, Any]:
        """Serializes event to dictionary."""
        d = super().to_dict()
        d.update({
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.dataset_id,
            "warning_code": self.warning_code,
            "message": self.message,
        })
        return d

