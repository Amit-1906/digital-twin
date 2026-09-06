"""Battery Benchmarking and Experimental Validation Application Service.

Coordinates experimental reference dataset registration, isolated co-simulation replay runs,
deterministic metric evaluation, and bounded benchmark result storage without modifying live digital twins.
"""

from collections import deque
import copy
from dataclasses import dataclass, field
import threading
from typing import Any, Mapping, Optional

from src.benchmarking.dataset import ExperimentalDataset
from src.benchmarking.engine import (
    BenchmarkExecutionConfig,
    BenchmarkExecutionEngine,
    BenchmarkExecutionResult,
)
from src.events.base import AbstractEventBus
from src.runtime.instance import DigitalTwinInstance
from src.services.exceptions import (
    DuplicateEntityError,
    EntityNotFoundError,
    InvalidServiceOperationError,
    ServiceError,
    TwinNotFoundError,
)
from src.services.twin_service import TwinApplicationService


class DatasetNotFoundError(EntityNotFoundError):
    """Raised when a requested reference dataset identifier does not exist."""


@dataclass(frozen=True)
class BenchmarkServiceConfig:
    """Configuration options governing the BenchmarkApplicationService."""

    max_result_history: int = 100
    max_batch_samples: int = 100_000

    def __post_init__(self) -> None:
        if self.max_result_history <= 0:
            raise ValueError(f"max_result_history must be strictly positive, got {self.max_result_history}.")
        if self.max_batch_samples <= 0:
            raise ValueError(f"max_batch_samples must be strictly positive, got {self.max_batch_samples}.")


class BenchmarkService:
    """Application service for orchestrating reference benchmarking and experimental evidence evaluation.

    Manages registered ExperimentalDataset instances, executes sandboxed BenchmarkExecutionEngine runs
    against digital twin instances managed by TwinApplicationService, and maintains bounded historical results.
    """

    def __init__(
        self,
        twin_service: TwinApplicationService,
        event_bus: Optional[AbstractEventBus] = None,
        config: Optional[BenchmarkServiceConfig] = None,
    ) -> None:
        if not isinstance(twin_service, TwinApplicationService):
            raise TypeError(f"Expected TwinApplicationService, got {type(twin_service).__name__}.")

        self._twin_service = twin_service
        self._event_bus = event_bus
        self._config = config or BenchmarkServiceConfig()

        self._datasets: dict[str, ExperimentalDataset] = {}
        self._results_by_id: dict[str, BenchmarkExecutionResult] = {}
        self._system_history: dict[str, deque[BenchmarkExecutionResult]] = {}
        self._global_history: deque[BenchmarkExecutionResult] = deque(maxlen=self._config.max_result_history)
        self._lock = threading.RLock()

    @property
    def twin_service(self) -> TwinApplicationService:
        """Attached TwinApplicationService managing digital twin instances."""
        return self._twin_service

    @property
    def config(self) -> BenchmarkServiceConfig:
        """Active benchmark service configuration."""
        return self._config

    def register_dataset(self, dataset: ExperimentalDataset) -> None:
        """Registers an immutable ExperimentalDataset into the benchmark service registry.

        Args:
            dataset: ExperimentalDataset instance with valid metadata and provenance.

        Raises:
            TypeError: If dataset is not an ExperimentalDataset.
            DuplicateEntityError: If dataset_id is already registered.
        """
        if not isinstance(dataset, ExperimentalDataset):
            raise TypeError(f"Expected ExperimentalDataset, got {type(dataset).__name__}.")

        with self._lock:
            if dataset.dataset_id in self._datasets:
                raise DuplicateEntityError(
                    f"Dataset with ID '{dataset.dataset_id}' is already registered.",
                    service_name="BenchmarkService",
                    details={"dataset_id": dataset.dataset_id},
                )
            self._datasets[dataset.dataset_id] = dataset

    def get_dataset(self, dataset_id: str) -> ExperimentalDataset:
        """Retrieves a registered ExperimentalDataset by its dataset_id.

        Args:
            dataset_id: Unique dataset identifier.

        Returns:
            ExperimentalDataset.

        Raises:
            DatasetNotFoundError: If dataset_id is not found.
        """
        with self._lock:
            if dataset_id not in self._datasets:
                raise DatasetNotFoundError(
                    f"Dataset with ID '{dataset_id}' does not exist.",
                    service_name="BenchmarkService",
                    details={"dataset_id": dataset_id},
                )
            return self._datasets[dataset_id]

    def list_datasets(self) -> list[dict[str, Any]]:
        """Returns metadata summaries for all registered reference datasets."""
        with self._lock:
            summaries = []
            for ds in self._datasets.values():
                ch = ds.reference_channels[0] if ds.reference_channels else None
                ev_role = ch.provenance.evidence_role.value if (ch and ch.provenance) else None
                sem_class = "PHYSICAL_REFERENCE_VALIDATION" if (ch and ch.provenance and ch.provenance.evidence_role.value == "PHYSICAL_REFERENCE") else "ALGORITHMIC_SYNTHETIC_VERIFICATION"

                summaries.append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.metadata.dataset_name,
                    "version": ds.metadata.version,
                    "total_frames": len(ds.frames),
                    "start_timestamp_ns": ds.start_timestamp_ns,
                    "end_timestamp_ns": ds.end_timestamp_ns,
                    "duration_seconds": ds.duration_seconds,
                    "reference_channels": [c.signal_name for c in ds.reference_channels],
                    "ground_truth_events_count": len(ds.ground_truth_events),
                    "semantic_class": sem_class,
                    "evidence_role": ev_role,
                    "description": ds.metadata.description,
                })
            return summaries

    def run_benchmark(
        self,
        system_id: str,
        dataset_id: str,
        config: Optional[BenchmarkExecutionConfig] = None,
    ) -> BenchmarkExecutionResult:
        """Executes an isolated benchmark co-simulation run of a registered dataset against a digital twin.

        Args:
            system_id: Identifier of the target digital twin.
            dataset_id: Identifier of the registered ExperimentalDataset.
            config: Optional BenchmarkExecutionConfig overriding tolerances.

        Returns:
            BenchmarkExecutionResult containing metrological evaluation metrics.

        Raises:
            TwinNotFoundError: If system_id does not exist.
            DatasetNotFoundError: If dataset_id does not exist.
        """
        # 1. Resolve twin from TwinApplicationService without mutating its lifecycle
        twin = self._twin_service.get_twin(system_id)
        # 2. Resolve dataset
        dataset = self.get_dataset(dataset_id)

        # 3. Create isolated sandbox copy of the digital twin to ensure zero mutation of live twin
        sandbox_twin = DigitalTwinInstance(
            battery_pack=twin.battery_pack if hasattr(twin, "battery_pack") else twin._pack,
            battery_model=copy.deepcopy(twin.battery_model if hasattr(twin, "battery_model") else twin._model),
            state_estimator=copy.deepcopy(twin.state_estimator if hasattr(twin, "state_estimator") else twin._estimator) if (hasattr(twin, "state_estimator") and twin.state_estimator) or twin._estimator else None,
            health_estimator=copy.deepcopy(twin.health_estimator if hasattr(twin, "health_estimator") else twin._health_estimator) if (hasattr(twin, "health_estimator") and twin.health_estimator) or twin._health_estimator else None,
            parameter_identifier=copy.deepcopy(twin.parameter_identifier if hasattr(twin, "parameter_identifier") else twin._parameter_identifier) if (hasattr(twin, "parameter_identifier") and twin.parameter_identifier) or twin._parameter_identifier else None,
            validation_engine=copy.deepcopy(twin.validation_engine if hasattr(twin, "validation_engine") else twin._validation_engine) if (hasattr(twin, "validation_engine") and twin.validation_engine) or twin._validation_engine else None,
            diagnostic_engine=copy.deepcopy(twin.diagnostic_engine if hasattr(twin, "diagnostic_engine") else twin._diagnostic_engine) if (hasattr(twin, "diagnostic_engine") and twin.diagnostic_engine) or twin._diagnostic_engine else None,
            config=copy.deepcopy(twin.config if hasattr(twin, "config") else twin._config),
        )
        if twin.is_initialized and twin.current_model_state:
            sandbox_twin.initialize(
                initial_soc=twin.current_model_state.soc_fraction,
                temperature_c=twin.current_model_state.temperature_c,
            )

        # 4. Merge configuration limits
        exec_cfg = config or BenchmarkExecutionConfig()
        if exec_cfg.max_batch_samples > self._config.max_batch_samples:
            exec_cfg = BenchmarkExecutionConfig(
                system_id=exec_cfg.system_id,
                alignment_policy=exec_cfg.alignment_policy,
                stratification_config=exec_cfg.stratification_config,
                evaluate_diagnostics=exec_cfg.evaluate_diagnostics,
                evaluate_signals=exec_cfg.evaluate_signals,
                signal_units=exec_cfg.signal_units,
                target_voltage_rmse_v=exec_cfg.target_voltage_rmse_v,
                target_temp_rmse_c=exec_cfg.target_temp_rmse_c,
                target_soc_rmse=exec_cfg.target_soc_rmse,
                min_diagnostic_precision=exec_cfg.min_diagnostic_precision,
                min_diagnostic_recall=exec_cfg.min_diagnostic_recall,
                max_batch_samples=self._config.max_batch_samples,
                auto_reset_instance=exec_cfg.auto_reset_instance,
            )

        # 5. Execute isolated benchmark run
        result = BenchmarkExecutionEngine.run_benchmark(
            instance=sandbox_twin,
            dataset=dataset,
            config=exec_cfg,
            event_bus=self._event_bus,
        )

        # 5. Record result in bounded history
        with self._lock:
            self._results_by_id[result.benchmark_id] = result
            if system_id not in self._system_history:
                self._system_history[system_id] = deque(maxlen=self._config.max_result_history)
            self._system_history[system_id].append(result)
            self._global_history.append(result)

            # Enforce bounds on _results_by_id
            if len(self._results_by_id) > self._config.max_result_history * 5:
                # Retain only IDs present in global or system histories
                retained_ids = {r.benchmark_id for r in self._global_history}
                for sys_deque in self._system_history.values():
                    retained_ids.update(r.benchmark_id for r in sys_deque)
                self._results_by_id = {
                    bid: res for bid, res in self._results_by_id.items()
                    if bid in retained_ids
                }

        return result

    def get_latest_result(self, system_id: str) -> Optional[BenchmarkExecutionResult]:
        """Retrieves the most recent benchmark execution result for a digital twin system."""
        with self._lock:
            history = self._system_history.get(system_id)
            if history and len(history) > 0:
                return history[-1]
            return None

    def get_result(self, benchmark_id: str) -> Optional[BenchmarkExecutionResult]:
        """Retrieves a specific benchmark execution result by its benchmark_id."""
        with self._lock:
            return self._results_by_id.get(benchmark_id)

    def get_history(
        self,
        system_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[BenchmarkExecutionResult]:
        """Retrieves historical benchmark execution results, optionally filtered by system_id.

        Args:
            system_id: Optional twin system identifier to filter by.
            limit: Optional maximum number of recent results to return.

        Returns:
            List of BenchmarkExecutionResult instances in reverse chronological order (newest first).
        """
        with self._lock:
            if system_id is not None:
                history = self._system_history.get(system_id, deque())
                results = list(history)
            else:
                results = list(self._global_history)

            results.reverse()
            if limit is not None and limit > 0:
                results = results[:limit]
            return results

    def reset(self, system_id: Optional[str] = None) -> None:
        """Resets benchmark service execution results history ONLY.

        WARNING: This operation clears benchmark execution cache/history.
        It DOES NOT reset the live DigitalTwinInstance, EKF, ECM, SOH, RLS, or telemetry history.

        Args:
            system_id: Optional system identifier to reset history for, or None to clear all benchmark history.
        """
        with self._lock:
            if system_id is not None:
                if system_id in self._system_history:
                    # Remove from results_by_id
                    for res in self._system_history[system_id]:
                        self._results_by_id.pop(res.benchmark_id, None)
                    self._system_history.pop(system_id, None)
                # Filter global history
                filtered_global = deque(
                    (r for r in self._global_history if r.metadata.get("system_id") != system_id),
                    maxlen=self._config.max_result_history,
                )
                self._global_history = filtered_global
            else:
                self._results_by_id.clear()
                self._system_history.clear()
                self._global_history.clear()
