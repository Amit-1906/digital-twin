"""Unit tests for BenchmarkApplicationService (Level 5.5 Phase 4)."""

import unittest

from src.benchmarking.dataset import (
    DatasetBuilder,
    DatasetMetadata,
    ReferenceObservationFrame,
)
from src.benchmarking.engine import BenchmarkExecutionConfig
from src.benchmarking.types import (
    BenchmarkSignalType,
    CalibrationStatus,
    EvidenceRole,
    ExperimentalGroundTruthEvent,
    ReferenceMeasurement,
    ReferenceProvenance,
    ReferenceSignalChannel,
    ReferenceSourceType,
)
from src.domain.battery.entities import BatteryPack
from src.domain.battery.enums import BatteryChemistry, CellFormFactor
from src.domain.battery.value_objects import (
    BatteryIdentification,
    BatteryTopology,
    CellConfiguration,
    ElectricalRatings,
    PackConfiguration,
    ThermalLimits,
)
from src.events.bus import DigitalTwinEventBus
from src.models.ecm import GenericECMModel, GenericECMParameters
from src.models.parameters.linear_ocv import LinearOCVModel
from src.models.types import ModelMetadata
from src.runtime.config import RuntimeConfig
from src.runtime.instance import DigitalTwinInstance
from src.services.benchmark_service import (
    BenchmarkService,
    BenchmarkServiceConfig,
    DatasetNotFoundError,
)
from src.services.exceptions import DuplicateEntityError, TwinNotFoundError
from src.services.twin_service import TwinApplicationService
from src.storage.memory_repository import (
    InMemoryStateHistoryRepository,
    InMemoryTelemetryRepository,
)


class TestBenchmarkService(unittest.TestCase):
    """Comprehensive test suite for BenchmarkService orchestration, isolation, and bounded history."""

    def setUp(self) -> None:
        self.prov = ReferenceProvenance(
            provenance_id="prov_test",
            source_type=ReferenceSourceType.LABORATORY_METROLOGY,
            calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
            evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
            measurement_method="FOUR_WIRE_KELVIN",
        )
        self.meta = DatasetMetadata(
            dataset_id="ds_test_01",
            dataset_name="Test Reference Dataset",
            version="1.0.0",
            created_at_utc="2026-09-04T00:00:00Z",
        )
        self.ident = BatteryIdentification(identifier="pack_service_test", display_name="Test Pack")
        self.cell_cfg = CellConfiguration(
            cell_id="cell_01",
            chemistry=BatteryChemistry.NMC,
            form_factor=CellFormFactor.CYLINDRICAL,
            nominal_voltage_v=3.7,
            min_voltage_v=2.8,
            max_voltage_v=4.2,
            nominal_capacity_ah=2.5,
        )
        self.pack_cfg = PackConfiguration(
            pack_id="pack_service_test",
            topology=BatteryTopology(series_count=1, parallel_count=1),
            electrical_ratings=ElectricalRatings(
                nominal_voltage_v=3.7,
                min_voltage_v=2.8,
                max_voltage_v=4.2,
                nominal_capacity_ah=2.5,
                nominal_energy_wh=9.25,
                max_continuous_charge_current_a=2.5,
                max_continuous_discharge_current_a=5.0,
                peak_charge_current_a=5.0,
                peak_discharge_current_a=10.0,
            ),
            thermal_limits=ThermalLimits(
                min_charge_temp_c=0.0,
                max_charge_temp_c=45.0,
                min_discharge_temp_c=-20.0,
                max_discharge_temp_c=60.0,
                warning_temp_c=60.0,
                critical_temp_c=75.0,
            ),
        )

        self.event_bus = DigitalTwinEventBus()
        self.twin_service = TwinApplicationService(
            event_bus=self.event_bus,
            telemetry_repo=InMemoryTelemetryRepository(),
            state_repo=InMemoryStateHistoryRepository(),
        )
        self.benchmark_service = BenchmarkService(
            twin_service=self.twin_service,
            event_bus=self.event_bus,
            config=BenchmarkServiceConfig(max_result_history=5),
        )

    def _create_sample_dataset(self, dataset_id: str = "ds_test_01") -> None:
        builder = DatasetBuilder(metadata=DatasetMetadata(dataset_id=dataset_id, dataset_name="Dataset A", version="1.0.0", created_at_utc="2026-09-04T00:00:00Z"))
        builder.set_battery_configuration(self.ident, self.pack_cfg, self.cell_cfg)
        ch_v = ReferenceSignalChannel(
            signal_name="voltage_v",
            signal_type=BenchmarkSignalType.VOLTAGE,
            unit="V",
            provenance=self.prov,
        )
        builder.add_reference_channel(ch_v)
        for i in range(1, 6):
            ts = i * 1_000_000_000
            meas = ReferenceMeasurement(value=4.18, unit="V", timestamp_ns=ts, provenance=self.prov)
            frame = ReferenceObservationFrame(
                timestamp_ns=ts,
                reference_signals={"voltage_v": meas},
                field_signals={"pack_current_a": 1.0, "pack_voltage_v": 4.18},
            )
            builder.add_frame(frame)
        return builder.build()

    def _register_test_twin(self, system_id: str = "twin_test_01") -> DigitalTwinInstance:
        pack = BatteryPack.create_monolithic_pack(
            identification=self.ident,
            configuration=self.pack_cfg,
            cell_config=self.cell_cfg,
        )
        ocv = LinearOCVModel(v_min_v=2.8, v_max_v=4.2)
        params = GenericECMParameters(nominal_capacity_ah=2.5, nominal_voltage_v=3.7, series_resistance_r0_ohm=0.02)
        meta = ModelMetadata(model_id=f"ecm_{system_id}", name="ECM Test", paradigm="EQUIVALENT_CIRCUIT")
        model = GenericECMModel(metadata=meta, parameters=params, ocv_model=ocv)
        twin = DigitalTwinInstance(
            battery_pack=pack,
            battery_model=model,
            config=RuntimeConfig(system_id=system_id, auto_publish_events=False),
        )
        twin.initialize(initial_soc=1.0, temperature_c=25.0)
        self.twin_service.register_twin_instance(twin)
        return twin

    # 1. Dataset Registration & Listing
    def test_dataset_registration_and_retrieval(self) -> None:
        dataset = self._create_sample_dataset("ds_alpha")
        self.benchmark_service.register_dataset(dataset)

        retrieved = self.benchmark_service.get_dataset("ds_alpha")
        self.assertEqual(retrieved.dataset_id, "ds_alpha")
        self.assertEqual(len(retrieved.frames), 5)

        summaries = self.benchmark_service.list_datasets()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["dataset_id"], "ds_alpha")
        self.assertEqual(summaries[0]["total_frames"], 5)

    def test_duplicate_dataset_registration_raises(self) -> None:
        dataset = self._create_sample_dataset("ds_dup")
        self.benchmark_service.register_dataset(dataset)

        with self.assertRaises(DuplicateEntityError):
            self.benchmark_service.register_dataset(dataset)

    def test_missing_dataset_retrieval_raises(self) -> None:
        with self.assertRaises(DatasetNotFoundError):
            self.benchmark_service.get_dataset("non_existent_ds")

    # 2. Benchmark Execution & Result Storage
    def test_benchmark_run_orchestration_and_result_storage(self) -> None:
        dataset = self._create_sample_dataset("ds_run")
        self.benchmark_service.register_dataset(dataset)
        self._register_test_twin("twin_01")

        result = self.benchmark_service.run_benchmark("twin_01", "ds_run")
        self.assertEqual(result.dataset_id, "ds_run")
        self.assertEqual(result.total_frames_evaluated, 5)
        self.assertTrue(result.is_passing)

        latest = self.benchmark_service.get_latest_result("twin_01")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.benchmark_id, result.benchmark_id)

        by_id = self.benchmark_service.get_result(result.benchmark_id)
        self.assertIsNotNone(by_id)
        self.assertEqual(by_id.benchmark_id, result.benchmark_id)

    # 3. Live Twin Isolation Guarantee
    def test_live_twin_state_remains_unchanged_after_benchmark(self) -> None:
        dataset = self._create_sample_dataset("ds_iso")
        self.benchmark_service.register_dataset(dataset)
        twin = self._register_test_twin("twin_iso")

        # Capture live twin state snapshot before benchmark
        soc_before = twin.current_model_state.soc_fraction
        temp_before = twin.current_model_state.temperature_c
        steps_before = twin.total_steps

        self.benchmark_service.run_benchmark("twin_iso", "ds_iso")

        # Verify live twin state is completely untouched
        self.assertEqual(twin.current_model_state.soc_fraction, soc_before)
        self.assertEqual(twin.current_model_state.temperature_c, temp_before)
        self.assertEqual(twin.total_steps, steps_before)

    # 4. Bounded Result History
    def test_bounded_result_history_enforcement(self) -> None:
        dataset = self._create_sample_dataset("ds_bound")
        self.benchmark_service.register_dataset(dataset)
        self._register_test_twin("twin_bound")

        # Run 8 benchmark iterations on service with max_result_history=5
        for _ in range(8):
            self.benchmark_service.run_benchmark("twin_bound", "ds_bound")

        history = self.benchmark_service.get_history("twin_bound")
        self.assertEqual(len(history), 5)  # Enforces capacity cap of 5

        global_history = self.benchmark_service.get_history(system_id=None)
        self.assertEqual(len(global_history), 5)

    # 5. Benchmark-Only Reset Semantics
    def test_benchmark_reset_clears_history_without_resetting_live_twin(self) -> None:
        dataset = self._create_sample_dataset("ds_reset")
        self.benchmark_service.register_dataset(dataset)
        twin = self._register_test_twin("twin_reset")

        self.benchmark_service.run_benchmark("twin_reset", "ds_reset")
        self.assertIsNotNone(self.benchmark_service.get_latest_result("twin_reset"))

        # Reset benchmark history
        self.benchmark_service.reset("twin_reset")

        # Verify benchmark history is cleared
        self.assertIsNone(self.benchmark_service.get_latest_result("twin_reset"))
        self.assertEqual(len(self.benchmark_service.get_history("twin_reset")), 0)

        # Verify twin instance is still registered and valid in twin_service
        self.assertTrue(self.twin_service.exists("twin_reset"))
        self.assertIsNotNone(self.twin_service.get_twin("twin_reset"))

    # 6. Missing Twin Error Propagation
    def test_missing_twin_raises_error(self) -> None:
        dataset = self._create_sample_dataset("ds_notwin")
        self.benchmark_service.register_dataset(dataset)

        with self.assertRaises(TwinNotFoundError):
            self.benchmark_service.run_benchmark("non_existent_twin", "ds_notwin")


if __name__ == "__main__":
    unittest.main()
