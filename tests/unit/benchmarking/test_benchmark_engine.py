"""Unit tests for Level 5.5 Phase 3 Benchmark Execution Engine and Events."""

import unittest

from src.benchmarking.dataset import (
    DatasetBuilder,
    DatasetMetadata,
    ReferenceObservationFrame,
)
from src.benchmarking.engine import (
    BenchmarkExecutionConfig,
    BenchmarkExecutionEngine,
)
from src.benchmarking.events import (
    BenchmarkDataQualityWarningEvent,
    BenchmarkExecutionCompletedEvent,
    BenchmarkExecutionStartedEvent,
)
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
from src.domain.battery.entities import BatteryPack


class TestBenchmarkExecutionEngine(unittest.TestCase):
    """Test suite for BenchmarkExecutionEngine end-to-end replay, alignment, metrics, and isolation."""

    def setUp(self) -> None:
        self.prov = ReferenceProvenance(
            provenance_id="prov_01",
            source_type=ReferenceSourceType.LABORATORY_METROLOGY,
            calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
            evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
            measurement_method="FOUR_WIRE_KELVIN",
        )
        self.metadata = DatasetMetadata(
            dataset_id="ds_engine_test",
            dataset_name="Engine Test Dataset",
            version="1.0.0",
            created_at_utc="2026-09-04T00:00:00Z",
        )
        self.ident = BatteryIdentification(identifier="pack_bench", display_name="Bench Pack")
        self.cell_cfg = CellConfiguration(
            cell_id="cell_nmc",
            chemistry=BatteryChemistry.NMC,
            form_factor=CellFormFactor.CYLINDRICAL,
            nominal_voltage_v=3.7,
            min_voltage_v=2.8,
            max_voltage_v=4.2,
            nominal_capacity_ah=2.5,
        )
        self.pack_cfg = PackConfiguration(
            pack_id="pack_bench",
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

    def _create_test_twin(self, system_id: str = "twin_bench") -> DigitalTwinInstance:
        pack = BatteryPack.create_monolithic_pack(
            identification=self.ident,
            configuration=self.pack_cfg,
            cell_config=self.cell_cfg,
        )
        ocv = LinearOCVModel(v_min_v=2.8, v_max_v=4.2)
        params = GenericECMParameters(nominal_capacity_ah=2.5, nominal_voltage_v=3.7, series_resistance_r0_ohm=0.02)
        meta = ModelMetadata(model_id="ecm_bench", name="ECM Bench", paradigm="EQUIVALENT_CIRCUIT")
        model = GenericECMModel(metadata=meta, parameters=params, ocv_model=ocv)
        return DigitalTwinInstance(
            battery_pack=pack,
            battery_model=model,
            config=RuntimeConfig(system_id=system_id, auto_publish_events=False),
        )

    # 1. End-to-end benchmark execution
    def test_end_to_end_benchmark_run(self) -> None:
        builder = DatasetBuilder(metadata=self.metadata)
        builder.set_battery_configuration(self.ident, self.pack_cfg, self.cell_cfg)

        ch_v = ReferenceSignalChannel(signal_name="voltage_v", signal_type=BenchmarkSignalType.VOLTAGE, unit="V", provenance=self.prov)
        builder.add_reference_channel(ch_v)

        # 10 steps of discharge at 1.0A
        for i in range(1, 11):
            ts = i * 1_000_000_000
            # Nominal terminal voltage with 0.02 ohm resistance: V = OCV(1.0) - I*R0 ~ 4.2 - 0.02 = 4.18V
            meas_v = ReferenceMeasurement(value=4.18, unit="V", timestamp_ns=ts, uncertainty_absolute=0.005, provenance=self.prov)
            frame = ReferenceObservationFrame(
                timestamp_ns=ts,
                reference_signals={"voltage_v": meas_v},
                field_signals={"pack_current_a": 1.0, "pack_voltage_v": 4.18, "ambient_temperature_c": 25.0},
            )
            builder.add_frame(frame)

        dataset = builder.build()
        twin = self._create_test_twin()

        bus = DigitalTwinEventBus()
        events = []
        bus.subscribe(event_type="benchmark.execution.started", handler=lambda e: events.append(e))
        bus.subscribe(event_type="benchmark.execution.completed", handler=lambda e: events.append(e))

        cfg = BenchmarkExecutionConfig(
            target_voltage_rmse_v=0.05,
        )

        result = BenchmarkExecutionEngine.run_benchmark(
            instance=twin,
            dataset=dataset,
            config=cfg,
            event_bus=bus,
        )

        self.assertEqual(result.dataset_id, "ds_engine_test")
        self.assertEqual(result.total_frames_evaluated, 10)
        self.assertTrue(result.is_passing)
        self.assertIsNotNone(result.signal_metrics)

        v_metrics = result.signal_metrics.get_overall("voltage_v")
        self.assertIsNotNone(v_metrics)
        self.assertEqual(v_metrics.valid_paired_samples, 10)
        self.assertLess(v_metrics.rmse.value, 0.05)

        # Verify event publication
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], BenchmarkExecutionStartedEvent)
        self.assertIsInstance(events[1], BenchmarkExecutionCompletedEvent)
        self.assertTrue(events[1].is_passing)

    # 2. Target threshold failure gating
    def test_target_threshold_failure(self) -> None:
        builder = DatasetBuilder(metadata=self.metadata)
        # Reference expects 4.18V, but twin with uncalibrated parameters will diverge
        for i in range(1, 6):
            ts = i * 1_000_000_000
            meas_v = ReferenceMeasurement(value=3.50, unit="V", timestamp_ns=ts, provenance=self.prov) # 3.5V vs simulated ~4.18V
            frame = ReferenceObservationFrame(
                timestamp_ns=ts,
                reference_signals={"voltage_v": meas_v},
                field_signals={"pack_current_a": 1.0, "ambient_temperature_c": 25.0},
            )
            builder.add_frame(frame)

        dataset = builder.build()
        twin = self._create_test_twin()

        # Strict target RMSE 0.02V
        cfg = BenchmarkExecutionConfig(target_voltage_rmse_v=0.02)
        result = BenchmarkExecutionEngine.run_benchmark(twin, dataset, cfg)

        self.assertFalse(result.is_passing)
        v_sum = result.signal_metrics.get_overall("voltage_v")
        self.assertGreater(v_sum.rmse.value, 0.02)

    # 3. Deterministic repeated execution
    def test_deterministic_repeated_execution(self) -> None:
        builder = DatasetBuilder(metadata=self.metadata)
        for i in range(1, 6):
            ts = i * 1_000_000_000
            meas_v = ReferenceMeasurement(value=4.18, unit="V", timestamp_ns=ts, provenance=self.prov)
            frame = ReferenceObservationFrame(
                timestamp_ns=ts,
                reference_signals={"voltage_v": meas_v},
                field_signals={"pack_current_a": 1.0},
            )
            builder.add_frame(frame)

        dataset = builder.build()
        twin1 = self._create_test_twin(system_id="twin_1")
        twin2 = self._create_test_twin(system_id="twin_2")

        res1 = BenchmarkExecutionEngine.run_benchmark(twin1, dataset)
        res2 = BenchmarkExecutionEngine.run_benchmark(twin2, dataset)

        # Same metrics
        v1 = res1.signal_metrics.get_overall("voltage_v")
        v2 = res2.signal_metrics.get_overall("voltage_v")
        self.assertAlmostEqual(v1.rmse.value, v2.rmse.value, places=6)
        self.assertAlmostEqual(v1.mae.value, v2.mae.value, places=6)

    # 4. Resource bound check
    def test_max_batch_samples_resource_bound(self) -> None:
        builder = DatasetBuilder(metadata=self.metadata)
        for i in range(1, 10):
            builder.add_frame(ReferenceObservationFrame(timestamp_ns=i * 1_000_000_000))
        dataset = builder.build()

        twin = self._create_test_twin()
        cfg = BenchmarkExecutionConfig(max_batch_samples=5)  # Bound is 5, but dataset has 9 frames

        with self.assertRaises(ValueError) as ctx:
            BenchmarkExecutionEngine.run_benchmark(twin, dataset, cfg)
        self.assertIn("exceeds configured max_batch_samples", str(ctx.exception))

    # 5. Type safety and failure state propagation
    def test_failure_state_propagation_type_errors(self) -> None:
        twin = self._create_test_twin()
        dataset = DatasetBuilder(metadata=self.metadata).build()

        with self.assertRaises(TypeError):
            BenchmarkExecutionEngine.run_benchmark("not_a_twin", dataset)  # type: ignore

        with self.assertRaises(TypeError):
            BenchmarkExecutionEngine.run_benchmark(twin, "not_a_dataset")  # type: ignore

    # 6. Replay isolation and live-state immutability
    def test_replay_isolation_and_immutability(self) -> None:
        builder = DatasetBuilder(metadata=self.metadata)
        for i in range(1, 6):
            meas_v = ReferenceMeasurement(value=4.18, unit="V", timestamp_ns=i * 1_000_000_000, provenance=self.prov)
            frame = ReferenceObservationFrame(
                timestamp_ns=i * 1_000_000_000,
                reference_signals={"voltage_v": meas_v},
                field_signals={"pack_current_a": 1.0},
            )
            builder.add_frame(frame)
        dataset = builder.build()
        twin = self._create_test_twin()

        frames_before = len(dataset.frames)
        result = BenchmarkExecutionEngine.run_benchmark(twin, dataset)
        frames_after = len(dataset.frames)

        self.assertEqual(frames_before, frames_after)
        self.assertEqual(result.total_frames_evaluated, 5)

    # 7. Diagnostic precision and recall threshold evaluation
    def test_diagnostic_precision_recall_thresholds(self) -> None:
        builder = DatasetBuilder(metadata=self.metadata)
        gt_event = ExperimentalGroundTruthEvent(
            event_id="gt_01",
            start_timestamp_ns=2_000_000_000,
            end_timestamp_ns=4_000_000_000,
            event_type="FAULT",
        )
        builder.add_ground_truth_event(gt_event)
        for i in range(1, 6):
            builder.add_frame(ReferenceObservationFrame(timestamp_ns=i * 1_000_000_000, field_signals={"pack_current_a": 0.0}))
        dataset = builder.build()
        twin = self._create_test_twin()

        # Config requiring 95% recall (which will fail if twin doesn't diagnose anything)
        cfg = BenchmarkExecutionConfig(min_diagnostic_recall=0.95)
        result = BenchmarkExecutionEngine.run_benchmark(twin, dataset, cfg)
        self.assertFalse(result.is_passing)
        self.assertIsNotNone(result.diagnostic_metrics)

    # 8. Serialization of execution result and events
    def test_result_and_event_serialization(self) -> None:
        twin = self._create_test_twin()
        dataset = DatasetBuilder(metadata=self.metadata).build()
        result = BenchmarkExecutionEngine.run_benchmark(twin, dataset)

        d = result.to_dict()
        self.assertIn("benchmark_id", d)
        self.assertIn("dataset_id", d)
        self.assertIn("is_passing", d)

        start_ev = BenchmarkExecutionStartedEvent(benchmark_id="b1", dataset_id="d1")
        self.assertEqual(start_ev.event_type, "benchmark.execution.started")
        self.assertIn("benchmark_id", start_ev.to_dict())

        comp_ev = BenchmarkExecutionCompletedEvent(benchmark_id="b1", dataset_id="d1", is_passing=True)
        self.assertEqual(comp_ev.event_type, "benchmark.execution.completed")
        self.assertTrue(comp_ev.to_dict()["is_passing"])

        warn_ev = BenchmarkDataQualityWarningEvent(benchmark_id="b1", dataset_id="d1", warning_code="GAP")
        self.assertEqual(warn_ev.event_type, "benchmark.quality.warning")
        self.assertEqual(warn_ev.to_dict()["warning_code"], "GAP")


if __name__ == "__main__":
    unittest.main()

