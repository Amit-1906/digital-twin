"""Unit and integration tests for Battery Benchmarking REST API routes."""

import unittest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import create_default_services
from src.benchmarking.dataset import (
    DatasetBuilder,
    DatasetMetadata,
    ReferenceObservationFrame,
)
from src.benchmarking.types import (
    BenchmarkSemanticClass,
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
from src.models.ecm import GenericECMModel, GenericECMParameters
from src.models.parameters.linear_ocv import LinearOCVModel
from src.models.types import ModelMetadata
from src.runtime.config import RuntimeConfig
from src.runtime.instance import DigitalTwinInstance


class TestBenchmarkRoutes(unittest.TestCase):
    """Test suite for benchmarking REST endpoints and response serialization."""

    def setUp(self) -> None:
        self.services = create_default_services()
        self.app = create_app(services=self.services)
        self.client = TestClient(self.app)

        self.prov = ReferenceProvenance(
            provenance_id="prov_api_test",
            source_type=ReferenceSourceType.LABORATORY_METROLOGY,
            calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
            evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
            measurement_method="FOUR_WIRE_KELVIN",
        )

        self.ident = BatteryIdentification(identifier="pack_api_test", display_name="API Test Pack")
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
            pack_id="pack_api_test",
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

        # Register sample reference dataset in benchmark service
        builder = DatasetBuilder(metadata=DatasetMetadata(
            dataset_id="ds_api_01",
            dataset_name="API Reference Benchmark",
            version="1.0.0",
            created_at_utc="2026-09-04T00:00:00Z",
            description="High precision lab benchmark",
        ))
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
        self.dataset = builder.build()
        self.services.benchmark_service.register_dataset(self.dataset)

        # Register test twin
        pack = BatteryPack.create_monolithic_pack(
            identification=self.ident,
            configuration=self.pack_cfg,
            cell_config=self.cell_cfg,
        )
        ocv = LinearOCVModel(v_min_v=2.8, v_max_v=4.2)
        params = GenericECMParameters(nominal_capacity_ah=2.5, nominal_voltage_v=3.7, series_resistance_r0_ohm=0.02)
        meta = ModelMetadata(model_id="ecm_api_test", name="ECM API Test", paradigm="EQUIVALENT_CIRCUIT")
        model = GenericECMModel(metadata=meta, parameters=params, ocv_model=ocv)
        twin = DigitalTwinInstance(
            battery_pack=pack,
            battery_model=model,
            config=RuntimeConfig(system_id="twin_api_01", auto_publish_events=False),
        )
        twin.initialize(initial_soc=1.0, temperature_c=25.0)
        self.services.twin_service.register_twin_instance(twin)

    # 1. Dataset Listing and Retrieval
    def test_list_and_get_datasets(self) -> None:
        res = self.client.get("/api/v1/benchmarks/datasets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["dataset_id"], "ds_api_01")
        self.assertEqual(data[0]["total_frames"], 5)

        res_single = self.client.get("/api/v1/benchmarks/datasets/ds_api_01")
        self.assertEqual(res_single.status_code, 200)
        self.assertEqual(res_single.json()["dataset_name"], "API Reference Benchmark")

    def test_get_dataset_not_found(self) -> None:
        res = self.client.get("/api/v1/benchmarks/datasets/missing_ds")
        self.assertEqual(res.status_code, 404)

    # 2. Benchmark Execution Run
    def test_run_benchmark_success(self) -> None:
        payload = {
            "dataset_id": "ds_api_01",
            "alignment_mode": "CAUSAL_STREAMING",
            "evaluate_signals": True,
            "target_voltage_rmse_v": 0.05,
        }
        res = self.client.post("/api/v1/benchmarks/twin_api_01/run", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["system_id"], "twin_api_01")
        self.assertEqual(data["dataset_id"], "ds_api_01")
        self.assertTrue(data["is_passing"])
        self.assertEqual(data["gating_status"], "PASS")
        self.assertEqual(data["total_frames_evaluated"], 5)
        self.assertIn("voltage_v", data["signals"])
        self.assertIsNotNone(data["signals"]["voltage_v"]["rmse"])

        # Retrieve latest
        res_lat = self.client.get("/api/v1/benchmarks/twin_api_01/latest")
        self.assertEqual(res_lat.status_code, 200)
        self.assertEqual(res_lat.json()["benchmark_id"], data["benchmark_id"])

        # Retrieve by ID
        res_by_id = self.client.get(f"/api/v1/benchmarks/results/{data['benchmark_id']}")
        self.assertEqual(res_by_id.status_code, 200)
        self.assertEqual(res_by_id.json()["benchmark_id"], data["benchmark_id"])

    # 3. Benchmark History & Reset
    def test_benchmark_history_and_reset(self) -> None:
        payload = {"dataset_id": "ds_api_01"}
        self.client.post("/api/v1/benchmarks/twin_api_01/run", json=payload)
        self.client.post("/api/v1/benchmarks/twin_api_01/run", json=payload)

        res_hist = self.client.get("/api/v1/benchmarks/twin_api_01/history")
        self.assertEqual(res_hist.status_code, 200)
        self.assertEqual(len(res_hist.json()), 2)

        res_global = self.client.get("/api/v1/benchmarks/results")
        self.assertEqual(res_global.status_code, 200)
        self.assertEqual(len(res_global.json()), 2)

        # Reset benchmark history
        res_reset = self.client.post("/api/v1/benchmarks/twin_api_01/reset")
        self.assertEqual(res_reset.status_code, 200)
        self.assertTrue(res_reset.json()["success"])

        # History should now be empty
        res_hist_after = self.client.get("/api/v1/benchmarks/twin_api_01/history")
        self.assertEqual(len(res_hist_after.json()), 0)

        # Twin is still present
        res_twin = self.client.get("/api/v1/twins/twin_api_01")
        self.assertEqual(res_twin.status_code, 200)

    # 4. Error Handling
    def test_run_benchmark_missing_twin_returns_404(self) -> None:
        res = self.client.post("/api/v1/benchmarks/missing_twin/run", json={"dataset_id": "ds_api_01"})
        self.assertEqual(res.status_code, 404)

    def test_run_benchmark_missing_dataset_returns_404(self) -> None:
        res = self.client.post("/api/v1/benchmarks/twin_api_01/run", json={"dataset_id": "missing_ds"})
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
