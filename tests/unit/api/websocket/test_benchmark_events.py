"""Unit tests verifying WebSocket bridging for Benchmark Events (Level 5.5 Phase 4)."""

import unittest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import create_default_services
from src.benchmarking.events import (
    BenchmarkDataQualityWarningEvent,
    BenchmarkExecutionCompletedEvent,
    BenchmarkExecutionStartedEvent,
)


class TestBenchmarkWebSocketEvents(unittest.TestCase):
    """Test suite verifying benchmark event delivery and isolation over WebSocket."""

    def setUp(self) -> None:
        self.services = create_default_services()
        self.app = create_app(services=self.services)
        self.client = TestClient(self.app)

        # Setup pack and twin
        pack_payload = {
            "schema_version": "1.0",
            "profile_id": "pack_bench_ws",
            "display_name": "Benchmark WS Pack",
            "chemistry": "NMC",
            "topology": {"series_count": 1, "parallel_count": 1},
            "cell_profile": {
                "cell_id": "cell_01",
                "chemistry": "NMC",
                "form_factor": "CYLINDRICAL",
                "nominal_voltage_v": 3.7,
                "min_voltage_v": 2.8,
                "max_voltage_v": 4.2,
                "nominal_capacity_ah": 2.5,
            },
            "ratings": {
                "nominal_pack_voltage_v": 3.7,
                "nominal_cell_voltage_v": 3.7,
                "nominal_capacity_ah": 2.5,
                "nominal_energy_wh": 9.25,
            },
            "voltage_limits": {
                "cell_min_cutoff_v": 2.8,
                "cell_max_cutoff_v": 4.2,
                "pack_min_cutoff_v": 2.8,
                "pack_max_cutoff_v": 4.2,
            },
            "current_limits": {
                "max_continuous_charge_a": 2.5,
                "max_continuous_discharge_a": 5.0,
                "peak_pulse_discharge_a": 10.0,
            },
            "thermal_limits": {
                "min_charge_temp_c": 0.0,
                "max_charge_temp_c": 45.0,
                "min_discharge_temp_c": -20.0,
                "max_discharge_temp_c": 60.0,
                "thermal_warning_temp_c": 60.0,
                "critical_thermal_runaway_temp_c": 75.0,
            },
        }
        self.client.post("/api/v1/packs", json=pack_payload)
        self.client.post(
            "/api/v1/twins",
            json={"system_id": "twin_bench_ws", "pack_id": "pack_bench_ws", "auto_initialize": True},
        )

    def test_benchmark_lifecycle_events_bridged_to_websocket(self) -> None:
        """Publishing benchmark events to EventBus bridges them to WebSocket subscribers."""
        with self.client.websocket_connect("/api/v1/ws/twins/twin_bench_ws") as ws:
            # 1. Receive connected welcome message
            welcome = ws.receive_json()
            self.assertEqual(welcome["type"], "connected")

            # 2. Publish BenchmarkExecutionStartedEvent
            self.services.event_bus.publish(
                BenchmarkExecutionStartedEvent(
                    benchmark_id="bench_ws_01",
                    dataset_id="ds_test",
                    dataset_name="Test Dataset",
                    total_frames=100,
                    source_id="twin_bench_ws",
                    timestamp_ns=1_000_000_000,
                )
            )
            msg_started = ws.receive_json()
            self.assertEqual(msg_started["type"], "twin_event")
            self.assertEqual(msg_started["event_type"], "benchmark.execution.started")
            self.assertEqual(msg_started["system_id"], "twin_bench_ws")

            # 3. Publish BenchmarkExecutionCompletedEvent
            self.services.event_bus.publish(
                BenchmarkExecutionCompletedEvent(
                    benchmark_id="bench_ws_01",
                    dataset_id="ds_test",
                    duration_seconds=0.125,
                    total_frames=100,
                    is_passing=True,
                    source_id="twin_bench_ws",
                    timestamp_ns=2_000_000_000,
                )
            )
            msg_completed = ws.receive_json()
            self.assertEqual(msg_completed["type"], "twin_event")
            self.assertEqual(msg_completed["event_type"], "benchmark.execution.completed")

            # 4. Publish BenchmarkDataQualityWarningEvent
            self.services.event_bus.publish(
                BenchmarkDataQualityWarningEvent(
                    benchmark_id="bench_ws_01",
                    dataset_id="ds_test",
                    warning_code="TEMPORAL_GAP",
                    message="Detected sampling interval gap exceeding threshold",
                    source_id="twin_bench_ws",
                    timestamp_ns=3_000_000_000,
                )
            )
            msg_warn = ws.receive_json()
            self.assertEqual(msg_warn["type"], "twin_event")
            self.assertEqual(msg_warn["event_type"], "benchmark.quality.warning")


if __name__ == "__main__":
    unittest.main()
