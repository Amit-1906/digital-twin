"""Unit tests for Digital Twin Diagnostic REST API routes (Level 5.4 Phase 4)."""

import unittest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import create_default_services


class TestDiagnosticAPIRoutes(unittest.TestCase):
    """Test suite verifying diagnostic REST endpoints, bounded history, hypothesis queries, and reset isolation."""

    def setUp(self) -> None:
        self.services = create_default_services()
        self.app = create_app(services=self.services)
        self.client = TestClient(self.app)

        # 1. Register a test pack
        pack_payload = {
            "schema_version": "1.0",
            "profile_id": "pack_diag_api",
            "display_name": "Diagnostic API Test Pack",
            "chemistry": "NMC",
            "topology": {"series_count": 1, "parallel_count": 1},
            "cell_profile": {
                "cell_id": "cell_nmc",
                "chemistry": "NMC",
                "form_factor": "CYLINDRICAL",
                "nominal_voltage_v": 3.7,
                "min_voltage_v": 2.8,
                "max_voltage_v": 4.2,
                "nominal_capacity_ah": 3.0,
            },
            "ratings": {
                "nominal_pack_voltage_v": 3.7,
                "nominal_cell_voltage_v": 3.7,
                "nominal_capacity_ah": 3.0,
                "nominal_energy_wh": 11.1,
            },
            "voltage_limits": {
                "cell_min_cutoff_v": 2.8,
                "cell_max_cutoff_v": 4.2,
                "pack_min_cutoff_v": 2.8,
                "pack_max_cutoff_v": 4.2,
            },
            "current_limits": {
                "max_continuous_charge_a": 3.0,
                "max_continuous_discharge_a": 6.0,
                "peak_pulse_discharge_a": 12.0,
            },
            "thermal_limits": {
                "min_charge_temp_c": 0.0,
                "max_charge_temp_c": 45.0,
                "min_discharge_temp_c": -20.0,
                "max_discharge_temp_c": 60.0,
                "thermal_warning_temp_c": 60.0,
                "critical_thermal_runaway_temp_c": 80.0,
            },
        }
        res_pack = self.client.post("/api/v1/packs", json=pack_payload)
        self.assertEqual(res_pack.status_code, 201)

        # 2. Create digital twin with EKF
        twin_payload = {
            "system_id": "twin_diag_01",
            "pack_id": "pack_diag_api",
            "estimator_type": "EKF",
            "series_resistance_r0_ohm": 0.025,
            "r1_ohm": 0.015,
            "c1_farad": 1000.0,
            "auto_initialize": True,
            "initial_soc": 0.90,
            "initial_temperature_c": 25.0,
        }
        res_twin = self.client.post("/api/v1/twins", json=twin_payload)
        self.assertEqual(res_twin.status_code, 201)

    def test_get_unstepped_diagnostics(self) -> None:
        """GET /api/v1/twins/{system_id}/diagnostics on unstepped twin returns initial nominal state."""
        res = self.client.get("/api/v1/twins/twin_diag_01/diagnostics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["system_id"], "twin_diag_01")
        self.assertEqual(data["lifecycle_state"], "NORMAL")
        self.assertEqual(data["severity"], "NORMAL")
        self.assertEqual(data["data_quality_status"], "VALID")

    def test_get_stepped_diagnostics_and_history(self) -> None:
        """Stepping twin populates active diagnostic assessment and records bounded history."""
        # Step twin
        snap = {
            "system_id": "twin_diag_01",
            "snapshot_id": "snap_01",
            "timestamp_ns": 1_000_000_000,
            "pack_voltage_v": 3.95,
            "pack_current_a": 2.0,
            "avg_cell_temperature_c": 26.5,
            "ambient_temperature_c": 25.0,
        }
        step_res = self.client.post("/api/v1/twins/twin_diag_01/step", json=snap)
        self.assertEqual(step_res.status_code, 200)

        # Check diagnostics
        diag_res = self.client.get("/api/v1/twins/twin_diag_01/diagnostics")
        self.assertEqual(diag_res.status_code, 200)
        diag_data = diag_res.json()
        self.assertEqual(diag_data["system_id"], "twin_diag_01")
        self.assertEqual(diag_data["timestamp_ns"], 1_000_000_000)
        self.assertIn("explanation_narrative", diag_data)

        # Check history
        hist_res = self.client.get("/api/v1/twins/twin_diag_01/diagnostics/history")
        self.assertEqual(hist_res.status_code, 200)
        history = hist_res.json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["timestamp_ns"], 1_000_000_000)

    def test_get_diagnostic_hypotheses_catalog(self) -> None:
        """GET /api/v1/twins/{system_id}/diagnostics/hypotheses returns all registered root cause hypotheses."""
        res = self.client.get("/api/v1/twins/twin_diag_01/diagnostics/hypotheses")
        self.assertEqual(res.status_code, 200)
        hypotheses = res.json()
        self.assertIsInstance(hypotheses, list)
        self.assertGreaterEqual(len(hypotheses), 5)
        hyp_ids = [h["hypothesis_id"] for h in hypotheses]
        self.assertIn("HYP_SENSOR_DRIFT", hyp_ids)
        self.assertIn("HYP_MODEL_MISMATCH", hyp_ids)
        self.assertIn("HYP_APPARENT_OHMIC_GROWTH", hyp_ids)
        self.assertIn("HYP_THERMAL_IMPAIRMENT", hyp_ids)

    def test_missing_twin_diagnostic_endpoints_return_404(self) -> None:
        """Querying diagnostics or hypotheses for non-existent twin returns HTTP 404."""
        self.assertEqual(self.client.get("/api/v1/twins/non_existent/diagnostics").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/twins/non_existent/diagnostics/history").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/twins/non_existent/diagnostics/hypotheses").status_code, 404)
        self.assertEqual(self.client.post("/api/v1/twins/non_existent/diagnostics/reset").status_code, 404)

    def test_diagnostic_reset_strict_isolation_from_model_and_estimator_state(self) -> None:
        """POST /diagnostics/reset strictly resets only diagnostic state and history, leaving model/EKF intact."""
        # 1. Step twin multiple times to advance simulation, estimator, and validation state
        for i in range(1, 4):
            snap = {
                "system_id": "twin_diag_01",
                "snapshot_id": f"snap_{i}",
                "timestamp_ns": i * 1_000_000_000,
                "pack_voltage_v": 3.90,
                "pack_current_a": 1.5,
                "avg_cell_temperature_c": 26.0,
            }
            self.client.post("/api/v1/twins/twin_diag_01/step", json=snap)

        # Check pre-reset state
        status_before = self.client.get("/api/v1/twins/twin_diag_01").json()
        self.assertEqual(status_before["total_steps"], 3)
        soc_before = status_before["current_soc"]
        self.assertIsNotNone(soc_before)

        hist_before = self.client.get("/api/v1/twins/twin_diag_01/diagnostics/history").json()
        self.assertEqual(len(hist_before), 3)

        # 2. Issue diagnostic reset
        reset_res = self.client.post("/api/v1/twins/twin_diag_01/diagnostics/reset")
        self.assertEqual(reset_res.status_code, 200)
        reset_data = reset_res.json()
        self.assertEqual(reset_data["status"], "RESET_SUCCESSFUL")
        self.assertEqual(reset_data["subsystem"], "DIAGNOSTICS")

        # 3. Verify diagnostic state is reset and history is cleared
        hist_after = self.client.get("/api/v1/twins/twin_diag_01/diagnostics/history").json()
        self.assertEqual(len(hist_after), 0)

        # 4. Verify model and runtime counters were NOT reset
        status_after = self.client.get("/api/v1/twins/twin_diag_01").json()
        self.assertEqual(status_after["total_steps"], 3)
        self.assertEqual(status_after["current_soc"], soc_before)


if __name__ == "__main__":
    unittest.main()
