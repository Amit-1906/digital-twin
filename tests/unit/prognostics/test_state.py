"""Unit tests for prognostic state snapshots and operating context representations."""

import unittest

from src.prognostics.provenance import (
    PrognosticQuantity,
    PrognosticValueProvenance,
)
from src.prognostics.state import (
    PrognosticOperatingContext,
    PrognosticStateSnapshot,
)
from src.prognostics.types import QuantityAvailability, ValueOriginType


class TestPrognosticState(unittest.TestCase):
    """Test suite verifying prognostic state snapshots and operating context contracts."""

    def setUp(self) -> None:
        self.prov_meas = PrognosticValueProvenance(origin_type=ValueOriginType.MEASURED)
        self.prov_est = PrognosticValueProvenance(origin_type=ValueOriginType.ESTIMATED)
        self.prov_der = PrognosticValueProvenance(origin_type=ValueOriginType.DERIVED)

    def test_prognostic_state_creation_with_full_quantities(self) -> None:
        ctx = PrognosticOperatingContext(
            ambient_temperature_c=25.0,
            cell_temperature_c=28.5,
            pack_current_a=-5.0,
            c_rate=1.0,
            soc_fraction=0.85,
            pack_voltage_v=3.92,
            load_profile_identifier="DYNAMIC_URBAN_CYCLE",
            duty_cycle_fraction=0.60,
        )

        snapshot = PrognosticStateSnapshot(
            snapshot_id="snap_prog_001",
            system_id="battery_sys_alpha",
            timestamp_ns=1_000_000_000,
            created_at_utc="2026-09-04T12:00:00Z",
            usable_capacity_ah=PrognosticQuantity(
                name="usable_capacity_ah",
                value=4.80,
                unit="Ah",
                provenance=self.prov_est,
            ),
            nominal_capacity_ah=PrognosticQuantity(
                name="nominal_capacity_ah",
                value=5.00,
                unit="Ah",
                provenance=self.prov_meas,
            ),
            capacity_retention_fraction=PrognosticQuantity(
                name="capacity_retention_fraction",
                value=0.96,
                unit="fraction",
                provenance=self.prov_der,
            ),
            internal_resistance_r0_ohm=PrognosticQuantity(
                name="internal_resistance_r0_ohm",
                value=0.024,
                unit="ohm",
                provenance=self.prov_est,
            ),
            baseline_resistance_r0_ohm=PrognosticQuantity(
                name="baseline_resistance_r0_ohm",
                value=0.020,
                unit="ohm",
                provenance=self.prov_meas,
            ),
            resistance_growth_ratio=PrognosticQuantity(
                name="resistance_growth_ratio",
                value=1.20,
                unit="ratio",
                provenance=self.prov_der,
            ),
            equivalent_full_cycles=PrognosticQuantity(
                name="equivalent_full_cycles",
                value=150.5,
                unit="EFC",
                provenance=self.prov_der,
            ),
            charge_throughput_ah=PrognosticQuantity(
                name="charge_throughput_ah",
                value=752.5,
                unit="Ah",
                provenance=self.prov_der,
            ),
            discharge_throughput_ah=PrognosticQuantity(
                name="discharge_throughput_ah",
                value=752.0,
                unit="Ah",
                provenance=self.prov_der,
            ),
            total_energy_throughput_wh=PrognosticQuantity(
                name="total_energy_throughput_wh",
                value=5560.0,
                unit="Wh",
                provenance=self.prov_der,
            ),
            calendar_age_seconds=PrognosticQuantity(
                name="calendar_age_seconds",
                value=86400.0 * 180.0,
                unit="seconds",
                provenance=self.prov_der,
            ),
            cumulative_thermal_exposure_deg_c_hours=PrognosticQuantity(
                name="cumulative_thermal_exposure_deg_c_hours",
                value=4500.0,
                unit="degC_hours",
                provenance=self.prov_der,
            ),
            operating_context=ctx,
        )

        self.assertEqual(snapshot.snapshot_id, "snap_prog_001")
        self.assertEqual(snapshot.system_id, "battery_sys_alpha")
        self.assertTrue(snapshot.is_quantity_available("usable_capacity_ah"))
        self.assertTrue(snapshot.is_quantity_available("capacity_retention_fraction"))
        self.assertTrue(snapshot.is_quantity_available("internal_resistance_r0_ohm"))
        self.assertEqual(len(snapshot.available_quantities()), 12)

        d = snapshot.to_dict()
        self.assertEqual(d["snapshot_id"], "snap_prog_001")
        self.assertEqual(d["usable_capacity_ah"]["value"], 4.8)
        self.assertEqual(d["operating_context"]["ambient_temperature_c"], 25.0)
        self.assertEqual(len(d["available_quantities"]), 12)

    def test_prognostic_state_with_sparse_optional_quantities(self) -> None:
        # Scenario: Minimal field telemetry with only capacity and EFC available
        snapshot = PrognosticStateSnapshot(
            snapshot_id="snap_sparse_001",
            system_id="lfp_cell_04",
            timestamp_ns=500_000_000,
            created_at_utc="2026-09-04T12:00:00Z",
            usable_capacity_ah=PrognosticQuantity(
                name="usable_capacity_ah",
                value=100.0,
                unit="Ah",
                provenance=self.prov_est,
            ),
            equivalent_full_cycles=PrognosticQuantity(
                name="equivalent_full_cycles",
                value=42.0,
                unit="EFC",
                provenance=self.prov_der,
            ),
            internal_resistance_r0_ohm=PrognosticQuantity(
                name="internal_resistance_r0_ohm",
                value=None,
                unit="ohm",
                availability=QuantityAvailability.UNAVAILABLE,
            ),
        )

        self.assertTrue(snapshot.is_quantity_available("usable_capacity_ah"))
        self.assertTrue(snapshot.is_quantity_available("equivalent_full_cycles"))
        self.assertFalse(snapshot.is_quantity_available("internal_resistance_r0_ohm"))
        self.assertFalse(snapshot.is_quantity_available("cumulative_thermal_exposure_deg_c_hours"))

        avail = snapshot.available_quantities()
        self.assertIn("usable_capacity_ah", avail)
        self.assertIn("equivalent_full_cycles", avail)
        self.assertNotIn("internal_resistance_r0_ohm", avail)

    def test_operating_context_validation(self) -> None:
        with self.assertRaises(ValueError):
            PrognosticOperatingContext(ambient_temperature_c=float("nan"))

        with self.assertRaises(ValueError):
            PrognosticOperatingContext(pack_current_a=float("inf"))

    def test_state_snapshot_validation(self) -> None:
        with self.assertRaises(ValueError):
            PrognosticStateSnapshot(
                snapshot_id="",
                system_id="sys",
                timestamp_ns=0,
                created_at_utc="2026-09-04T00:00:00Z",
            )

        with self.assertRaises(ValueError):
            PrognosticStateSnapshot(
                snapshot_id="id",
                system_id="",
                timestamp_ns=0,
                created_at_utc="2026-09-04T00:00:00Z",
            )

        with self.assertRaises(ValueError):
            PrognosticStateSnapshot(
                snapshot_id="id",
                system_id="sys",
                timestamp_ns=-1,
                created_at_utc="2026-09-04T00:00:00Z",
            )

    def test_state_snapshot_immutability(self) -> None:
        snapshot = PrognosticStateSnapshot(
            snapshot_id="snap_immut",
            system_id="sys",
            timestamp_ns=0,
            created_at_utc="2026-09-04T00:00:00Z",
        )
        with self.assertRaises(AttributeError):
            snapshot.system_id = "modified"  # type: ignore


if __name__ == "__main__":
    unittest.main()
