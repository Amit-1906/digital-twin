"""Unit tests for degradation observations, trajectories, and trajectory builder."""

import unittest

from src.prognostics.provenance import (
    PrognosticQuantity,
    PrognosticValueProvenance,
)
from src.prognostics.state import PrognosticOperatingContext
from src.prognostics.trajectory import (
    DegradationObservation,
    DegradationTrajectory,
    TrajectoryBuilder,
)
from src.prognostics.trajectory_types import (
    TrajectoryOrderingStatus,
    TrajectoryQuantityType,
)
from src.prognostics.types import QuantityAvailability, ValueOriginType


class TestDegradationTrajectory(unittest.TestCase):
    """Test suite verifying degradation observations and trajectory container contracts."""

    def setUp(self) -> None:
        self.prov_est = PrognosticValueProvenance(
            origin_type=ValueOriginType.ESTIMATED,
            source_identifier="soh_capacity_tracker",
        )
        self.prov_meas = PrognosticValueProvenance(
            origin_type=ValueOriginType.MEASURED,
            source_identifier="lab_cycler_channel_01",
        )

    def _create_sample_obs(
        self,
        timestamp_s: float,
        capacity_ah: float = 4.8,
        r0_ohm: float = 0.025,
        efc: float = 100.0,
    ) -> DegradationObservation:
        ts_ns = int(timestamp_s * 1_000_000_000)
        return DegradationObservation(
            timestamp_ns=ts_ns,
            created_at_utc="2026-09-04T00:00:00Z",
            usable_capacity_ah=PrognosticQuantity(
                name="usable_capacity_ah",
                value=capacity_ah,
                unit="Ah",
                provenance=self.prov_est,
            ),
            internal_resistance_r0_ohm=PrognosticQuantity(
                name="internal_resistance_r0_ohm",
                value=r0_ohm,
                unit="ohm",
                provenance=self.prov_meas,
            ),
            equivalent_full_cycles=PrognosticQuantity(
                name="equivalent_full_cycles",
                value=efc,
                unit="EFC",
                provenance=self.prov_est,
            ),
        )

    def test_observation_creation_and_serialization(self) -> None:
        obs = self._create_sample_obs(10.0, capacity_ah=4.95, r0_ohm=0.022, efc=10.0)
        self.assertEqual(obs.timestamp_ns, 10_000_000_000)
        self.assertEqual(obs.get_value("usable_capacity_ah"), 4.95)
        self.assertEqual(obs.get_value("internal_resistance_r0_ohm"), 0.022)
        self.assertTrue(obs.is_quantity_available("usable_capacity_ah"))
        self.assertFalse(obs.is_quantity_available("total_energy_throughput_wh"))

        d = obs.to_dict()
        self.assertEqual(d["timestamp_ns"], 10_000_000_000)
        self.assertEqual(d["usable_capacity_ah"]["value"], 4.95)
        self.assertIn("usable_capacity_ah", d["available_quantities"])

    def test_observation_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            DegradationObservation(timestamp_ns=-1, created_at_utc="2026-09-04T00:00:00Z")

        with self.assertRaises(ValueError):
            DegradationObservation(timestamp_ns=0, created_at_utc="")

    def test_strictly_monotonic_trajectory(self) -> None:
        obs1 = self._create_sample_obs(0.0, capacity_ah=5.0, efc=0.0)
        obs2 = self._create_sample_obs(86400.0 * 10, capacity_ah=4.95, efc=50.0)
        obs3 = self._create_sample_obs(86400.0 * 20, capacity_ah=4.90, efc=100.0)

        traj = DegradationTrajectory(
            trajectory_id="traj_mono_01",
            system_id="cell_nmc_01",
            quantity_type=TrajectoryQuantityType.MULTIVARIATE,
            observations=(obs1, obs2, obs3),
        )

        self.assertEqual(traj.total_observations, 3)
        self.assertEqual(traj.duration_seconds, 86400.0 * 20)
        self.assertEqual(traj.ordering_status, TrajectoryOrderingStatus.STRICTLY_MONOTONIC)

        cap_series = traj.get_series("usable_capacity_ah")
        self.assertEqual(len(cap_series), 3)
        self.assertEqual(cap_series[0][1], 5.0)
        self.assertEqual(cap_series[2][1], 4.90)

    def test_monotonic_with_duplicates_trajectory(self) -> None:
        obs1 = self._create_sample_obs(10.0, capacity_ah=5.0)
        obs2 = self._create_sample_obs(10.0, capacity_ah=5.0)  # Duplicate timestamp
        obs3 = self._create_sample_obs(20.0, capacity_ah=4.9)

        traj = DegradationTrajectory(
            trajectory_id="traj_dup_01",
            system_id="cell_01",
            observations=(obs1, obs2, obs3),
        )
        self.assertEqual(traj.ordering_status, TrajectoryOrderingStatus.MONOTONIC_WITH_DUPLICATES)

    def test_retrograde_detected_trajectory(self) -> None:
        obs1 = self._create_sample_obs(20.0, capacity_ah=5.0)
        obs2 = self._create_sample_obs(10.0, capacity_ah=4.9)  # Retrograde timestamp

        traj = DegradationTrajectory(
            trajectory_id="traj_retro_01",
            system_id="cell_01",
            observations=(obs1, obs2),
        )
        self.assertEqual(traj.ordering_status, TrajectoryOrderingStatus.RETROGRADE_DETECTED)

    def test_empty_and_single_point_trajectories(self) -> None:
        empty_traj = DegradationTrajectory(
            trajectory_id="traj_empty",
            system_id="cell_01",
            observations=(),
        )
        self.assertEqual(empty_traj.ordering_status, TrajectoryOrderingStatus.EMPTY)
        self.assertEqual(empty_traj.duration_seconds, 0.0)

        single_traj = DegradationTrajectory(
            trajectory_id="traj_single",
            system_id="cell_01",
            observations=(self._create_sample_obs(0.0),),
        )
        self.assertEqual(single_traj.ordering_status, TrajectoryOrderingStatus.SINGLE_POINT)
        self.assertEqual(single_traj.duration_seconds, 0.0)

    def test_trajectory_builder(self) -> None:
        obs1 = self._create_sample_obs(30.0, capacity_ah=4.8)
        obs2 = self._create_sample_obs(10.0, capacity_ah=5.0)
        obs3 = self._create_sample_obs(20.0, capacity_ah=4.9)

        builder = TrajectoryBuilder(
            trajectory_id="traj_builder_01",
            system_id="sys_pack_01",
            quantity_type=TrajectoryQuantityType.CAPACITY,
        )
        builder.add_observation(obs1)
        builder.add_observations([obs2, obs3])

        # Without sorting: keeps original insertion order (retrograde)
        traj_unsorted = builder.build(sort_chronologically=False)
        self.assertEqual(traj_unsorted.ordering_status, TrajectoryOrderingStatus.RETROGRADE_DETECTED)

        # With sorting: sorted monotonically
        traj_sorted = builder.build(sort_chronologically=True)
        self.assertEqual(traj_sorted.ordering_status, TrajectoryOrderingStatus.STRICTLY_MONOTONIC)
        self.assertEqual(traj_sorted.observations[0].timestamp_ns, 10_000_000_000)
        self.assertEqual(traj_sorted.observations[2].timestamp_ns, 30_000_000_000)

    def test_trajectory_immutability(self) -> None:
        obs = self._create_sample_obs(0.0)
        traj = DegradationTrajectory(
            trajectory_id="traj_immut",
            system_id="sys_01",
            observations=(obs,),
        )
        with self.assertRaises(AttributeError):
            traj.trajectory_id = "modified"  # type: ignore


if __name__ == "__main__":
    unittest.main()
