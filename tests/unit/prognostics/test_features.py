"""Unit tests for prognostic feature extraction, rates, trends, and operating context metrics."""

import unittest

from src.prognostics.features import (
    DegradationFeatureSet,
    PrognosticFeature,
    TrajectoryFeatureExtractor,
)
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
    FeatureQualityStatus,
    RateDomain,
    TrajectoryQuantityType,
)
from src.prognostics.types import ValueOriginType


class TestPrognosticFeatures(unittest.TestCase):
    """Test suite verifying descriptive degradation feature extraction."""

    def setUp(self) -> None:
        self.prov = PrognosticValueProvenance(origin_type=ValueOriginType.ESTIMATED)
        self.extractor = TrajectoryFeatureExtractor(
            min_observations=2,
            rolling_window_size=3,
        )

    def _build_synthetic_trajectory(
        self,
        num_points: int = 10,
        initial_capacity: float = 5.0,
        capacity_fade_per_day: float = 0.005,
        initial_r0: float = 0.020,
        r0_growth_per_day: float = 0.0002,
        time_step_days: float = 5.0,
    ) -> DegradationTrajectory:
        builder = TrajectoryBuilder(
            trajectory_id="traj_synthetic_01",
            system_id="cell_test_01",
            quantity_type=TrajectoryQuantityType.MULTIVARIATE,
        )

        for i in range(num_points):
            t_days = i * time_step_days
            ts_ns = int(t_days * 86400.0 * 1_000_000_000)
            cap = initial_capacity - i * (capacity_fade_per_day * time_step_days)
            r0 = initial_r0 + i * (r0_growth_per_day * time_step_days)
            efc = float(i * 20)
            ah_thru = float(i * 100.0)

            ctx = PrognosticOperatingContext(
                ambient_temperature_c=25.0 + (i % 3),
                cell_temperature_c=28.0 + (i % 5),
                pack_current_a=2.5,
                soc_fraction=0.80 - (i * 0.01),
                pack_voltage_v=3.85,
            )

            obs = DegradationObservation(
                timestamp_ns=ts_ns,
                created_at_utc="2026-09-04T00:00:00Z",
                usable_capacity_ah=PrognosticQuantity(
                    name="usable_capacity_ah",
                    value=cap,
                    unit="Ah",
                    provenance=self.prov,
                ),
                internal_resistance_r0_ohm=PrognosticQuantity(
                    name="internal_resistance_r0_ohm",
                    value=r0,
                    unit="ohm",
                    provenance=self.prov,
                ),
                equivalent_full_cycles=PrognosticQuantity(
                    name="equivalent_full_cycles",
                    value=efc,
                    unit="EFC",
                    provenance=self.prov,
                ),
                discharge_throughput_ah=PrognosticQuantity(
                    name="discharge_throughput_ah",
                    value=ah_thru,
                    unit="Ah",
                    provenance=self.prov,
                ),
                operating_context=ctx,
            )
            builder.add_observation(obs)

        return builder.build()

    def test_prognostic_feature_value_object(self) -> None:
        feat = PrognosticFeature(
            feature_name="capacity_fade_rate_per_day",
            value=0.005,
            unit="Ah/day",
            quality_status=FeatureQualityStatus.VALID,
            rate_domain=RateDomain.PER_DAY,
            calculation_method="ENDPOINT_RATE",
            source_quantity="usable_capacity_ah",
            source_observation_count=10,
        )
        self.assertTrue(feat.is_valid)
        self.assertEqual(feat.value, 0.005)
        self.assertEqual(feat.unit, "Ah/day")

        d = feat.to_dict()
        self.assertEqual(d["feature_name"], "capacity_fade_rate_per_day")
        self.assertEqual(d["value"], 0.005)
        self.assertEqual(d["quality_status"], "VALID")
        self.assertEqual(d["rate_domain"], "PER_DAY")

    def test_prognostic_feature_validation_invariants(self) -> None:
        with self.assertRaises(ValueError):
            # VALID status cannot have None value
            PrognosticFeature(
                feature_name="test_feat",
                value=None,
                unit="Ah",
                quality_status=FeatureQualityStatus.VALID,
            )

        with self.assertRaises(ValueError):
            # NaN is prohibited
            PrognosticFeature(
                feature_name="test_feat",
                value=float("nan"),
                unit="Ah",
                quality_status=FeatureQualityStatus.VALID,
            )

    def test_feature_extraction_pipeline_endpoints_and_deltas(self) -> None:
        traj = self._build_synthetic_trajectory(
            num_points=10,
            initial_capacity=5.0,
            capacity_fade_per_day=0.005,
            time_step_days=5.0,
        )
        # Total span: 9 steps * 5 days = 45 days.
        # Total fade: 9 * (0.005 * 5) = 0.225 Ah.
        # Final capacity: 5.0 - 0.225 = 4.775 Ah.

        feat_set = self.extractor.extract_features(traj)
        self.assertIsInstance(feat_set, DegradationFeatureSet)
        self.assertEqual(feat_set.system_id, "cell_test_01")

        # Elapsed time
        self.assertAlmostEqual(feat_set.get_value("elapsed_time_days") or 0.0, 45.0, places=4)

        # Capacity fade endpoint delta
        self.assertAlmostEqual(feat_set.get_value("usable_capacity_ah_delta") or 0.0, 0.225, places=4)
        self.assertAlmostEqual(feat_set.get_value("usable_capacity_ah_initial_value") or 0.0, 5.0, places=4)
        self.assertAlmostEqual(feat_set.get_value("usable_capacity_ah_latest_value") or 0.0, 4.775, places=4)

        # Resistance growth endpoint delta
        # Initial R0 = 0.020. Growth per day = 0.0002. Total growth = 45 * 0.0002 = 0.009.
        self.assertAlmostEqual(feat_set.get_value("internal_resistance_r0_ohm_delta") or 0.0, 0.009, places=5)
        self.assertAlmostEqual(feat_set.get_value("internal_resistance_r0_ohm_initial_value") or 0.0, 0.020, places=5)
        self.assertAlmostEqual(feat_set.get_value("internal_resistance_r0_ohm_latest_value") or 0.0, 0.029, places=5)

    def test_degradation_rates_per_domain(self) -> None:
        traj = self._build_synthetic_trajectory(
            num_points=10,
            initial_capacity=5.0,
            capacity_fade_per_day=0.005,
            time_step_days=5.0,
        )
        feat_set = self.extractor.extract_features(traj)

        # Rate per day: 0.225 Ah / 45 days = 0.005 Ah/day
        rate_day = feat_set.get_value("usable_capacity_ah_rate_per_day")
        self.assertIsNotNone(rate_day)
        self.assertAlmostEqual(rate_day or 0.0, 0.005, places=5)

        # Rate per EFC: Total EFC = 9 * 20 = 180 EFC. Rate = 0.225 Ah / 180 EFC = 0.00125 Ah/EFC
        rate_efc = feat_set.get_value("usable_capacity_ah_rate_per_efc")
        self.assertIsNotNone(rate_efc)
        self.assertAlmostEqual(rate_efc or 0.0, 0.00125, places=5)

        # Rate per Ah throughput: Total Ah = 9 * 100 = 900 Ah. Rate = 0.225 / 900 = 0.00025 Ah/Ah
        rate_ah = feat_set.get_value("usable_capacity_ah_rate_per_ah")
        self.assertIsNotNone(rate_ah)
        self.assertAlmostEqual(rate_ah or 0.0, 0.00025, places=5)

    def test_ols_trend_and_rolling_slope(self) -> None:
        traj = self._build_synthetic_trajectory(
            num_points=10,
            initial_capacity=5.0,
            capacity_fade_per_day=0.005,
            time_step_days=5.0,
        )
        feat_set = self.extractor.extract_features(traj)

        # Capacity OLS slope (negative since capacity is fading): -0.005 Ah/day
        ols_slope = feat_set.get_value("usable_capacity_ah_ols_slope_per_day")
        self.assertIsNotNone(ols_slope)
        self.assertAlmostEqual(ols_slope or 0.0, -0.005, places=4)

        # R^2 for purely linear synthetic line should be 1.0
        r_sq = feat_set.get_value("usable_capacity_ah_ols_r_squared")
        self.assertIsNotNone(r_sq)
        self.assertAlmostEqual(r_sq or 0.0, 1.0, places=4)

        # Rolling slope recent
        rolling_slope = feat_set.get_value("usable_capacity_ah_rolling_slope_recent_per_day")
        self.assertIsNotNone(rolling_slope)
        self.assertAlmostEqual(rolling_slope or 0.0, -0.005, places=4)

    def test_zero_denominator_guard(self) -> None:
        # Single timestamp with 2 observations (duration = 0)
        obs1 = DegradationObservation(
            timestamp_ns=1000,
            created_at_utc="2026-09-04T00:00:00Z",
            usable_capacity_ah=PrognosticQuantity(name="cap", value=5.0, unit="Ah"),
            equivalent_full_cycles=PrognosticQuantity(name="efc", value=10.0, unit="EFC"),
        )
        obs2 = DegradationObservation(
            timestamp_ns=1000,
            created_at_utc="2026-09-04T00:00:00Z",
            usable_capacity_ah=PrognosticQuantity(name="cap", value=4.9, unit="Ah"),
            equivalent_full_cycles=PrognosticQuantity(name="efc", value=10.0, unit="EFC"),
        )

        traj = DegradationTrajectory(
            trajectory_id="traj_zero_den",
            system_id="cell_01",
            observations=(obs1, obs2),
        )

        feat_set = self.extractor.extract_features(traj)
        rate_feat = feat_set.get_feature("usable_capacity_ah_rate_per_second")
        self.assertIsNotNone(rate_feat)
        self.assertEqual(rate_feat.quality_status, FeatureQualityStatus.ZERO_DENOMINATOR)
        self.assertIsNone(rate_feat.value)
        self.assertFalse(rate_feat.is_valid)

        efc_feat = feat_set.get_feature("usable_capacity_ah_rate_per_efc")
        self.assertIsNotNone(efc_feat)
        self.assertEqual(efc_feat.quality_status, FeatureQualityStatus.ZERO_DENOMINATOR)
        self.assertIsNone(efc_feat.value)

    def test_insufficient_data_handling(self) -> None:
        # Single observation when min_observations is 2
        obs = DegradationObservation(
            timestamp_ns=1000,
            created_at_utc="2026-09-04T00:00:00Z",
            usable_capacity_ah=PrognosticQuantity(name="cap", value=5.0, unit="Ah"),
        )
        traj = DegradationTrajectory(
            trajectory_id="traj_single",
            system_id="cell_01",
            observations=(obs,),
        )

        feat_set = self.extractor.extract_features(traj)
        delta_feat = feat_set.get_feature("usable_capacity_ah_delta")
        self.assertIsNotNone(delta_feat)
        self.assertEqual(delta_feat.quality_status, FeatureQualityStatus.INSUFFICIENT_DATA)
        self.assertIsNone(delta_feat.value)

    def test_operating_context_statistics(self) -> None:
        traj = self._build_synthetic_trajectory(num_points=10)
        feat_set = self.extractor.extract_features(traj)

        mean_cell_temp = feat_set.get_value("mean_cell_temperature_c")
        self.assertIsNotNone(mean_cell_temp)
        self.assertTrue(28.0 <= (mean_cell_temp or 0.0) <= 33.0)

        mean_curr = feat_set.get_value("mean_absolute_current_a")
        self.assertEqual(mean_curr, 2.5)

        mean_soc = feat_set.get_value("mean_soc_fraction")
        self.assertIsNotNone(mean_soc)

    def test_deterministic_feature_extraction(self) -> None:
        traj = self._build_synthetic_trajectory(num_points=8)
        fs1 = self.extractor.extract_features(traj, feature_set_id="fs_test_fixed")
        fs2 = self.extractor.extract_features(traj, feature_set_id="fs_test_fixed")

        d1 = fs1.to_dict()
        d2 = fs2.to_dict()
        # Exclude timestamp for strict value equivalence
        d1.pop("extracted_at_utc")
        d2.pop("extracted_at_utc")
        self.assertEqual(d1, d2)

    def test_capacity_and_resistance_separation(self) -> None:
        traj = self._build_synthetic_trajectory(num_points=5)
        feat_set = self.extractor.extract_features(traj)

        # Confirm distinct feature keys for capacity and resistance
        self.assertIn("usable_capacity_ah_delta", feat_set.features)
        self.assertIn("internal_resistance_r0_ohm_delta", feat_set.features)
        self.assertIn("usable_capacity_ah_rate_per_day", feat_set.features)
        self.assertIn("internal_resistance_r0_ohm_rate_per_day", feat_set.features)


if __name__ == "__main__":
    unittest.main()
