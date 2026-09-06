"""Unit tests for the Historical Linear EOL-Crossing RUL Estimator."""

import unittest
from typing import Optional

from src.prognostics.estimators.base import EstimatorConfig
from src.prognostics.estimators.linear import HistoricalLinearRULEstimator
from src.prognostics.horizon import PrognosticHorizon
from src.prognostics.prediction import EOLDefinition
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
from src.prognostics.trajectory_types import TrajectoryQuantityType
from src.prognostics.types import (
    ComparisonOperator,
    EOLCriterionType,
    HorizonDomain,
    PrognosticStatus,
    ValueOriginType,
)


class TestHistoricalLinearRULEstimator(unittest.TestCase):
    """Comprehensive test suite for HistoricalLinearRULEstimator."""

    def setUp(self) -> None:
        self.estimator = HistoricalLinearRULEstimator()
        self.prov = PrognosticValueProvenance(origin_type=ValueOriginType.ESTIMATED)

    def _create_observation(
        self,
        efc: float,
        timestamp_s: float,
        capacity_ah: Optional[float] = None,
        soh_c: Optional[float] = None,
        r0_ohm: Optional[float] = None,
        r_ratio: Optional[float] = None,
        r0_baseline: Optional[float] = None,
        ah_throughput: Optional[float] = None,
        wh_throughput: Optional[float] = None,
        temp_c: float = 25.0,
    ) -> DegradationObservation:
        ts_ns = int(timestamp_s * 1_000_000_000)
        obs_meta = {}
        if r0_baseline is not None:
            obs_meta["baseline_resistance_r0_ohm"] = r0_baseline

        return DegradationObservation(
            timestamp_ns=ts_ns,
            created_at_utc="2026-09-04T00:00:00Z",
            usable_capacity_ah=(
                PrognosticQuantity(name="usable_capacity_ah", value=capacity_ah, unit="Ah", provenance=self.prov)
                if capacity_ah is not None
                else None
            ),
            capacity_retention_fraction=(
                PrognosticQuantity(name="capacity_retention_fraction", value=soh_c, unit="fraction", provenance=self.prov)
                if soh_c is not None
                else None
            ),
            internal_resistance_r0_ohm=(
                PrognosticQuantity(name="internal_resistance_r0_ohm", value=r0_ohm, unit="ohm", provenance=self.prov)
                if r0_ohm is not None
                else None
            ),
            resistance_growth_ratio=(
                PrognosticQuantity(name="resistance_growth_ratio", value=r_ratio, unit="ratio", provenance=self.prov)
                if r_ratio is not None
                else None
            ),
            equivalent_full_cycles=PrognosticQuantity(
                name="equivalent_full_cycles",
                value=efc,
                unit="EFC",
                provenance=self.prov,
            ),
            discharge_throughput_ah=(
                PrognosticQuantity(name="discharge_throughput_ah", value=ah_throughput, unit="Ah", provenance=self.prov)
                if ah_throughput is not None
                else None
            ),
            total_energy_throughput_wh=(
                PrognosticQuantity(name="total_energy_throughput_wh", value=wh_throughput, unit="Wh", provenance=self.prov)
                if wh_throughput is not None
                else None
            ),
            operating_context=PrognosticOperatingContext(cell_temperature_c=temp_c, pack_current_a=1.0),
            metadata=obs_meta,
        )

    # 1. Exact Synthetic Mathematical Verification (Part 23)
    def test_exact_synthetic_mathematical_example(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=3.00)
        obs2 = self._create_observation(efc=40.0, timestamp_s=86400.0 * 20, capacity_ah=2.60)

        traj = DegradationTrajectory(
            trajectory_id="traj_math_01",
            system_id="cell_math_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_240",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=2.40,
            unit="Ah",
            comparison_operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
        )

        horizon = PrognosticHorizon(domain=HorizonDomain.CYCLE, horizon_value=100.0, unit="cycles")
        pred = self.estimator.estimate(traj, eol, horizon)

        self.assertTrue(pred.is_valid)
        self.assertEqual(pred.status, PrognosticStatus.VALID)
        self.assertAlmostEqual(pred.rul_value or 0.0, 20.0, places=5)
        self.assertEqual(pred.rul_unit, "cycles")
        self.assertEqual(pred.rul_domain, HorizonDomain.CYCLE)
        self.assertIsNone(pred.uncertainty)  # Uncertainty not modeled in Level 6.3

        meta = pred.reproducibility_metadata
        self.assertAlmostEqual(meta["last_observation_domain_value"], 40.0, places=4)
        self.assertAlmostEqual(meta["predicted_eol_domain_value"], 60.0, places=4)
        self.assertAlmostEqual(meta["degradation_rate"], 0.01, places=5)
        self.assertEqual(meta["degradation_rate_unit"], "Ah/cycles")

    # 2. Causality and Future-Data Leakage Isolation Test
    def test_causality_and_future_data_isolation(self) -> None:
        """Verifies that future observations past cutoff X do not contaminate prediction at cutoff X."""
        # Baseline historical data through cutoff X (t = 20 days, EFC = 40, Capacity = 2.60 Ah)
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=3.00)
        obs2 = self._create_observation(efc=40.0, timestamp_s=86400.0 * 20, capacity_ah=2.60)
        # Additional future observation at t = 50 days (EFC = 100, Capacity = 1.90 Ah - steep drop)
        obs_future = self._create_observation(efc=100.0, timestamp_s=86400.0 * 50, capacity_ah=1.90)

        traj_historical = DegradationTrajectory(
            trajectory_id="traj_hist",
            system_id="cell_causality",
            observations=(obs1, obs2),
        )
        traj_extended = DegradationTrajectory(
            trajectory_id="traj_ext",
            system_id="cell_causality",
            observations=(obs1, obs2, obs_future),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_240",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=2.40,
            unit="Ah",
        )

        cutoff_ts_ns = int(86400.0 * 20 * 1_000_000_000)
        cfg_cutoff = EstimatorConfig(prediction_cutoff_timestamp_ns=cutoff_ts_ns)

        # 1. Prediction on purely historical trajectory
        pred_hist = self.estimator.estimate(traj_historical, eol)
        # 2. Prediction on extended trajectory with explicit cutoff at X
        pred_ext_with_cutoff = self.estimator.estimate(traj_extended, eol, config=cfg_cutoff)

        # Both predictions MUST be identical
        self.assertTrue(pred_hist.is_valid)
        self.assertTrue(pred_ext_with_cutoff.is_valid)
        self.assertEqual(pred_hist.rul_value, pred_ext_with_cutoff.rul_value)
        self.assertAlmostEqual(pred_ext_with_cutoff.rul_value or 0.0, 20.0, places=5)
        self.assertEqual(
            pred_hist.reproducibility_metadata["degradation_rate"],
            pred_ext_with_cutoff.reproducibility_metadata["degradation_rate"],
        )
        self.assertEqual(pred_ext_with_cutoff.reproducibility_metadata["source_observation_count"], 2)

    # 3. Capacity Retention Threshold (SOH_C <= 0.80) Mapping
    def test_capacity_retention_soh_threshold(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, soh_c=1.00)
        obs2 = self._create_observation(efc=200.0, timestamp_s=86400.0 * 50, soh_c=0.90)

        traj = DegradationTrajectory(
            trajectory_id="traj_soh_01",
            system_id="cell_soh_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_soh_80",
            criterion_type=EOLCriterionType.CAPACITY_RETENTION_THRESHOLD,
            threshold_value=0.80,
            unit="fraction",
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertTrue(pred.is_valid)
        self.assertAlmostEqual(pred.rul_value or 0.0, 200.0, places=4)
        self.assertEqual(pred.rul_domain, HorizonDomain.CYCLE)

    # 4. Absolute Internal Resistance Threshold (R0 >= 0.050 Ohm) Mapping
    def test_absolute_internal_resistance_threshold(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, r0_ohm=0.020)
        obs2 = self._create_observation(efc=100.0, timestamp_s=86400.0 * 25, r0_ohm=0.026)

        traj = DegradationTrajectory(
            trajectory_id="traj_r0_01",
            system_id="cell_r0_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_r0_50",
            criterion_type=EOLCriterionType.ABSOLUTE_RESISTANCE_THRESHOLD,
            threshold_value=0.050,
            unit="ohm",
            comparison_operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertTrue(pred.is_valid)
        self.assertAlmostEqual(pred.rul_value or 0.0, 400.0, places=4)

    # 5. Resistance Growth Ratio Threshold (with explicit resistance_growth_ratio)
    def test_resistance_growth_ratio_threshold_explicit(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, r_ratio=1.00)
        obs2 = self._create_observation(efc=100.0, timestamp_s=86400.0 * 25, r_ratio=1.25)

        traj = DegradationTrajectory(
            trajectory_id="traj_rratio_01",
            system_id="cell_rratio_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_rratio_2x",
            criterion_type=EOLCriterionType.RESISTANCE_GROWTH_RATIO_THRESHOLD,
            threshold_value=2.00,
            unit="ratio",
            comparison_operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertTrue(pred.is_valid)
        self.assertAlmostEqual(pred.rul_value or 0.0, 300.0, places=4)

    # 6. Resistance Growth Ratio Threshold with Explicit Baseline Resistance
    def test_resistance_growth_ratio_derived_from_explicit_baseline(self) -> None:
        """Verifies ratio is computed from R0 / R0_baseline when explicit baseline is provided."""
        # Baseline R0 = 0.020 Ohm recorded on observations
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, r0_ohm=0.020, r0_baseline=0.020)
        obs2 = self._create_observation(efc=100.0, timestamp_s=86400.0 * 25, r0_ohm=0.025, r0_baseline=0.020)

        traj = DegradationTrajectory(
            trajectory_id="traj_rratio_base",
            system_id="cell_rratio_base",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_rratio_2x",
            criterion_type=EOLCriterionType.RESISTANCE_GROWTH_RATIO_THRESHOLD,
            threshold_value=2.00,
            unit="ratio",
            comparison_operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        )

        # Ratio at 0: 0.020/0.020 = 1.00; Ratio at 100: 0.025/0.020 = 1.25
        # Rate: 0.25 / 100 = 0.0025 per EFC -> RUL to 2.00: 0.75 / 0.0025 = 300 EFC
        pred = self.estimator.estimate(traj, eol)
        self.assertTrue(pred.is_valid)
        self.assertAlmostEqual(pred.rul_value or 0.0, 300.0, places=4)

    # 7. Resistance Growth Ratio without Baseline Returns INSUFFICIENT_DATA (No Arbitrary Guessing)
    def test_resistance_growth_ratio_without_baseline_returns_insufficient_data(self) -> None:
        """Verifies that missing baseline resistance never arbitrarily assumes sample 0 is baseline."""
        # Only R0 present, but neither r_ratio nor r0_baseline is provided
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, r0_ohm=0.020)
        obs2 = self._create_observation(efc=100.0, timestamp_s=86400.0 * 25, r0_ohm=0.025)

        traj = DegradationTrajectory(
            trajectory_id="traj_no_base",
            system_id="cell_no_base",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_rratio_2x",
            criterion_type=EOLCriterionType.RESISTANCE_GROWTH_RATIO_THRESHOLD,
            threshold_value=2.00,
            unit="ratio",
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertFalse(pred.is_valid)
        self.assertEqual(pred.status, PrognosticStatus.INSUFFICIENT_DATA)
        self.assertIsNone(pred.rul_value)

    # 8. Time-Domain RUL Estimation
    def test_time_domain_rul_estimation(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=5.00)
        obs2 = self._create_observation(efc=0.0, timestamp_s=86400.0 * 100, capacity_ah=4.80)

        traj = DegradationTrajectory(
            trajectory_id="traj_time_01",
            system_id="cell_time_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_400",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=4.00,
            unit="Ah",
        )

        horizon = PrognosticHorizon(domain=HorizonDomain.TIME, horizon_value=500.0, unit="days")
        pred = self.estimator.estimate(traj, eol, horizon)

        self.assertTrue(pred.is_valid)
        self.assertAlmostEqual(pred.rul_value or 0.0, 400.0, places=4)
        self.assertEqual(pred.rul_domain, HorizonDomain.TIME)
        self.assertEqual(pred.rul_unit, "days")

    # 9. Ah Throughput Domain RUL Estimation
    def test_ah_throughput_domain_rul_estimation(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=5.00, ah_throughput=0.0)
        obs2 = self._create_observation(efc=50.0, timestamp_s=86400.0 * 20, capacity_ah=4.90, ah_throughput=500.0)

        traj = DegradationTrajectory(
            trajectory_id="traj_ah_01",
            system_id="cell_ah_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_400",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=4.00,
            unit="Ah",
        )

        horizon = PrognosticHorizon(domain=HorizonDomain.AMPERE_HOUR_THROUGHPUT, horizon_value=10000.0, unit="Ah")
        pred = self.estimator.estimate(traj, eol, horizon)

        self.assertTrue(pred.is_valid)
        self.assertAlmostEqual(pred.rul_value or 0.0, 4500.0, places=4)
        self.assertEqual(pred.rul_domain, HorizonDomain.AMPERE_HOUR_THROUGHPUT)

    # 10. Forecast Domain / Unit Consistency (No Implicit Dimensional Conversion)
    def test_incompatible_domain_channel_returns_insufficient_data(self) -> None:
        """Verifies that requesting an unpopulated forecast domain produces an explicit non-valid result without conversion."""
        # Trajectory has EFC and capacity, but NO energy throughput (Wh)
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=5.00)
        obs2 = self._create_observation(efc=50.0, timestamp_s=86400.0 * 20, capacity_ah=4.90)

        traj = DegradationTrajectory(
            trajectory_id="traj_no_wh",
            system_id="cell_no_wh",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_400",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=4.00,
            unit="Ah",
        )

        # Explicitly request Wh throughput horizon
        horizon = PrognosticHorizon(domain=HorizonDomain.WATT_HOUR_THROUGHPUT, horizon_value=10000.0, unit="Wh")
        pred = self.estimator.estimate(traj, eol, horizon)

        self.assertFalse(pred.is_valid)
        self.assertEqual(pred.status, PrognosticStatus.INSUFFICIENT_DATA)
        self.assertIsNone(pred.rul_value)
        self.assertIn("No observation points found", pred.diagnostic_message or "")

    # 11. Already At or Beyond Configured EOL (RUL = 0.0) Semantics Audit
    def test_already_at_or_beyond_eol_exact_semantics(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=3.00)
        obs2 = self._create_observation(efc=50.0, timestamp_s=86400.0 * 20, capacity_ah=2.30)

        traj = DegradationTrajectory(
            trajectory_id="traj_beyond_01",
            system_id="cell_beyond_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_240",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=2.40,
            unit="Ah",
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertTrue(pred.is_valid)
        self.assertEqual(pred.status, PrognosticStatus.VALID)
        self.assertEqual(pred.rul_value, 0.0)
        self.assertEqual(pred.reproducibility_metadata["condition"], "ALREADY_AT_OR_BEYOND_EOL")
        self.assertEqual(pred.reproducibility_metadata["extrapolation_distance"], 0.0)
        self.assertEqual(pred.reproducibility_metadata["extrapolation_ratio"], 0.0)

    # 12. Extrapolation Metadata Audit
    def test_extrapolation_metadata_correctness_and_isolation(self) -> None:
        obs1 = self._create_observation(efc=10.0, timestamp_s=0.0, capacity_ah=3.00)
        obs2 = self._create_observation(efc=50.0, timestamp_s=86400.0 * 20, capacity_ah=2.60)

        traj = DegradationTrajectory(
            trajectory_id="traj_meta_01",
            system_id="cell_meta_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_240",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=2.40,
            unit="Ah",
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertTrue(pred.is_valid)
        meta = pred.reproducibility_metadata

        # Delta x = 50 - 10 = 40; Deg rate = 0.40 / 40 = 0.01 Ah/cycle
        # RUL = (2.60 - 2.40) / 0.01 = 20 cycles
        self.assertEqual(meta["last_observation_domain_value"], 50.0)
        self.assertEqual(meta["predicted_eol_domain_value"], 70.0)
        self.assertEqual(meta["extrapolation_distance"], 20.0)
        self.assertEqual(meta["observed_history_span"], 40.0)
        self.assertEqual(meta["extrapolation_ratio"], 0.5)  # 20 / 40 = 0.5
        self.assertIsNone(pred.uncertainty)  # Not presented as uncertainty

    # 13. Wrong Slope Direction (Trajectory Rebounding / Not Approaching EOL)
    def test_wrong_slope_direction_returns_out_of_domain(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=2.80)
        obs2 = self._create_observation(efc=50.0, timestamp_s=86400.0 * 20, capacity_ah=2.90)

        traj = DegradationTrajectory(
            trajectory_id="traj_rebound_01",
            system_id="cell_rebound_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_240",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=2.40,
            unit="Ah",
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertFalse(pred.is_valid)
        self.assertEqual(pred.status, PrognosticStatus.OUT_OF_DOMAIN)
        self.assertIsNone(pred.rul_value)

    # 14. Flat / Zero Slope (No Degradation Detected)
    def test_zero_slope_returns_out_of_domain(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=3.00)
        obs2 = self._create_observation(efc=50.0, timestamp_s=86400.0 * 20, capacity_ah=3.00)

        traj = DegradationTrajectory(
            trajectory_id="traj_flat_01",
            system_id="cell_flat_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_240",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=2.40,
            unit="Ah",
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertFalse(pred.is_valid)
        self.assertEqual(pred.status, PrognosticStatus.OUT_OF_DOMAIN)
        self.assertIsNone(pred.rul_value)

    # 15. Insufficient History Handling
    def test_insufficient_history_returns_insufficient_history_status(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=3.00)

        traj = DegradationTrajectory(
            trajectory_id="traj_single_01",
            system_id="cell_single_01",
            observations=(obs1,),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_240",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=2.40,
            unit="Ah",
        )

        pred = self.estimator.estimate(traj, eol)
        self.assertFalse(pred.is_valid)
        self.assertEqual(pred.status, PrognosticStatus.INSUFFICIENT_HISTORY)
        self.assertIsNone(pred.rul_value)

    # 16. OLS Descriptive Historical Trend Slope Option
    def test_ols_slope_linear_estimation(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=5.00)
        obs2 = self._create_observation(efc=50.0, timestamp_s=86400.0 * 20, capacity_ah=4.90)
        obs3 = self._create_observation(efc=100.0, timestamp_s=86400.0 * 40, capacity_ah=4.80)

        traj = DegradationTrajectory(
            trajectory_id="traj_ols_01",
            system_id="cell_ols_01",
            observations=(obs1, obs2, obs3),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_400",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=4.00,
            unit="Ah",
        )

        cfg = EstimatorConfig(use_ols_slope=True)
        pred = self.estimator.estimate(traj, eol, config=cfg)

        self.assertTrue(pred.is_valid)
        self.assertAlmostEqual(pred.rul_value or 0.0, 400.0, places=4)
        self.assertEqual(pred.reproducibility_metadata["slope_method"], "ORDINARY_LEAST_SQUARES")

    # 17. Determinism and Immutability of Inputs
    def test_input_immutability_and_execution_determinism(self) -> None:
        obs1 = self._create_observation(efc=0.0, timestamp_s=0.0, capacity_ah=3.00)
        obs2 = self._create_observation(efc=50.0, timestamp_s=86400.0 * 20, capacity_ah=2.70)

        traj = DegradationTrajectory(
            trajectory_id="traj_det_01",
            system_id="cell_det_01",
            observations=(obs1, obs2),
        )

        eol = EOLDefinition(
            criterion_id="eol_cap_240",
            criterion_type=EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            threshold_value=2.40,
            unit="Ah",
        )
        cfg = EstimatorConfig(min_observations=2)

        # Snapshot initial state
        traj_obs_count = len(traj.observations)
        eol_thresh = eol.threshold_value
        cfg_min_obs = cfg.min_observations

        pred1 = self.estimator.estimate(traj, eol, config=cfg)
        pred2 = self.estimator.estimate(traj, eol, config=cfg)

        # Verify inputs remained untouched
        self.assertEqual(len(traj.observations), traj_obs_count)
        self.assertEqual(eol.threshold_value, eol_thresh)
        self.assertEqual(cfg.min_observations, cfg_min_obs)

        # Verify outputs are identical
        self.assertEqual(pred1.rul_value, pred2.rul_value)
        self.assertEqual(pred1.status, pred2.status)
        self.assertEqual(pred1.rul_unit, pred2.rul_unit)
        self.assertEqual(pred1.reproducibility_metadata, pred2.reproducibility_metadata)


if __name__ == "__main__":
    unittest.main()

