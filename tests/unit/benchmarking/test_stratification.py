"""Unit tests for Level 5.5 Phase 2 Operating-Context Stratification and Bucketing."""

import unittest

from src.benchmarking.stratification import (
    ContextDimension,
    ContextSliceIdentifier,
    NumericRangeBucket,
    OperatingContextStratifier,
    StratificationConfig,
)


class TestBenchmarkingStratification(unittest.TestCase):
    """Test suite for context slicing, range buckets, and multi-dimensional stratification."""

    def test_range_bucket_bounds_and_contains(self) -> None:
        bucket = NumericRangeBucket(
            name="MID_SOC",
            min_value=0.20,
            max_value=0.80,
            inclusive_min=True,
            inclusive_max=True,
        )
        self.assertTrue(bucket.contains(0.20))
        self.assertTrue(bucket.contains(0.50))
        self.assertTrue(bucket.contains(0.80))
        self.assertFalse(bucket.contains(0.19))
        self.assertFalse(bucket.contains(0.81))
        self.assertFalse(bucket.contains(float("nan")))

        # Exclusive upper bound
        bucket_ex = NumericRangeBucket(
            name="LOW_SOC",
            min_value=0.0,
            max_value=0.20,
            inclusive_min=True,
            inclusive_max=False,
        )
        self.assertTrue(bucket_ex.contains(0.0))
        self.assertTrue(bucket_ex.contains(0.199))
        self.assertFalse(bucket_ex.contains(0.20))

        # Invalid bucket bounds
        with self.assertRaises(ValueError):
            NumericRangeBucket(name="BAD", min_value=1.0, max_value=0.0)

    def test_default_stratification_classification(self) -> None:
        strat = OperatingContextStratifier()

        # Rest condition at room temperature, mid SOC
        ctx = strat.classify(
            current_a=0.01,
            voltage_v=3.7,
            temperature_c=25.0,
            soc_fraction=0.50,
        )
        self.assertEqual(ctx.operating_regime, "REST")
        self.assertEqual(ctx.soc_bucket, "MID_SOC")
        self.assertEqual(ctx.temperature_bucket, "ROOM_TEMPERATURE")
        self.assertEqual(ctx.current_bucket, "REST")

        # Verify active slices
        self.assertTrue(ctx.matches_slice(ContextSliceIdentifier(ContextDimension.OVERALL, "ALL")))
        self.assertTrue(ctx.matches_slice(ContextSliceIdentifier(ContextDimension.REGIME, "REST")))
        self.assertTrue(ctx.matches_slice(ContextSliceIdentifier(ContextDimension.SOC, "MID_SOC")))
        self.assertTrue(ctx.matches_slice(ContextSliceIdentifier(ContextDimension.TEMPERATURE, "ROOM_TEMPERATURE")))
        self.assertFalse(ctx.matches_slice(ContextSliceIdentifier(ContextDimension.SOC, "LOW_SOC")))

    def test_charge_discharge_regimes(self) -> None:
        strat = OperatingContextStratifier()

        ctx_chg = strat.classify(current_a=-10.0)
        self.assertEqual(ctx_chg.operating_regime, "CHARGE")

        ctx_dis = strat.classify(current_a=15.0)
        self.assertEqual(ctx_dis.operating_regime, "DISCHARGE")

        ctx_hint = strat.classify(current_a=15.0, operating_context_hint="DYNAMIC_PULSE")
        self.assertEqual(ctx_hint.operating_regime, "DYNAMIC_PULSE")

    def test_custom_chemistry_and_c_rate_stratification(self) -> None:
        # Custom LTO chemistry boundaries (e.g. operational from -30C to +60C, nominal capacity 50Ah)
        custom_cfg = StratificationConfig(
            soc_buckets=(
                NumericRangeBucket("DEEP_DISCHARGE", 0.0, 0.10),
                NumericRangeBucket("LINEAR_RANGE", 0.10, 0.90),
                NumericRangeBucket("OVERPOTENTIAL_ZONE", 0.90, 1.0),
            ),
            temperature_buckets_c=(
                NumericRangeBucket("EXTREME_COLD", -40.0, -10.0),
                NumericRangeBucket("STANDARD_OPERATING", -10.0, 45.0),
                NumericRangeBucket("THERMAL_STRESS", 45.0, 85.0),
            ),
            current_crate_buckets=(
                NumericRangeBucket("SUB_1C", 0.0, 1.0),
                NumericRangeBucket("FAST_CHARGE_RATE", 1.0, 10.0),
            ),
            voltage_buckets_v=(
                NumericRangeBucket("NOMINAL_LTO_SPAN", 1.8, 2.7),
            ),
            nominal_capacity_ah=50.0,
            custom_phase_labels=("CYCLE_01", "CYCLE_02"),
        )

        strat = OperatingContextStratifier()
        # Test 100A current (= 2.0 C-rate on 50Ah pack), temperature -20C, voltage 2.3V, SOC 0.05
        ctx = strat.classify(
            current_a=100.0,
            voltage_v=2.3,
            temperature_c=-20.0,
            soc_fraction=0.05,
            phase_hint="CYCLE_01",
            config=custom_cfg,
        )
        self.assertEqual(ctx.soc_bucket, "DEEP_DISCHARGE")
        self.assertEqual(ctx.temperature_bucket, "EXTREME_COLD")
        self.assertEqual(ctx.current_bucket, "FAST_CHARGE_RATE")
        self.assertEqual(ctx.voltage_bucket, "NOMINAL_LTO_SPAN")
        self.assertEqual(ctx.phase, "CYCLE_01")
        self.assertTrue(ctx.matches_slice(ContextSliceIdentifier(ContextDimension.PHASE, "CYCLE_01")))

    def test_serialization(self) -> None:
        slice_id = ContextSliceIdentifier(ContextDimension.SOC, "LOW_SOC")
        d = slice_id.to_dict()
        self.assertEqual(d["dimension"], "SOC")
        self.assertEqual(d["bucket_name"], "LOW_SOC")


if __name__ == "__main__":
    unittest.main()
