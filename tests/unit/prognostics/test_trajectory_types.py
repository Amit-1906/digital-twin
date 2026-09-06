"""Unit tests for trajectory enumerations and classifications."""

import unittest

from src.prognostics.trajectory_types import (
    FeatureQualityStatus,
    RateDomain,
    TrajectoryOrderingStatus,
    TrajectoryQuantityType,
    TrendDirection,
)


class TestTrajectoryTypes(unittest.TestCase):
    """Test suite verifying trajectory and feature extraction enumerations."""

    def test_trajectory_quantity_type_enum(self) -> None:
        expected = {
            "CAPACITY",
            "CAPACITY_RETENTION",
            "INTERNAL_RESISTANCE",
            "RESISTANCE_GROWTH_RATIO",
            "EQUIVALENT_FULL_CYCLES",
            "THROUGHPUT_AH",
            "THROUGHPUT_WH",
            "TEMPERATURE",
            "MULTIVARIATE",
            "CUSTOM",
        }
        self.assertEqual({q.value for q in TrajectoryQuantityType}, expected)

    def test_trajectory_ordering_status_enum(self) -> None:
        expected = {
            "STRICTLY_MONOTONIC",
            "MONOTONIC_WITH_DUPLICATES",
            "RETROGRADE_DETECTED",
            "EMPTY",
            "SINGLE_POINT",
        }
        self.assertEqual({s.value for s in TrajectoryOrderingStatus}, expected)

    def test_rate_domain_enum(self) -> None:
        expected = {
            "PER_SECOND",
            "PER_HOUR",
            "PER_DAY",
            "PER_EFC",
            "PER_AMPERE_HOUR",
            "PER_WATT_HOUR",
            "NONE",
        }
        self.assertEqual({r.value for r in RateDomain}, expected)

    def test_feature_quality_status_enum(self) -> None:
        expected = {
            "VALID",
            "INSUFFICIENT_DATA",
            "DATA_QUALITY_FAILED",
            "UNAVAILABLE",
            "ZERO_DENOMINATOR",
            "UNDEFINED_VARIANCE",
            "OUT_OF_DOMAIN",
        }
        self.assertEqual({q.value for q in FeatureQualityStatus}, expected)

    def test_trend_direction_enum(self) -> None:
        expected = {"DEGRADING", "STABLE", "RECOVERING", "INDETERMINATE"}
        self.assertEqual({t.value for t in TrendDirection}, expected)


if __name__ == "__main__":
    unittest.main()
