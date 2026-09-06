"""Unit tests for prognostics types, enums, and classifications."""

import unittest

from src.prognostics.types import (
    ComparisonOperator,
    EOLCriterionType,
    HorizonDomain,
    PrognosticStatus,
    QuantityAvailability,
    TimeUnit,
    UncertaintyType,
    ValueOriginType,
)


class TestPrognosticsTypes(unittest.TestCase):
    """Test suite verifying prognostic enumerations and status contracts."""

    def test_prognostic_status_enum_values(self) -> None:
        expected_statuses = {
            "VALID",
            "UNINITIALIZED",
            "INSUFFICIENT_DATA",
            "INSUFFICIENT_HISTORY",
            "DATA_QUALITY_FAILED",
            "OUT_OF_DOMAIN",
            "MODEL_UNAVAILABLE",
        }
        actual_statuses = {s.value for s in PrognosticStatus}
        self.assertEqual(actual_statuses, expected_statuses)

    def test_value_origin_type_enum_values(self) -> None:
        expected_origins = {"MEASURED", "ESTIMATED", "MODELED", "DERIVED", "UNKNOWN"}
        actual_origins = {o.value for o in ValueOriginType}
        self.assertEqual(actual_origins, expected_origins)

    def test_quantity_availability_enum_values(self) -> None:
        expected = {"AVAILABLE", "UNAVAILABLE", "ESTIMATED_INDIRECT", "OUT_OF_RANGE"}
        self.assertEqual({a.value for a in QuantityAvailability}, expected)

    def test_eol_criterion_type_enum_values(self) -> None:
        expected = {
            "CAPACITY_RETENTION_THRESHOLD",
            "ABSOLUTE_CAPACITY_THRESHOLD",
            "RESISTANCE_GROWTH_RATIO_THRESHOLD",
            "ABSOLUTE_RESISTANCE_THRESHOLD",
            "PEAK_POWER_THRESHOLD",
            "APPLICATION_SPECIFIC",
            "CUSTOM",
        }
        self.assertEqual({c.value for c in EOLCriterionType}, expected)

    def test_comparison_operator_enum_values(self) -> None:
        expected = {
            "LESS_THAN_OR_EQUAL",
            "GREATER_THAN_OR_EQUAL",
            "LESS_THAN",
            "GREATER_THAN",
            "EQUALS",
        }
        self.assertEqual({op.value for op in ComparisonOperator}, expected)

    def test_horizon_domain_enum_values(self) -> None:
        expected = {
            "TIME",
            "CYCLE",
            "AMPERE_HOUR_THROUGHPUT",
            "WATT_HOUR_THROUGHPUT",
            "CUSTOM",
        }
        self.assertEqual({h.value for h in HorizonDomain}, expected)

    def test_time_unit_enum_values(self) -> None:
        expected = {"SECONDS", "MINUTES", "HOURS", "DAYS", "WEEKS", "MONTHS", "YEARS"}
        self.assertEqual({u.value for u in TimeUnit}, expected)

    def test_uncertainty_type_enum_values(self) -> None:
        expected = {
            "CONFIDENCE_BOUNDS",
            "STANDARD_DEVIATION",
            "PERCENTILE_INTERVAL",
            "EXPANDED_UNCERTAINTY",
            "DETERMINISTIC_NO_UNCERTAINTY",
            "UNDEFINED",
        }
        self.assertEqual({u.value for u in UncertaintyType}, expected)


if __name__ == "__main__":
    unittest.main()
