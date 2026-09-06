"""Unit tests for prognostic value provenance and quantity containers."""

import math
import unittest

from src.prognostics.provenance import (
    PrognosticQuantity,
    PrognosticValueProvenance,
)
from src.prognostics.types import QuantityAvailability, ValueOriginType


class TestPrognosticProvenance(unittest.TestCase):
    """Test suite verifying prognostic provenance and quantity value objects."""

    def test_prognostic_value_provenance_creation_and_serialization(self) -> None:
        prov = PrognosticValueProvenance(
            origin_type=ValueOriginType.MEASURED,
            source_identifier="cycler_arbin_ch04",
            timestamp_ns=1_700_000_000_000_000,
            quality_rating="HIGH_PRECISION",
            measurement_uncertainty=0.005,
            metadata={"calibration_date": "2026-01-15", "instrument": "Arbin LBT21084"},
        )
        self.assertEqual(prov.origin_type, ValueOriginType.MEASURED)
        self.assertEqual(prov.source_identifier, "cycler_arbin_ch04")
        self.assertEqual(prov.timestamp_ns, 1_700_000_000_000_000)
        self.assertEqual(prov.quality_rating, "HIGH_PRECISION")
        self.assertEqual(prov.measurement_uncertainty, 0.005)

        d = prov.to_dict()
        self.assertEqual(d["origin_type"], "MEASURED")
        self.assertEqual(d["source_identifier"], "cycler_arbin_ch04")
        self.assertEqual(d["measurement_uncertainty"], 0.005)
        self.assertEqual(d["metadata"]["instrument"], "Arbin LBT21084")

    def test_provenance_validation_errors(self) -> None:
        with self.assertRaises(TypeError):
            PrognosticValueProvenance(origin_type="NOT_AN_ENUM")  # type: ignore

        with self.assertRaises(ValueError):
            PrognosticValueProvenance(timestamp_ns=-1)

        with self.assertRaises(ValueError):
            PrognosticValueProvenance(measurement_uncertainty=-0.1)

        with self.assertRaises(ValueError):
            PrognosticValueProvenance(measurement_uncertainty=float("nan"))

    def test_provenance_immutability(self) -> None:
        prov = PrognosticValueProvenance(
            origin_type=ValueOriginType.ESTIMATED,
            source_identifier="ekf_observer",
        )
        with self.assertRaises(AttributeError):
            prov.origin_type = ValueOriginType.MEASURED  # type: ignore

    def test_prognostic_quantity_available(self) -> None:
        prov = PrognosticValueProvenance(origin_type=ValueOriginType.ESTIMATED)
        q = PrognosticQuantity(
            name="usable_capacity_ah",
            value=2.45,
            unit="Ah",
            availability=QuantityAvailability.AVAILABLE,
            provenance=prov,
            description="Usable capacity estimated via deep cycling integration",
        )
        self.assertTrue(q.is_available)
        self.assertEqual(q.value, 2.45)
        self.assertEqual(q.unit, "Ah")

        d = q.to_dict()
        self.assertEqual(d["name"], "usable_capacity_ah")
        self.assertEqual(d["value"], 2.45)
        self.assertEqual(d["is_available"], True)
        self.assertEqual(d["availability"], "AVAILABLE")
        self.assertIsNotNone(d["provenance"])

    def test_prognostic_quantity_unavailable(self) -> None:
        q = PrognosticQuantity(
            name="cumulative_thermal_exposure",
            value=None,
            unit="degC_hours",
            availability=QuantityAvailability.UNAVAILABLE,
            description="Thermal tracking not active on legacy hardware",
        )
        self.assertFalse(q.is_available)
        self.assertIsNone(q.value)

        d = q.to_dict()
        self.assertEqual(d["availability"], "UNAVAILABLE")
        self.assertFalse(d["is_available"])
        self.assertIsNone(d["value"])

    def test_prognostic_quantity_available_requires_finite_value(self) -> None:
        with self.assertRaises(ValueError):
            # Cannot be None when marked AVAILABLE
            PrognosticQuantity(
                name="capacity",
                value=None,
                unit="Ah",
                availability=QuantityAvailability.AVAILABLE,
            )

        with self.assertRaises(ValueError):
            # Cannot be NaN
            PrognosticQuantity(
                name="capacity",
                value=float("nan"),
                unit="Ah",
                availability=QuantityAvailability.AVAILABLE,
            )

        with self.assertRaises(ValueError):
            # Cannot be Inf
            PrognosticQuantity(
                name="capacity",
                value=float("inf"),
                unit="Ah",
                availability=QuantityAvailability.AVAILABLE,
            )

    def test_prognostic_quantity_immutability(self) -> None:
        q = PrognosticQuantity(name="test", value=1.0, unit="V")
        with self.assertRaises(AttributeError):
            q.value = 2.0  # type: ignore


if __name__ == "__main__":
    unittest.main()
