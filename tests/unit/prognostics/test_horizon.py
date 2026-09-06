"""Unit tests for prognostic horizon contracts across projection domains."""

import unittest

from src.prognostics.horizon import PrognosticHorizon
from src.prognostics.types import HorizonDomain


class TestPrognosticHorizon(unittest.TestCase):
    """Test suite verifying multi-domain prognostic horizon contracts."""

    def test_time_based_horizon(self) -> None:
        h = PrognosticHorizon(
            domain=HorizonDomain.TIME,
            horizon_value=86400.0 * 30.0,  # 30 days in seconds
            unit="seconds",
            start_value=0.0,
            description="30-day forward degradation projection",
        )
        self.assertEqual(h.domain, HorizonDomain.TIME)
        self.assertEqual(h.horizon_value, 2592000.0)
        self.assertEqual(h.unit, "seconds")
        self.assertEqual(h.start_value, 0.0)
        self.assertEqual(h.end_value, 2592000.0)

        d = h.to_dict()
        self.assertEqual(d["domain"], "TIME")
        self.assertEqual(d["horizon_value"], 2592000.0)
        self.assertEqual(d["unit"], "seconds")
        self.assertEqual(d["end_value"], 2592000.0)

    def test_cycle_based_horizon(self) -> None:
        h = PrognosticHorizon(
            domain=HorizonDomain.CYCLE,
            horizon_value=500.0,
            unit="cycles",
            start_value=250.0,
            description="500 cycles forward from cycle 250",
        )
        self.assertEqual(h.domain, HorizonDomain.CYCLE)
        self.assertEqual(h.horizon_value, 500.0)
        self.assertEqual(h.start_value, 250.0)
        self.assertEqual(h.end_value, 750.0)

        d = h.to_dict()
        self.assertEqual(d["domain"], "CYCLE")
        self.assertEqual(d["start_value"], 250.0)
        self.assertEqual(d["end_value"], 750.0)

    def test_ampere_hour_throughput_horizon(self) -> None:
        h = PrognosticHorizon(
            domain=HorizonDomain.AMPERE_HOUR_THROUGHPUT,
            horizon_value=10000.0,
            unit="Ah",
            start_value=1500.0,
            description="10,000 Ah throughput horizon",
        )
        self.assertEqual(h.domain, HorizonDomain.AMPERE_HOUR_THROUGHPUT)
        self.assertEqual(h.end_value, 11500.0)

    def test_watt_hour_throughput_horizon(self) -> None:
        h = PrognosticHorizon(
            domain=HorizonDomain.WATT_HOUR_THROUGHPUT,
            horizon_value=36000.0,
            unit="Wh",
            start_value=5000.0,
            description="36 kWh forward energy horizon",
        )
        self.assertEqual(h.domain, HorizonDomain.WATT_HOUR_THROUGHPUT)
        self.assertEqual(h.end_value, 41000.0)

    def test_custom_domain_horizon(self) -> None:
        h = PrognosticHorizon(
            domain=HorizonDomain.CUSTOM,
            horizon_value=100.0,
            unit="operating_missions",
            description="100 standardized flight missions",
        )
        self.assertEqual(h.domain, HorizonDomain.CUSTOM)
        self.assertEqual(h.unit, "operating_missions")

    def test_horizon_validation_errors(self) -> None:
        with self.assertRaises(TypeError):
            PrognosticHorizon(domain="TIME", horizon_value=100.0, unit="seconds")  # type: ignore

        with self.assertRaises(ValueError):
            # Zero horizon value
            PrognosticHorizon(domain=HorizonDomain.TIME, horizon_value=0.0, unit="seconds")

        with self.assertRaises(ValueError):
            # Negative horizon value
            PrognosticHorizon(domain=HorizonDomain.TIME, horizon_value=-10.0, unit="seconds")

        with self.assertRaises(ValueError):
            # Negative start value
            PrognosticHorizon(
                domain=HorizonDomain.TIME,
                horizon_value=10.0,
                unit="seconds",
                start_value=-1.0,
            )

        with self.assertRaises(ValueError):
            # Empty unit
            PrognosticHorizon(domain=HorizonDomain.TIME, horizon_value=10.0, unit="")

    def test_horizon_immutability(self) -> None:
        h = PrognosticHorizon(domain=HorizonDomain.TIME, horizon_value=100.0, unit="s")
        with self.assertRaises(AttributeError):
            h.horizon_value = 200.0  # type: ignore


if __name__ == "__main__":
    unittest.main()
