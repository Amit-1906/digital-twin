"""Unit tests for base RUL estimator interfaces and configuration."""

import unittest

from src.prognostics.estimators.base import (
    AbstractRULEstimator,
    EstimatorConfig,
)
from src.prognostics.types import HorizonDomain


class TestBaseEstimator(unittest.TestCase):
    """Test suite verifying base RUL estimator contracts and configurations."""

    def test_estimator_config_defaults_and_serialization(self) -> None:
        cfg = EstimatorConfig()
        self.assertEqual(cfg.min_observations, 2)
        self.assertEqual(cfg.slope_threshold, 1e-9)
        self.assertFalse(cfg.use_ols_slope)
        self.assertIsNone(cfg.default_domain)

        d = cfg.to_dict()
        self.assertEqual(d["min_observations"], 2)
        self.assertEqual(d["slope_threshold"], 1e-9)
        self.assertFalse(d["use_ols_slope"])
        self.assertIsNone(d["default_domain"])

    def test_estimator_config_custom_values(self) -> None:
        cfg = EstimatorConfig(
            min_observations=5,
            slope_threshold=1e-6,
            use_ols_slope=True,
            default_domain=HorizonDomain.CYCLE,
            custom_parameters={"extrapolate_mode": "linear"},
        )
        self.assertEqual(cfg.min_observations, 5)
        self.assertTrue(cfg.use_ols_slope)
        self.assertEqual(cfg.default_domain, HorizonDomain.CYCLE)
        self.assertEqual(cfg.custom_parameters["extrapolate_mode"], "linear")

    def test_estimator_config_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            EstimatorConfig(min_observations=0)

        with self.assertRaises(ValueError):
            EstimatorConfig(slope_threshold=-1.0)

        with self.assertRaises(TypeError):
            EstimatorConfig(default_domain="CYCLE")  # type: ignore

    def test_estimator_config_prediction_cutoff(self) -> None:
        cfg = EstimatorConfig(prediction_cutoff_timestamp_ns=1_000_000_000)
        self.assertEqual(cfg.prediction_cutoff_timestamp_ns, 1_000_000_000)
        d = cfg.to_dict()
        self.assertEqual(d["prediction_cutoff_timestamp_ns"], 1_000_000_000)

        with self.assertRaises(ValueError):
            EstimatorConfig(prediction_cutoff_timestamp_ns=-1)

        with self.assertRaises(ValueError):
            EstimatorConfig(prediction_cutoff_timestamp_ns="1000")  # type: ignore

    def test_abstract_estimator_cannot_be_instantiated_directly(self) -> None:
        with self.assertRaises(TypeError):
            AbstractRULEstimator()  # type: ignore


if __name__ == "__main__":
    unittest.main()

