"""Unit tests for prognostic prediction results, EOL definitions, and uncertainty specifications."""

import unittest

from src.prognostics.horizon import PrognosticHorizon
from src.prognostics.prediction import (
    EOLDefinition,
    PredictionUncertainty,
    PrognosticMethodIdentity,
    RULPredictionResult,
)
from src.prognostics.state import PrognosticOperatingContext
from src.prognostics.types import (
    ComparisonOperator,
    EOLCriterionType,
    HorizonDomain,
    PrognosticStatus,
    UncertaintyType,
)


class TestPrognosticPrediction(unittest.TestCase):
    """Test suite verifying RUL prediction results, EOL contracts, and uncertainty contracts."""

    def setUp(self) -> None:
        self.eol_soh = EOLDefinition(
            criterion_id="eol_soh_80",
            criterion_type=EOLCriterionType.CAPACITY_RETENTION_THRESHOLD,
            threshold_value=0.80,
            unit="fraction",
            comparison_operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
            name="Standard Automotive 80% SOH EOL",
            description="End-of-life declared when capacity retention drops to 80%",
            version="1.0.0",
        )

        self.eol_res = EOLDefinition(
            criterion_id="eol_res_200",
            criterion_type=EOLCriterionType.RESISTANCE_GROWTH_RATIO_THRESHOLD,
            threshold_value=2.0,
            unit="ratio",
            comparison_operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
            name="2.0x Internal Resistance Growth EOL",
            version="1.0.0",
        )

        self.method_id = PrognosticMethodIdentity(
            method_id="empirical_power_law_v1",
            method_version="1.2.0",
            model_id="arrhenius_sei_growth_model",
            model_version="2.0.1",
            algorithm_paradigm="EMPIRICAL_SEI",
            parameters={"activation_energy_ev": 0.35, "alpha_c": 0.5},
        )

        self.uncertainty_bounds = PredictionUncertainty(
            uncertainty_type=UncertaintyType.CONFIDENCE_BOUNDS,
            lower_bound=450.0,
            upper_bound=550.0,
            coverage_parameter=0.95,
            standard_deviation=25.5,
            unit="cycles",
            description="95% two-sided confidence interval",
        )

    def test_eol_definition_creation_and_serialization(self) -> None:
        self.assertEqual(self.eol_soh.criterion_id, "eol_soh_80")
        self.assertEqual(self.eol_soh.threshold_value, 0.80)
        self.assertEqual(self.eol_soh.comparison_operator, ComparisonOperator.LESS_THAN_OR_EQUAL)

        d = self.eol_soh.to_dict()
        self.assertEqual(d["criterion_type"], "CAPACITY_RETENTION_THRESHOLD")
        self.assertEqual(d["threshold_value"], 0.80)
        self.assertEqual(d["unit"], "fraction")
        self.assertEqual(d["comparison_operator"], "LESS_THAN_OR_EQUAL")

    def test_eol_definition_validation(self) -> None:
        with self.assertRaises(ValueError):
            EOLDefinition(
                criterion_id="",
                criterion_type=EOLCriterionType.CAPACITY_RETENTION_THRESHOLD,
                threshold_value=0.8,
                unit="fraction",
            )

        with self.assertRaises(TypeError):
            EOLDefinition(
                criterion_id="id",
                criterion_type="NOT_AN_ENUM",  # type: ignore
                threshold_value=0.8,
                unit="fraction",
            )

        with self.assertRaises(ValueError):
            EOLDefinition(
                criterion_id="id",
                criterion_type=EOLCriterionType.CAPACITY_RETENTION_THRESHOLD,
                threshold_value=float("nan"),
                unit="fraction",
            )

    def test_method_identity_creation_and_serialization(self) -> None:
        self.assertEqual(self.method_id.method_id, "empirical_power_law_v1")
        self.assertEqual(self.method_id.method_version, "1.2.0")
        d = self.method_id.to_dict()
        self.assertEqual(d["algorithm_paradigm"], "EMPIRICAL_SEI")
        self.assertEqual(d["parameters"]["activation_energy_ev"], 0.35)

    def test_prediction_uncertainty_validation(self) -> None:
        # lower_bound > upper_bound should raise ValueError
        with self.assertRaises(ValueError):
            PredictionUncertainty(
                uncertainty_type=UncertaintyType.CONFIDENCE_BOUNDS,
                lower_bound=600.0,
                upper_bound=500.0,
            )

        # negative standard deviation should raise ValueError
        with self.assertRaises(ValueError):
            PredictionUncertainty(
                uncertainty_type=UncertaintyType.STANDARD_DEVIATION,
                standard_deviation=-5.0,
            )

    def test_valid_rul_prediction_result(self) -> None:
        ctx = PrognosticOperatingContext(ambient_temperature_c=25.0, c_rate=0.5)

        pred = RULPredictionResult(
            prediction_id="pred_rul_20260904_001",
            prediction_timestamp_ns=1_700_000_000_000_000,
            created_at_utc="2026-09-04T12:00:00Z",
            system_id="pack_nmc_01",
            status=PrognosticStatus.VALID,
            rul_value=512.0,
            rul_unit="cycles",
            rul_domain=HorizonDomain.CYCLE,
            uncertainty=self.uncertainty_bounds,
            eol_definition=self.eol_soh,
            method_identity=self.method_id,
            operating_context=ctx,
            input_state_snapshot_id="snap_prog_001",
            diagnostic_message="Prediction nominal within operational validity domain",
            reproducibility_metadata={"git_commit": "abcdef123", "random_seed": 42},
        )

        self.assertTrue(pred.is_valid)
        self.assertEqual(pred.rul_value, 512.0)
        self.assertEqual(pred.rul_unit, "cycles")
        self.assertEqual(pred.rul_domain, HorizonDomain.CYCLE)

        d = pred.to_dict()
        self.assertEqual(d["prediction_id"], "pred_rul_20260904_001")
        self.assertEqual(d["status"], "VALID")
        self.assertEqual(d["is_valid"], True)
        self.assertEqual(d["rul_value"], 512.0)
        self.assertEqual(d["uncertainty"]["lower_bound"], 450.0)
        self.assertEqual(d["uncertainty"]["upper_bound"], 550.0)
        self.assertEqual(d["eol_definition"]["threshold_value"], 0.80)
        self.assertEqual(d["method_identity"]["method_id"], "empirical_power_law_v1")
        self.assertEqual(d["operating_context"]["ambient_temperature_c"], 25.0)

    def test_non_valid_prediction_statuses_require_none_rul_value(self) -> None:
        non_valid_statuses = [
            PrognosticStatus.UNINITIALIZED,
            PrognosticStatus.INSUFFICIENT_DATA,
            PrognosticStatus.INSUFFICIENT_HISTORY,
            PrognosticStatus.DATA_QUALITY_FAILED,
            PrognosticStatus.OUT_OF_DOMAIN,
            PrognosticStatus.MODEL_UNAVAILABLE,
        ]

        for st in non_valid_statuses:
            with self.subTest(status=st):
                # Correct usage: rul_value=None
                pred = RULPredictionResult(
                    prediction_id=f"pred_{st.value.lower()}",
                    prediction_timestamp_ns=1_000_000_000,
                    created_at_utc="2026-09-04T00:00:00Z",
                    system_id="sys_01",
                    status=st,
                    rul_value=None,
                    rul_unit="cycles",
                    rul_domain=HorizonDomain.CYCLE,
                    diagnostic_message=f"Status is {st.value}",
                )
                self.assertFalse(pred.is_valid)
                self.assertIsNone(pred.rul_value)

                # Invalid usage: Providing a float (even 0.0) for an unavailable/failed status must fail
                with self.assertRaises(ValueError):
                    RULPredictionResult(
                        prediction_id=f"pred_bad_{st.value.lower()}",
                        prediction_timestamp_ns=1_000_000_000,
                        created_at_utc="2026-09-04T00:00:00Z",
                        system_id="sys_01",
                        status=st,
                        rul_value=0.0,  # Prohibited! RUL=0 must not represent unavailable states
                        rul_unit="cycles",
                        rul_domain=HorizonDomain.CYCLE,
                    )

    def test_valid_status_requires_non_none_rul_value(self) -> None:
        with self.assertRaises(ValueError):
            RULPredictionResult(
                prediction_id="pred_valid_missing_rul",
                prediction_timestamp_ns=1_000_000_000,
                created_at_utc="2026-09-04T00:00:00Z",
                system_id="sys_01",
                status=PrognosticStatus.VALID,
                rul_value=None,  # Prohibited for VALID status!
                rul_unit="cycles",
                rul_domain=HorizonDomain.CYCLE,
            )

    def test_rul_value_out_of_uncertainty_bounds_raises(self) -> None:
        # rul_value < lower_bound
        with self.assertRaises(ValueError):
            RULPredictionResult(
                prediction_id="pred_out_lower",
                prediction_timestamp_ns=1_000_000_000,
                created_at_utc="2026-09-04T00:00:00Z",
                system_id="sys_01",
                status=PrognosticStatus.VALID,
                rul_value=400.0,
                rul_unit="cycles",
                rul_domain=HorizonDomain.CYCLE,
                uncertainty=self.uncertainty_bounds,  # bounds are [450, 550]
            )

        # rul_value > upper_bound
        with self.assertRaises(ValueError):
            RULPredictionResult(
                prediction_id="pred_out_upper",
                prediction_timestamp_ns=1_000_000_000,
                created_at_utc="2026-09-04T00:00:00Z",
                system_id="sys_01",
                status=PrognosticStatus.VALID,
                rul_value=600.0,
                rul_unit="cycles",
                rul_domain=HorizonDomain.CYCLE,
                uncertainty=self.uncertainty_bounds,  # bounds are [450, 550]
            )

    def test_rul_value_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            RULPredictionResult(
                prediction_id="pred_negative",
                prediction_timestamp_ns=1_000_000_000,
                created_at_utc="2026-09-04T00:00:00Z",
                system_id="sys_01",
                status=PrognosticStatus.VALID,
                rul_value=-10.0,
                rul_unit="cycles",
                rul_domain=HorizonDomain.CYCLE,
            )

    def test_rul_prediction_result_immutability(self) -> None:
        pred = RULPredictionResult(
            prediction_id="pred_immut",
            prediction_timestamp_ns=1_000_000_000,
            created_at_utc="2026-09-04T00:00:00Z",
            system_id="sys_01",
            status=PrognosticStatus.VALID,
            rul_value=100.0,
            rul_unit="cycles",
            rul_domain=HorizonDomain.CYCLE,
        )
        with self.assertRaises(AttributeError):
            pred.rul_value = 200.0  # type: ignore


if __name__ == "__main__":
    unittest.main()
