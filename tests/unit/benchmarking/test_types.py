"""Unit tests for Level 5.5 Phase 1 Benchmarking Types, Provenance, and Measurement Contracts."""

import math
import unittest

from src.benchmarking.types import (
    BenchmarkSemanticClass,
    BenchmarkSignalType,
    CalibrationStatus,
    EvidenceRole,
    ExperimentalGroundTruthEvent,
    ReferenceDataQuality,
    ReferenceMeasurement,
    ReferenceProvenance,
    ReferenceSignalChannel,
    ReferenceSourceType,
)


class TestBenchmarkingTypes(unittest.TestCase):
    """Test suite for Level 5.5 foundational value objects and provenance contracts."""

    def setUp(self) -> None:
        self.valid_provenance = ReferenceProvenance(
            provenance_id="prov_lab_001",
            source_type=ReferenceSourceType.LABORATORY_METROLOGY,
            calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
            evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
            measurement_method="FOUR_WIRE_KELVIN",
            instrument_identifier="MACCOR_SERIES_4000_CH1",
            calibration_expiry_date="2027-12-31",
            facility_or_environment="National Battery Metrology Lab",
            metadata={"ambient_control": "true"},
        )

    # 1. Provenance construction & immutability
    def test_provenance_construction_and_fields(self) -> None:
        prov = self.valid_provenance
        self.assertEqual(prov.provenance_id, "prov_lab_001")
        self.assertEqual(prov.source_type, ReferenceSourceType.LABORATORY_METROLOGY)
        self.assertEqual(prov.calibration_status, CalibrationStatus.CERTIFIED_CALIBRATED)
        self.assertEqual(prov.evidence_role, EvidenceRole.PHYSICAL_REFERENCE)
        self.assertEqual(prov.measurement_method, "FOUR_WIRE_KELVIN")
        self.assertEqual(prov.instrument_identifier, "MACCOR_SERIES_4000_CH1")
        self.assertEqual(prov.facility_or_environment, "National Battery Metrology Lab")
        self.assertEqual(prov.metadata["ambient_control"], "true")

    def test_provenance_immutability(self) -> None:
        prov = self.valid_provenance
        with self.assertRaises(AttributeError):
            prov.provenance_id = "new_id"  # type: ignore

    def test_provenance_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            ReferenceProvenance(
                provenance_id="",
                source_type=ReferenceSourceType.LABORATORY_METROLOGY,
                calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
                evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
                measurement_method="FOUR_WIRE_KELVIN",
            )
        with self.assertRaises(TypeError):
            ReferenceProvenance(
                provenance_id="p1",
                source_type="INVALID_SOURCE",  # type: ignore
                calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
                evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
                measurement_method="FOUR_WIRE_KELVIN",
            )
        with self.assertRaises(ValueError):
            ReferenceProvenance(
                provenance_id="p1",
                source_type=ReferenceSourceType.LABORATORY_METROLOGY,
                calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
                evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
                measurement_method="",
            )

    # 2. Measurement construction, unit preservation, timestamp preservation
    def test_measurement_construction(self) -> None:
        meas = ReferenceMeasurement(
            value=3.6521,
            unit="V",
            timestamp_ns=1_000_000_000,
            uncertainty_absolute=0.0005,
            provenance=self.valid_provenance,
            signal_type=BenchmarkSignalType.VOLTAGE,
            quality_status=ReferenceDataQuality.VALID,
            observed_at_ns=999_999_950,
        )
        self.assertEqual(meas.value, 3.6521)
        self.assertEqual(meas.unit, "V")
        self.assertEqual(meas.timestamp_ns, 1_000_000_000)
        self.assertEqual(meas.observed_at_ns, 999_999_950)
        self.assertEqual(meas.uncertainty_absolute, 0.0005)
        self.assertEqual(meas.signal_type, BenchmarkSignalType.VOLTAGE)
        self.assertEqual(meas.quality_status, ReferenceDataQuality.VALID)
        self.assertEqual(meas.provenance, self.valid_provenance)

    # 3. Absolute and Relative uncertainty
    def test_absolute_and_relative_uncertainty(self) -> None:
        # Absolute only
        m_abs = ReferenceMeasurement(
            value=100.0,
            unit="A",
            timestamp_ns=0,
            uncertainty_absolute=0.05,
        )
        self.assertEqual(m_abs.effective_uncertainty(), 0.05)

        # Relative only (0.1% = 0.001)
        m_rel = ReferenceMeasurement(
            value=100.0,
            unit="A",
            timestamp_ns=0,
            uncertainty_relative_fraction=0.001,
        )
        self.assertEqual(m_rel.effective_uncertainty(), 0.10)

        # Both: returns maximum effective uncertainty
        m_both = ReferenceMeasurement(
            value=100.0,
            unit="A",
            timestamp_ns=0,
            uncertainty_absolute=0.05,
            uncertainty_relative_fraction=0.001,  # 0.10
        )
        self.assertEqual(m_both.effective_uncertainty(), 0.10)

        # Neither: returns 0.0
        m_none = ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=0)
        self.assertEqual(m_none.effective_uncertainty(), 0.0)

    # 4. Uncertainty validation & rejection of negative, NaN, Inf
    def test_invalid_uncertainty_rejection(self) -> None:
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=0, uncertainty_absolute=-0.01)
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=0, uncertainty_absolute=float("nan"))
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=0, uncertainty_absolute=float("inf"))
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=0, uncertainty_relative_fraction=-0.05)
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=0, uncertainty_relative_fraction=float("nan"))

    # 5. Non-finite values and negative timestamp rejection
    def test_nan_inf_and_negative_timestamp_rejection(self) -> None:
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=float("nan"), unit="V", timestamp_ns=0)
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=float("inf"), unit="V", timestamp_ns=0)
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=float("-inf"), unit="V", timestamp_ns=0)
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=3.7, unit="", timestamp_ns=0)
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=-1)
        with self.assertRaises(ValueError):
            ReferenceMeasurement(value=3.7, unit="V", timestamp_ns=0, observed_at_ns=-5)

    # 6. Serialization
    def test_serialization_deterministic_dict(self) -> None:
        meas = ReferenceMeasurement(
            value=25.4,
            unit="degC",
            timestamp_ns=500_000_000,
            uncertainty_absolute=0.1,
            provenance=self.valid_provenance,
            signal_type=BenchmarkSignalType.TEMPERATURE,
        )
        d = meas.to_dict()
        self.assertEqual(d["value"], 25.4)
        self.assertEqual(d["unit"], "degC")
        self.assertEqual(d["timestamp_ns"], 500_000_000)
        self.assertEqual(d["uncertainty_absolute"], 0.1)
        self.assertEqual(d["signal_type"], "TEMPERATURE")
        self.assertIsNotNone(d["provenance"])
        self.assertEqual(d["provenance"]["provenance_id"], "prov_lab_001")
        self.assertEqual(d["provenance"]["source_type"], "LABORATORY_METROLOGY")

    # 7. ReferenceSignalChannel
    def test_reference_signal_channel_validation(self) -> None:
        ch = ReferenceSignalChannel(
            signal_name="cell_01_voltage_v",
            signal_type=BenchmarkSignalType.VOLTAGE,
            unit="V",
            provenance=self.valid_provenance,
            nominal_sampling_rate_hz=100.0,
            nominal_uncertainty_absolute=0.001,
        )
        self.assertEqual(ch.signal_name, "cell_01_voltage_v")
        self.assertEqual(ch.nominal_sampling_rate_hz, 100.0)
        self.assertEqual(ch.nominal_uncertainty_absolute, 0.001)

        d = ch.to_dict()
        self.assertEqual(d["signal_name"], "cell_01_voltage_v")
        self.assertEqual(d["unit"], "V")

        # Invalid channel validations
        with self.assertRaises(ValueError):
            ReferenceSignalChannel(
                signal_name="",
                signal_type=BenchmarkSignalType.VOLTAGE,
                unit="V",
                provenance=self.valid_provenance,
            )
        with self.assertRaises(ValueError):
            ReferenceSignalChannel(
                signal_name="sig",
                signal_type=BenchmarkSignalType.VOLTAGE,
                unit="V",
                provenance=self.valid_provenance,
                nominal_sampling_rate_hz=-10.0,
            )

    # 8. ExperimentalGroundTruthEvent
    def test_ground_truth_event_validation(self) -> None:
        ev = ExperimentalGroundTruthEvent(
            event_id="ev_thermal_01",
            start_timestamp_ns=10_000_000_000,
            end_timestamp_ns=25_000_000_000,
            event_type="THERMAL_DISSIPATION_IMPAIRMENT",
            target_component="FAN_BANK_01",
            injected_magnitude=0.0,
            injected_unit="RPM",
            expected_hypothesis_id="HYP_THERMAL_IMPAIRMENT",
            description="Simulated total cooling airflow failure.",
        )
        self.assertEqual(ev.event_id, "ev_thermal_01")
        self.assertEqual(ev.duration_seconds, 15.0)
        self.assertEqual(ev.expected_hypothesis_id, "HYP_THERMAL_IMPAIRMENT")

        d = ev.to_dict()
        self.assertEqual(d["duration_seconds"], 15.0)
        self.assertEqual(d["event_type"], "THERMAL_DISSIPATION_IMPAIRMENT")

        # Invalid start/end ordering
        with self.assertRaises(ValueError):
            ExperimentalGroundTruthEvent(
                event_id="ev_bad",
                start_timestamp_ns=20_000_000_000,
                end_timestamp_ns=10_000_000_000,
                event_type="FAULT",
            )
        with self.assertRaises(ValueError):
            ExperimentalGroundTruthEvent(
                event_id="",
                start_timestamp_ns=0,
                event_type="FAULT",
            )


if __name__ == "__main__":
    unittest.main()
