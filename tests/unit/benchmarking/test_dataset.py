"""Unit tests for Level 5.5 Phase 1 Dataset Abstraction, Observation Frames, and Builders."""

import unittest

from src.benchmarking.dataset import (
    DatasetBuilder,
    DatasetMetadata,
    ExperimentalDataset,
    ReferenceObservationFrame,
)
from src.benchmarking.types import (
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
from src.domain.battery.enums import BatteryChemistry, CellFormFactor
from src.domain.battery.value_objects import (
    BatteryIdentification,
    BatteryTopology,
    CellConfiguration,
    ElectricalRatings,
    PackConfiguration,
    ThermalLimits,
)


class TestBenchmarkingDataset(unittest.TestCase):
    """Test suite for Level 5.5 ReferenceObservationFrame, ExperimentalDataset, and DatasetBuilder."""

    def setUp(self) -> None:
        self.provenance = ReferenceProvenance(
            provenance_id="prov_lab_001",
            source_type=ReferenceSourceType.LABORATORY_METROLOGY,
            calibration_status=CalibrationStatus.CERTIFIED_CALIBRATED,
            evidence_role=EvidenceRole.PHYSICAL_REFERENCE,
            measurement_method="FOUR_WIRE_KELVIN",
            instrument_identifier="MACCOR_SERIES_4000",
        )
        self.metadata = DatasetMetadata(
            dataset_id="ds_wltp_lab_001",
            dataset_name="WLTP Drive Cycle Validation 25C",
            version="1.0.0",
            created_at_utc="2026-09-04T00:00:00Z",
            facility_or_provider="Battery Metrology Institute",
            operator_name="Dr. E. Faraday",
            test_bench_identifier="BENCH_ALPHA",
            description="High precision 100Hz lab reference data under WLTP excitation.",
        )
        # Universal battery configuration (chemistry-agnostic)
        self.battery_ident = BatteryIdentification(identifier="pack_test_01", display_name="Test Bench Pack")
        self.cell_cfg = CellConfiguration(
            cell_id="cell_lfp_generic",
            chemistry=BatteryChemistry.LFP,
            form_factor=CellFormFactor.PRISMATIC,
            nominal_voltage_v=3.2,
            min_voltage_v=2.5,
            max_voltage_v=3.65,
            nominal_capacity_ah=100.0,
        )
        self.pack_cfg = PackConfiguration(
            pack_id="pack_test_01",
            topology=BatteryTopology(series_count=16, parallel_count=1),
            electrical_ratings=ElectricalRatings(
                nominal_voltage_v=51.2,
                min_voltage_v=40.0,
                max_voltage_v=58.4,
                nominal_capacity_ah=100.0,
                nominal_energy_wh=5120.0,
                max_continuous_charge_current_a=100.0,
                max_continuous_discharge_current_a=200.0,
                peak_charge_current_a=200.0,
                peak_discharge_current_a=400.0,
            ),
            thermal_limits=ThermalLimits(
                min_charge_temp_c=0.0,
                max_charge_temp_c=45.0,
                min_discharge_temp_c=-20.0,
                max_discharge_temp_c=60.0,
                warning_temp_c=60.0,
                critical_temp_c=75.0,
            ),
        )

    # 1. ReferenceObservationFrame tests
    def test_observation_frame_construction(self) -> None:
        meas_v = ReferenceMeasurement(
            value=52.412,
            unit="V",
            timestamp_ns=1_000_000_000,
            uncertainty_absolute=0.005,
            provenance=self.provenance,
            signal_type=BenchmarkSignalType.VOLTAGE,
        )
        frame = ReferenceObservationFrame(
            timestamp_ns=1_000_000_000,
            observed_at_ns=999_999_900,
            sequence_number=1,
            reference_signals={"pack_voltage_v": meas_v},
            field_signals={"bms_voltage_v": 52.40, "pack_current_a": 10.5},
            bms_estimates={"bms_soc": 0.85},
            operating_context_hint="DISCHARGE_CC",
            quality_status=ReferenceDataQuality.VALID,
        )
        self.assertEqual(frame.timestamp_ns, 1_000_000_000)
        self.assertEqual(frame.timestamp_seconds, 1.0)
        self.assertEqual(frame.observed_at_ns, 999_999_900)
        self.assertEqual(frame.sequence_number, 1)
        self.assertEqual(frame.get_reference_value("pack_voltage_v"), 52.412)
        self.assertEqual(frame.get_field_value("pack_current_a"), 10.5)
        self.assertEqual(frame.get_bms_estimate("bms_soc"), 0.85)
        self.assertIsNone(frame.get_reference_value("missing_signal"))

        d = frame.to_dict()
        self.assertEqual(d["timestamp_ns"], 1_000_000_000)
        self.assertIn("pack_voltage_v", d["reference_signals"])
        self.assertEqual(d["field_signals"]["pack_current_a"], 10.5)

    def test_observation_frame_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            ReferenceObservationFrame(timestamp_ns=-1)
        with self.assertRaises(ValueError):
            ReferenceObservationFrame(timestamp_ns=0, observed_at_ns=-1)
        with self.assertRaises(ValueError):
            ReferenceObservationFrame(timestamp_ns=0, sequence_number=-1)
        with self.assertRaises(ValueError):
            ReferenceObservationFrame(
                timestamp_ns=0,
                field_signals={"current_a": float("nan")},
            )
        with self.assertRaises(ValueError):
            ReferenceObservationFrame(
                timestamp_ns=0,
                bms_estimates={"soc": float("inf")},
            )
        with self.assertRaises(TypeError):
            ReferenceObservationFrame(
                timestamp_ns=0,
                reference_signals={"voltage": 52.0},  # type: ignore (must be ReferenceMeasurement)
            )

    # 2. DatasetMetadata tests
    def test_dataset_metadata_validation(self) -> None:
        meta = self.metadata
        self.assertEqual(meta.dataset_id, "ds_wltp_lab_001")
        self.assertEqual(meta.version, "1.0.0")
        d = meta.to_dict()
        self.assertEqual(d["dataset_id"], "ds_wltp_lab_001")

        with self.assertRaises(ValueError):
            DatasetMetadata(dataset_id="", dataset_name="Name", version="1.0", created_at_utc="2026-01-01")

    # 3. ExperimentalDataset construction, immutability, and queries
    def test_experimental_dataset_construction_and_immutability(self) -> None:
        ch_v = ReferenceSignalChannel(
            signal_name="pack_voltage_v",
            signal_type=BenchmarkSignalType.VOLTAGE,
            unit="V",
            provenance=self.provenance,
            nominal_sampling_rate_hz=10.0,
            nominal_uncertainty_absolute=0.005,
        )
        meas_v1 = ReferenceMeasurement(value=52.0, unit="V", timestamp_ns=1_000_000_000, provenance=self.provenance)
        meas_v2 = ReferenceMeasurement(value=51.9, unit="V", timestamp_ns=2_000_000_000, provenance=self.provenance)
        meas_v3 = ReferenceMeasurement(value=51.8, unit="V", timestamp_ns=3_000_000_000, provenance=self.provenance)

        f1 = ReferenceObservationFrame(timestamp_ns=1_000_000_000, reference_signals={"pack_voltage_v": meas_v1})
        f2 = ReferenceObservationFrame(timestamp_ns=2_000_000_000, reference_signals={"pack_voltage_v": meas_v2})
        f3 = ReferenceObservationFrame(timestamp_ns=3_000_000_000, reference_signals={"pack_voltage_v": meas_v3})

        ev = ExperimentalGroundTruthEvent(
            event_id="ev_pulse_01",
            start_timestamp_ns=1_500_000_000,
            end_timestamp_ns=2_500_000_000,
            event_type="DISCHARGE_PULSE",
        )

        dataset = ExperimentalDataset(
            metadata=self.metadata,
            battery_identification=self.battery_ident,
            pack_configuration=self.pack_cfg,
            cell_configuration=self.cell_cfg,
            reference_channels=(ch_v,),
            frames=(f1, f2, f3),
            ground_truth_events=(ev,),
        )

        self.assertEqual(dataset.dataset_id, "ds_wltp_lab_001")
        self.assertEqual(dataset.total_frames, 3)
        self.assertEqual(dataset.start_timestamp_ns, 1_000_000_000)
        self.assertEqual(dataset.end_timestamp_ns, 3_000_000_000)
        self.assertEqual(dataset.duration_seconds, 2.0)
        self.assertEqual(dataset.available_signals, ("pack_voltage_v",))
        self.assertEqual(dataset.ground_truth_event_count, 1)

        # Immutability
        with self.assertRaises(AttributeError):
            dataset.metadata = None  # type: ignore

        # Frame query
        self.assertEqual(dataset.get_frame(1).timestamp_ns, 2_000_000_000)
        with self.assertRaises(IndexError):
            dataset.get_frame(10)

        # Serialization
        d = dataset.to_dict()
        self.assertEqual(d["total_frames"], 3)
        self.assertEqual(d["duration_seconds"], 2.0)

    # 4. Monotonicity and duplicate timestamp validation
    def test_dataset_monotonicity_and_duplicate_timestamp_validation(self) -> None:
        f1 = ReferenceObservationFrame(timestamp_ns=1_000_000_000)
        f2_retrograde = ReferenceObservationFrame(timestamp_ns=500_000_000)  # Retrograde
        f2_duplicate = ReferenceObservationFrame(timestamp_ns=1_000_000_000)   # Duplicate

        # Retrograde timestamps strictly rejected when require_monotonic_timestamps=True
        with self.assertRaises(ValueError):
            ExperimentalDataset(
                metadata=self.metadata,
                frames=(f1, f2_retrograde),
                require_monotonic_timestamps=True,
            )

        # Duplicate timestamps rejected by default when allow_duplicate_timestamps=False
        with self.assertRaises(ValueError):
            ExperimentalDataset(
                metadata=self.metadata,
                frames=(f1, f2_duplicate),
                require_monotonic_timestamps=True,
                allow_duplicate_timestamps=False,
            )

        # Duplicate timestamps permitted when allow_duplicate_timestamps=True
        ds_dup = ExperimentalDataset(
            metadata=self.metadata,
            frames=(f1, f2_duplicate),
            require_monotonic_timestamps=True,
            allow_duplicate_timestamps=True,
        )
        self.assertEqual(ds_dup.total_frames, 2)

    # 5. Time range query slicing
    def test_query_time_range_slicing(self) -> None:
        frames = tuple(
            ReferenceObservationFrame(timestamp_ns=i * 1_000_000_000)
            for i in range(1, 11)
        )
        ev1 = ExperimentalGroundTruthEvent(event_id="e1", start_timestamp_ns=2_000_000_000, end_timestamp_ns=4_000_000_000, event_type="PULSE")
        ev2 = ExperimentalGroundTruthEvent(event_id="e2", start_timestamp_ns=7_000_000_000, end_timestamp_ns=9_000_000_000, event_type="PULSE")

        ds = ExperimentalDataset(
            metadata=self.metadata,
            frames=frames,
            ground_truth_events=(ev1, ev2),
        )

        sliced = ds.query_time_range(
            start_time_ns=3_000_000_000,
            end_time_ns=6_000_000_000,
        )
        self.assertEqual(sliced.total_frames, 4)
        self.assertEqual(sliced.start_timestamp_ns, 3_000_000_000)
        self.assertEqual(sliced.end_timestamp_ns, 6_000_000_000)
        self.assertEqual(sliced.ground_truth_event_count, 1)  # Only ev1 overlaps
        self.assertEqual(sliced.ground_truth_events[0].event_id, "e1")

    # 6. DatasetBuilder fluent construction
    def test_dataset_builder(self) -> None:
        builder = DatasetBuilder(metadata=self.metadata)
        builder.set_battery_configuration(
            identification=self.battery_ident,
            pack_config=self.pack_cfg,
            cell_config=self.cell_cfg,
        )
        ch_v = ReferenceSignalChannel(
            signal_name="voltage_v",
            signal_type=BenchmarkSignalType.VOLTAGE,
            unit="V",
            provenance=self.provenance,
        )
        builder.add_reference_channel(ch_v)

        frames = [
            ReferenceObservationFrame(
                timestamp_ns=i * 1_000_000_000,
                reference_signals={"voltage_v": ReferenceMeasurement(value=50.0 - i * 0.1, unit="V", timestamp_ns=i * 1_000_000_000)},
            )
            for i in range(1, 6)
        ]
        builder.add_frames(frames)
        builder.add_ground_truth_event(
            ExperimentalGroundTruthEvent(
                event_id="ev_rest",
                start_timestamp_ns=1_000_000_000,
                end_timestamp_ns=5_000_000_000,
                event_type="NOMINAL_REST",
            )
        )
        builder.set_timestamp_policy(require_monotonic=True, allow_duplicates=False)

        dataset = builder.build()
        self.assertEqual(dataset.total_frames, 5)
        self.assertEqual(dataset.pack_configuration.pack_id, "pack_test_01")
        self.assertEqual(dataset.ground_truth_event_count, 1)


if __name__ == "__main__":
    unittest.main()
