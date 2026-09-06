"""Integration and Orchestration unit tests for Diagnostics (Level 5.4 Phase 4)."""

import unittest
from src.diagnostics.config import DiagnosticThresholdConfig
from src.diagnostics.engine import DiagnosticEngine
from src.diagnostics.events import (
    BatteryDiagnosticAssessedEvent,
    CriticalAdvisoryAlertEvent,
)
from src.diagnostics.hypotheses import ThermalDissipationImpairmentHypothesis
from src.diagnostics.types import (
    DiagnosticCategory,
    DiagnosticEvidenceItem,
    DiagnosticSeverity,
    EvidenceEvaluationStatus,
    FaultLifecycleState,
    OperatingContext,
)
from src.domain.battery.entities import BatteryPack
from src.domain.battery.enums import BatteryChemistry, CellFormFactor
from src.domain.battery.value_objects import (
    BatteryIdentification,
    BatteryTopology,
    CellConfiguration,
    ElectricalRatings,
    PackConfiguration,
    ThermalLimits,
)
from src.estimators.ekf import ExtendedKalmanFilter
from src.events.bus import DigitalTwinEventBus
from src.models.ecm.generic_ecm import GenericECMModel
from src.models.ecm.parameters import GenericECMParameters, RCBranchParameters
from src.models.parameters.linear_ocv import LinearOCVModel
from src.models.types import ModelMetadata
from src.replay.engine import DriveCycleReplayEngine, ReplayConfig
from src.runtime.config import RuntimeConfig
from src.runtime.instance import DigitalTwinInstance
from src.services.twin_service import TwinApplicationService
from src.telemetry.enums import TelemetryQuality
from src.telemetry.snapshots import TelemetrySnapshot
from src.validation.types import (
    ModelValidationReport,
    ModelValidationState,
    SignalProvenance,
    ValidationWindowReport,
)


class TestDiagnosticsIntegration(unittest.TestCase):
    """Test suite verifying runtime stepping, event throttling, critical advisory alerts, and replay isolation."""

    def _create_test_twin(
        self,
        system_id: str = "twin_test",
        event_bus: DigitalTwinEventBus = None,
        attach_diagnostics: bool = True,
    ) -> DigitalTwinInstance:
        ident = BatteryIdentification(identifier=f"pack_{system_id}", display_name="Test Pack")
        cell_cfg = CellConfiguration(
            cell_id="cell_nmc",
            chemistry=BatteryChemistry.NMC,
            form_factor=CellFormFactor.CYLINDRICAL,
            nominal_voltage_v=3.7,
            min_voltage_v=2.8,
            max_voltage_v=4.2,
            nominal_capacity_ah=2.5,
        )
        ratings = ElectricalRatings(
            nominal_voltage_v=3.7,
            min_voltage_v=2.8,
            max_voltage_v=4.2,
            nominal_capacity_ah=2.5,
            nominal_energy_wh=9.25,
            max_continuous_charge_current_a=2.5,
            max_continuous_discharge_current_a=5.0,
            peak_charge_current_a=5.0,
            peak_discharge_current_a=10.0,
        )
        limits = ThermalLimits(
            min_charge_temp_c=0.0,
            max_charge_temp_c=45.0,
            min_discharge_temp_c=-20.0,
            max_discharge_temp_c=60.0,
            warning_temp_c=60.0,
            critical_temp_c=80.0,
        )
        cfg_pack = PackConfiguration(
            pack_id=f"pack_{system_id}",
            topology=BatteryTopology(series_count=1, parallel_count=1),
            electrical_ratings=ratings,
            thermal_limits=limits,
        )
        pack = BatteryPack.create_monolithic_pack(
            identification=ident,
            configuration=cfg_pack,
            cell_config=cell_cfg,
        )

        ocv = LinearOCVModel(v_min_v=2.8, v_max_v=4.2)
        params = GenericECMParameters(
            nominal_capacity_ah=2.5,
            nominal_voltage_v=3.7,
            series_resistance_r0_ohm=0.02,
            rc_branches=(RCBranchParameters(resistance_r_ohm=0.01, capacitance_c_farad=1000.0),),
        )
        meta = ModelMetadata(model_id=f"ecm_{system_id}", name="ECM Test", paradigm="EQUIVALENT_CIRCUIT")
        model = GenericECMModel(metadata=meta, parameters=params, ocv_model=ocv)
        estimator = ExtendedKalmanFilter(estimator_id=f"ekf_{system_id}", parameters=params, ocv_model=ocv)

        diag_engine = DiagnosticEngine(system_id=system_id) if attach_diagnostics else None

        cfg = RuntimeConfig(system_id=system_id, auto_publish_events=True)
        twin = DigitalTwinInstance(
            battery_pack=pack,
            battery_model=model,
            state_estimator=estimator,
            diagnostic_engine=diag_engine,
            event_bus=event_bus,
            config=cfg,
        )
        twin.initialize(initial_soc=0.90, initial_soh=1.0, temperature_c=25.0)
        return twin

    def test_runtime_stepping_produces_diagnostic_report_and_event(self) -> None:
        """DigitalTwinInstance.step() executes DiagnosticEngine and publishes BatteryDiagnosticAssessedEvent."""
        bus = DigitalTwinEventBus()
        published_events = []
        bus.subscribe(event_type="twin.diagnostic", handler=lambda e: published_events.append(e))

        twin = self._create_test_twin(system_id="twin_step", event_bus=bus)
        snap = TelemetrySnapshot(
            system_id="twin_step",
            snapshot_id="snap_1",
            timestamp_ns=1_000_000_000,
            pack_voltage_v=3.95,
            pack_current_a=2.0,
            avg_cell_temperature_c=25.5,
        )

        twin.step(snap)

        # Assert report populated
        self.assertIsNotNone(twin.latest_diagnostic_report)
        self.assertEqual(twin.latest_diagnostic_report.system_id, "twin_step")
        self.assertEqual(twin.latest_diagnostic_report.timestamp_ns, 1_000_000_000)

        # Assert event published
        self.assertEqual(len(published_events), 1)
        self.assertIsInstance(published_events[0], BatteryDiagnosticAssessedEvent)
        self.assertEqual(published_events[0].source_id, "twin_step")

    def test_event_throttling_steady_state_heartbeat(self) -> None:
        """Steady-state execution publishes on initial assessment, lifecycle transitions, and periodic heartbeat."""
        bus = DigitalTwinEventBus()
        diag_events = []
        bus.subscribe(event_type="twin.diagnostic", handler=lambda e: diag_events.append(e))

        twin = self._create_test_twin(system_id="twin_throttle", event_bus=bus)

        # Step 105 times with identical steady state
        for i in range(1, 106):
            snap = TelemetrySnapshot(
                system_id="twin_throttle",
                snapshot_id=f"snap_{i}",
                timestamp_ns=i * 1_000_000_000,
                pack_voltage_v=3.90,
                pack_current_a=1.0,
                avg_cell_temperature_c=25.0,
            )
            twin.step(snap)

        # Expected: Step 1 (initial assessment), Step 5 (debounce transition to DIAGNOSED), Step 101 (heartbeat) -> total 3 events
        self.assertEqual(len(diag_events), 3)
        self.assertEqual(diag_events[0].timestamp_ns, 1_000_000_000)
        self.assertEqual(diag_events[1].timestamp_ns, 5 * 1_000_000_000)
        self.assertEqual(diag_events[2].timestamp_ns, 101 * 1_000_000_000)

    def test_critical_advisory_alert_event_publication(self) -> None:
        """DIAGNOSED_CRITICAL condition publishes both BatteryDiagnosticAssessedEvent and CriticalAdvisoryAlertEvent."""
        bus = DigitalTwinEventBus()
        crit_events = []
        bus.subscribe(event_type="twin.diagnostic.critical_advisory", handler=lambda e: crit_events.append(e))

        cfg = DiagnosticThresholdConfig(persistence_debounce_steps=1, critical_min_corroborating_channels=2)
        engine = DiagnosticEngine(system_id="twin_crit", config=cfg, hypotheses=[ThermalDissipationImpairmentHypothesis()])

        # Validated model report
        win = ValidationWindowReport(
            window_id="w_val",
            system_id="twin_crit",
            start_timestamp_ns=0,
            end_timestamp_ns=1_000_000_000,
            duration_s=1.0,
            sample_count=20,
            state=ModelValidationState.VALIDATED,
        )
        val_report = ModelValidationReport(system_id="twin_crit", timestamp_ns=1_000_000_000, active_window=win)

        cfg_pack = PackConfiguration(
            pack_id="pack_crit",
            topology=BatteryTopology(series_count=1, parallel_count=1),
            electrical_ratings=ElectricalRatings(
                nominal_voltage_v=3.7,
                min_voltage_v=2.8,
                max_voltage_v=4.2,
                nominal_capacity_ah=2.5,
                nominal_energy_wh=9.25,
                max_continuous_charge_current_a=2.5,
                max_continuous_discharge_current_a=5.0,
                peak_charge_current_a=5.0,
                peak_discharge_current_a=10.0,
            ),
            thermal_limits=ThermalLimits(min_charge_temp_c=0.0, max_charge_temp_c=45.0, min_discharge_temp_c=-20.0, max_discharge_temp_c=60.0, warning_temp_c=60.0, critical_temp_c=80.0),
        )
        ident = BatteryIdentification(identifier="pack_crit", display_name="Crit Test Pack")
        cell_cfg = CellConfiguration(
            cell_id="cell_nmc",
            chemistry=BatteryChemistry.NMC,
            form_factor=CellFormFactor.CYLINDRICAL,
            nominal_voltage_v=3.7,
            min_voltage_v=2.8,
            max_voltage_v=4.2,
            nominal_capacity_ah=2.5,
        )
        pack = BatteryPack.create_monolithic_pack(
            identification=ident,
            configuration=cfg_pack,
            cell_config=cell_cfg,
        )
        ocv = LinearOCVModel(v_min_v=2.8, v_max_v=4.2)
        params = GenericECMParameters(nominal_capacity_ah=2.5, nominal_voltage_v=3.7, series_resistance_r0_ohm=0.02)
        meta = ModelMetadata(model_id="ecm_crit", name="ECM Crit", paradigm="EQUIVALENT_CIRCUIT")
        model = GenericECMModel(metadata=meta, parameters=params, ocv_model=ocv)

        twin = DigitalTwinInstance(
            battery_pack=pack,
            battery_model=model,
            diagnostic_engine=engine,
            event_bus=bus,
            config=RuntimeConfig(system_id="twin_crit", auto_publish_events=True),
        )
        twin._latest_validation_report = val_report

        # Step with elevated thermal signals to trigger DIAGNOSED_CRITICAL
        snap = TelemetrySnapshot(
            system_id="twin_crit",
            snapshot_id="snap_crit",
            timestamp_ns=1_000_000_000,
            pack_voltage_v=3.5,
            pack_current_a=2.0,
            avg_cell_temperature_c=55.0,
            ambient_temperature_c=25.0,
        )
        twin.step(snap)

        self.assertEqual(len(crit_events), 1)
        self.assertEqual(crit_events[0].source_id, "twin_crit")
        self.assertEqual(crit_events[0].payload["lifecycle_state"], "DIAGNOSED_CRITICAL")

    def test_critical_event_flooding_prevention_and_lifecycle_transitions(self) -> None:
        """Verifies no event flooding in DIAGNOSED_CRITICAL, re-entry alert permission, and heartbeat bounds."""
        bus = DigitalTwinEventBus()
        diag_events = []
        crit_events = []
        bus.subscribe(event_type="twin.diagnostic", handler=lambda e: diag_events.append(e))
        bus.subscribe(event_type="twin.diagnostic.critical_advisory", handler=lambda e: crit_events.append(e))

        cfg = DiagnosticThresholdConfig(
            persistence_debounce_steps=1,
            recovery_hysteresis_steps=1,
            critical_min_corroborating_channels=2,
        )
        engine = DiagnosticEngine(system_id="twin_flood", config=cfg, hypotheses=[ThermalDissipationImpairmentHypothesis()])

        win = ValidationWindowReport(
            window_id="w_val",
            system_id="twin_flood",
            start_timestamp_ns=0,
            end_timestamp_ns=1_000_000_000,
            duration_s=1.0,
            sample_count=20,
            state=ModelValidationState.VALIDATED,
        )
        val_report = ModelValidationReport(system_id="twin_flood", timestamp_ns=1_000_000_000, active_window=win)

        cfg_pack = PackConfiguration(
            pack_id="pack_flood",
            topology=BatteryTopology(series_count=1, parallel_count=1),
            electrical_ratings=ElectricalRatings(
                nominal_voltage_v=3.7,
                min_voltage_v=2.8,
                max_voltage_v=4.2,
                nominal_capacity_ah=2.5,
                nominal_energy_wh=9.25,
                max_continuous_charge_current_a=2.5,
                max_continuous_discharge_current_a=5.0,
                peak_charge_current_a=5.0,
                peak_discharge_current_a=10.0,
            ),
            thermal_limits=ThermalLimits(
                min_charge_temp_c=0.0,
                max_charge_temp_c=45.0,
                min_discharge_temp_c=-20.0,
                max_discharge_temp_c=60.0,
                warning_temp_c=60.0,
                critical_temp_c=80.0,
            ),
        )
        ident = BatteryIdentification(identifier="pack_flood", display_name="Flood Test Pack")
        cell_cfg = CellConfiguration(
            cell_id="cell_nmc",
            chemistry=BatteryChemistry.NMC,
            form_factor=CellFormFactor.CYLINDRICAL,
            nominal_voltage_v=3.7,
            min_voltage_v=2.8,
            max_voltage_v=4.2,
            nominal_capacity_ah=2.5,
        )
        pack = BatteryPack.create_monolithic_pack(
            identification=ident,
            configuration=cfg_pack,
            cell_config=cell_cfg,
        )
        ocv = LinearOCVModel(v_min_v=2.8, v_max_v=4.2)
        params = GenericECMParameters(nominal_capacity_ah=2.5, nominal_voltage_v=3.7, series_resistance_r0_ohm=0.02)
        meta = ModelMetadata(model_id="ecm_flood", name="ECM Flood", paradigm="EQUIVALENT_CIRCUIT")
        model = GenericECMModel(metadata=meta, parameters=params, ocv_model=ocv)

        twin = DigitalTwinInstance(
            battery_pack=pack,
            battery_model=model,
            diagnostic_engine=engine,
            event_bus=bus,
            config=RuntimeConfig(system_id="twin_flood", auto_publish_events=True),
        )
        twin._latest_validation_report = val_report

        # 1. NORMAL -> DIAGNOSED_CRITICAL: exactly one immediate critical advisory
        snap_crit_1 = TelemetrySnapshot(
            system_id="twin_flood",
            snapshot_id="snap_crit_1",
            timestamp_ns=1_000_000_000,
            pack_voltage_v=3.5,
            pack_current_a=2.0,
            avg_cell_temperature_c=55.0,
            ambient_temperature_c=25.0,
        )
        twin.step(snap_crit_1)
        self.assertEqual(len(crit_events), 1)
        self.assertEqual(len(diag_events), 1)

        # 2. DIAGNOSED_CRITICAL -> DIAGNOSED_CRITICAL (sustained over 10 steps): NO repeated critical advisory every step
        for i in range(2, 12):
            snap = TelemetrySnapshot(
                system_id="twin_flood",
                snapshot_id=f"snap_crit_{i}",
                timestamp_ns=i * 1_000_000_000,
                pack_voltage_v=3.5,
                pack_current_a=2.0,
                avg_cell_temperature_c=55.0,
                ambient_temperature_c=25.0,
            )
            twin.step(snap)

        # Total crit events must still be exactly 1 (no flood!), diag events still 1 (no steady-state flood)
        self.assertEqual(len(crit_events), 1)
        self.assertEqual(len(diag_events), 1)

        # 3. DIAGNOSED_CRITICAL -> DIAGNOSED (e.g. Model Validation degrades): no critical advisory, but immediate assessment on lifecycle change
        val_report_degraded = ModelValidationReport(
            system_id="twin_flood",
            timestamp_ns=12 * 1_000_000_000,
            active_window=ValidationWindowReport(
                window_id="w_deg",
                system_id="twin_flood",
                start_timestamp_ns=0,
                end_timestamp_ns=12 * 1_000_000_000,
                duration_s=12.0,
                sample_count=20,
                state=ModelValidationState.DEGRADED,
            ),
        )
        twin._latest_validation_report = val_report_degraded

        snap_nominal = TelemetrySnapshot(
            system_id="twin_flood",
            snapshot_id="snap_nom_12",
            timestamp_ns=12 * 1_000_000_000,
            pack_voltage_v=3.9,
            pack_current_a=0.0,
            avg_cell_temperature_c=25.0,
            ambient_temperature_c=25.0,
        )
        twin.step(snap_nominal)
        self.assertEqual(twin.latest_diagnostic_report.lifecycle_state, FaultLifecycleState.DIAGNOSED)
        self.assertEqual(len(crit_events), 1)  # No new critical event
        self.assertEqual(len(diag_events), 2)  # 1 new assessment event due to state transition

        # 4. DIAGNOSED -> DIAGNOSED_CRITICAL again (Model Validation restored to VALIDATED): a new critical advisory IS permitted
        twin._latest_validation_report = val_report  # Validated state restored
        snap_crit_reenter = TelemetrySnapshot(
            system_id="twin_flood",
            snapshot_id="snap_crit_13",
            timestamp_ns=13 * 1_000_000_000,
            pack_voltage_v=3.5,
            pack_current_a=2.0,
            avg_cell_temperature_c=55.0,
            ambient_temperature_c=25.0,
        )
        twin.step(snap_crit_reenter)
        self.assertEqual(twin.latest_diagnostic_report.lifecycle_state, FaultLifecycleState.DIAGNOSED_CRITICAL)
        self.assertEqual(len(crit_events), 2)  # Re-entering permits exactly 1 new critical advisory
        self.assertEqual(len(diag_events), 3)  # Assessment event published on transition

    def test_service_diagnostic_history_bounding_maxlen_100(self) -> None:
        """TwinApplicationService bounds diagnostic history to maximum 100 entries per twin."""
        service = TwinApplicationService()
        twin = self._create_test_twin(system_id="twin_bounded")
        service.register_twin_instance(twin)

        # Step 150 times
        for i in range(1, 151):
            snap = TelemetrySnapshot(
                system_id="twin_bounded",
                snapshot_id=f"snap_{i}",
                timestamp_ns=i * 1_000_000_000,
                pack_voltage_v=3.90,
                pack_current_a=1.0,
            )
            service.step_twin("twin_bounded", snap)

        history = service.get_twin_diagnostic_history("twin_bounded")
        self.assertEqual(len(history), 100)
        # Oldest kept record is index 51, newest is 150
        self.assertEqual(history[0].timestamp_ns, 51 * 1_000_000_000)
        self.assertEqual(history[-1].timestamp_ns, 150 * 1_000_000_000)

    def test_data_gapped_context_vs_data_quality_failed_semantics(self) -> None:
        """dt > gap_threshold is OperatingContext.DATA_GAPPED and INSUFFICIENT_EVIDENCE, not DATA_QUALITY_FAILED."""
        twin = self._create_test_twin(system_id="twin_gap")
        s0 = TelemetrySnapshot(system_id="twin_gap", snapshot_id="s0", timestamp_ns=0, pack_current_a=1.0)
        twin.step(s0)

        # 20 second gap (> 5s threshold)
        s1 = TelemetrySnapshot(system_id="twin_gap", snapshot_id="s1", timestamp_ns=20 * 1_000_000_000, pack_current_a=1.0)
        twin.step(s1)

        rep = twin.latest_diagnostic_report
        self.assertIsNotNone(rep)
        self.assertEqual(rep.operating_context, OperatingContext.DATA_GAPPED)
        self.assertEqual(rep.lifecycle_state, FaultLifecycleState.INSUFFICIENT_EVIDENCE)
        self.assertEqual(rep.data_quality_status, "DATA_GAPPED")

        # Now test corrupt telemetry (TelemetryQuality.INVALID) -> DATA_QUALITY_FAILED
        s_corrupt = TelemetrySnapshot(
            system_id="twin_gap",
            snapshot_id="s_corrupt",
            timestamp_ns=21 * 1_000_000_000,
            pack_voltage_v=3.9,
            pack_current_a=1.0,
            quality=TelemetryQuality.INVALID,
        )
        twin.step(s_corrupt)
        rep_corrupt = twin.latest_diagnostic_report
        self.assertEqual(rep_corrupt.lifecycle_state, FaultLifecycleState.DATA_QUALITY_FAILED)
        self.assertEqual(rep_corrupt.data_quality_status, "FAILED")

    def test_replay_sandbox_isolation_leaves_live_diagnostics_unmutated(self) -> None:
        """DriveCycleReplayEngine collects diagnostic reports without mutating live twin diagnostic state."""
        live_twin = self._create_test_twin(system_id="twin_live")
        snap_live = TelemetrySnapshot(
            system_id="twin_live",
            snapshot_id="live_01",
            timestamp_ns=1_000_000_000,
            pack_voltage_v=3.95,
            pack_current_a=1.5,
        )
        live_twin.step(snap_live)
        live_report_before = live_twin.latest_diagnostic_report

        # Create replay sandbox instance
        replay_twin = self._create_test_twin(system_id="twin_replay_sandbox")
        replay_snaps = [
            TelemetrySnapshot(system_id="twin_replay_sandbox", snapshot_id=f"rep_{i}", timestamp_ns=i * 1_000_000_000, pack_voltage_v=3.80, pack_current_a=2.0)
            for i in range(1, 6)
        ]

        engine = DriveCycleReplayEngine()
        res = engine.replay_snapshots(replay_twin, replay_snaps, config=ReplayConfig(auto_reset_instance=True))

        # Check replay result captured diagnostics
        self.assertEqual(len(res.diagnostic_reports), 5)
        self.assertEqual(res.to_dict()["diagnostic_reports_count"], 5)

        # Verify live twin was completely untouched
        self.assertEqual(live_twin.latest_diagnostic_report.timestamp_ns, live_report_before.timestamp_ns)
        self.assertEqual(live_twin.total_steps, 1)

    def test_reset_diagnostics_subsystem_isolation(self) -> None:
        """reset_diagnostics() resets only diagnostic report and engine, leaving other states intact."""
        twin = self._create_test_twin(system_id="twin_iso")
        snap = TelemetrySnapshot(
            system_id="twin_iso",
            snapshot_id="s1",
            timestamp_ns=1_000_000_000,
            pack_voltage_v=3.90,
            pack_current_a=2.0,
            avg_cell_temperature_c=25.0,
        )
        twin.step(snap)

        # Cache pre-reset states
        model_state_before = twin.current_model_state
        estimator_state_before = twin.current_estimation_state
        steps_before = twin.total_steps
        self.assertIsNotNone(twin.latest_diagnostic_report)

        # Call reset_diagnostics
        twin.reset_diagnostics()

        # Diagnostic state reset
        self.assertIsNone(twin.latest_diagnostic_report)

        # Other states completely preserved
        self.assertEqual(twin.current_model_state, model_state_before)
        self.assertEqual(twin.current_estimation_state, estimator_state_before)
        self.assertEqual(twin.total_steps, steps_before)


if __name__ == "__main__":
    unittest.main()
