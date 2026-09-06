"""Historical Linear EOL-Crossing RUL Estimator.

Implements a transparent, deterministic baseline RUL estimation algorithm that calculates
the linear extrapolation crossing point against a configured End-of-Life (EOL) degradation threshold.

Note: No empirical RUL model or machine-learning RUL model was implemented.
Ordinary Least Squares (OLS) or endpoint estimation is used strictly as a descriptive historical
trend estimator for the transparent linear baseline. It is not a learned universal battery model,
an ML model, a chemistry-specific degradation model, or a validated physical lifetime model.
"""

import datetime
import math
from typing import Any, Mapping, Optional, Sequence, Tuple
import uuid

from src.prognostics.estimators.base import (
    AbstractRULEstimator,
    EstimatorConfig,
)
from src.prognostics.horizon import PrognosticHorizon
from src.prognostics.prediction import (
    EOLDefinition,
    PrognosticMethodIdentity,
    RULPredictionResult,
)
from src.prognostics.trajectory import DegradationObservation, DegradationTrajectory
from src.prognostics.trajectory_types import TrajectoryQuantityType
from src.prognostics.types import (
    EOLCriterionType,
    HorizonDomain,
    PrognosticStatus,
)


class HistoricalLinearRULEstimator(AbstractRULEstimator):
    """Deterministic linear historical trajectory EOL-crossing RUL estimator.

    Solves for the future domain point $x_{eol}$ where the linear trajectory projection:
        y(x) = y_{current} + m * (x - x_{current})
    crosses the configured End-of-Life boundary $y_{eol}$, yielding:
        RUL = (y_{current} - y_{eol}) / degradation_rate  (for decreasing quantities)
        RUL = (y_{eol} - y_{current}) / growth_rate       (for increasing quantities)

    Causality Guarantee:
        When `prediction_cutoff_timestamp_ns` is configured in `EstimatorConfig`, only observations
        with `timestamp_ns <= prediction_cutoff_timestamp_ns` are evaluated. Future observations
        are strictly ignored and cannot contaminate historical trend estimation.
    """

    def __init__(self) -> None:
        self._method_identity = PrognosticMethodIdentity(
            method_id="historical_linear_eol_crossing",
            method_version="1.0.0",
            model_id="first_order_linear_extrapolation",
            model_version="1.0.0",
            algorithm_paradigm="DETERMINISTIC_LINEAR_EXTRAPOLATION",
            parameters={"extrapolation_model": "first_order_linear_crossing"},
        )

    @property
    def method_identity(self) -> PrognosticMethodIdentity:
        return self._method_identity

    @property
    def supported_quantities(self) -> Sequence[TrajectoryQuantityType]:
        return (
            TrajectoryQuantityType.CAPACITY,
            TrajectoryQuantityType.CAPACITY_RETENTION,
            TrajectoryQuantityType.INTERNAL_RESISTANCE,
            TrajectoryQuantityType.RESISTANCE_GROWTH_RATIO,
            TrajectoryQuantityType.MULTIVARIATE,
        )

    @property
    def supported_eol_criteria(self) -> Sequence[EOLCriterionType]:
        return (
            EOLCriterionType.CAPACITY_RETENTION_THRESHOLD,
            EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD,
            EOLCriterionType.RESISTANCE_GROWTH_RATIO_THRESHOLD,
            EOLCriterionType.ABSOLUTE_RESISTANCE_THRESHOLD,
        )

    @property
    def supported_domains(self) -> Sequence[HorizonDomain]:
        return (
            HorizonDomain.CYCLE,
            HorizonDomain.TIME,
            HorizonDomain.AMPERE_HOUR_THROUGHPUT,
            HorizonDomain.WATT_HOUR_THROUGHPUT,
        )

    def estimate(
        self,
        trajectory: DegradationTrajectory,
        eol_definition: EOLDefinition,
        horizon: Optional[PrognosticHorizon] = None,
        config: Optional[EstimatorConfig] = None,
    ) -> RULPredictionResult:
        """Evaluates deterministic linear EOL crossing RUL prediction."""
        if not isinstance(trajectory, DegradationTrajectory):
            raise TypeError(f"Expected DegradationTrajectory, got {type(trajectory).__name__}.")
        if not isinstance(eol_definition, EOLDefinition):
            raise TypeError(f"Expected EOLDefinition, got {type(eol_definition).__name__}.")
        if horizon is not None and not isinstance(horizon, PrognosticHorizon):
            raise TypeError(f"Expected PrognosticHorizon, got {type(horizon).__name__}.")
        if config is not None and not isinstance(config, EstimatorConfig):
            raise TypeError(f"Expected EstimatorConfig, got {type(config).__name__}.")

        cfg = config or EstimatorConfig()
        pred_id = f"pred_lin_{trajectory.trajectory_id}_{uuid.uuid4().hex[:8]}"
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Determine effective prediction timestamp respecting causality cutoff
        cutoff_ns = cfg.prediction_cutoff_timestamp_ns
        if cutoff_ns is not None:
            pred_ts = cutoff_ns
        else:
            pred_ts = trajectory.end_timestamp_ns or 0

        # 1. Map EOL Criterion to Target Trajectory Quantity
        mapping = self._map_eol_criterion(eol_definition.criterion_type)
        if mapping is None:
            return self._build_non_valid_result(
                prediction_id=pred_id,
                prediction_timestamp_ns=pred_ts,
                created_at_utc=now_utc,
                system_id=trajectory.system_id,
                status=PrognosticStatus.INSUFFICIENT_DATA,
                rul_domain=horizon.domain if horizon else HorizonDomain.CYCLE,
                rul_unit=horizon.unit if horizon else "cycles",
                eol_definition=eol_definition,
                diagnostic_message=(
                    f"EOL criterion '{eol_definition.criterion_type.value}' is not supported by "
                    f"the historical linear estimator."
                ),
            )

        field_name, is_decreasing, unit_y = mapping

        # 2. Resolve Forecast Domain & Unit
        domain, unit_x = self._resolve_forecast_domain(trajectory, horizon, cfg)

        # 3. Extract Paired (x, y) Observation Series respecting Cutoff and EOL baseline metadata
        pairs, latest_obs = self._extract_domain_series(
            trajectory=trajectory,
            field_name=field_name,
            domain=domain,
            unit_x=unit_x,
            cutoff_timestamp_ns=cutoff_ns,
            eol_definition=eol_definition,
        )
        n_pairs = len(pairs)

        # Check Insufficient Data / History
        if n_pairs == 0:
            return self._build_non_valid_result(
                prediction_id=pred_id,
                prediction_timestamp_ns=pred_ts,
                created_at_utc=now_utc,
                system_id=trajectory.system_id,
                status=PrognosticStatus.INSUFFICIENT_DATA,
                rul_domain=domain,
                rul_unit=unit_x,
                eol_definition=eol_definition,
                diagnostic_message=(
                    f"No observation points found containing both quantity '{field_name}' "
                    f"and domain channel '{domain.value}'."
                ),
                observation_count=0,
            )

        if n_pairs < cfg.min_observations:
            return self._build_non_valid_result(
                prediction_id=pred_id,
                prediction_timestamp_ns=pred_ts,
                created_at_utc=now_utc,
                system_id=trajectory.system_id,
                status=PrognosticStatus.INSUFFICIENT_HISTORY,
                rul_domain=domain,
                rul_unit=unit_x,
                eol_definition=eol_definition,
                diagnostic_message=(
                    f"Insufficient observations ({n_pairs} < {cfg.min_observations}) "
                    f"for field '{field_name}' in domain '{domain.value}'."
                ),
                observation_count=n_pairs,
            )

        # Extract initial and latest values within causality boundary
        x_0, y_0 = pairs[0]
        x_latest, y_latest = pairs[-1]
        delta_x = x_latest - x_0
        y_eol = float(eol_definition.threshold_value)

        # Check Non-positive Domain Span
        if delta_x <= 0.0:
            return self._build_non_valid_result(
                prediction_id=pred_id,
                prediction_timestamp_ns=pred_ts,
                created_at_utc=now_utc,
                system_id=trajectory.system_id,
                status=PrognosticStatus.DATA_QUALITY_FAILED,
                rul_domain=domain,
                rul_unit=unit_x,
                eol_definition=eol_definition,
                diagnostic_message=f"Observation domain span is non-positive (delta_x = {delta_x}).",
                observation_count=n_pairs,
            )

        # 4. Check If System Is Already At or Beyond Configured EOL
        is_beyond_eol = False
        if is_decreasing and y_latest <= y_eol:
            is_beyond_eol = True
        elif not is_decreasing and y_latest >= y_eol:
            is_beyond_eol = True

        op_ctx = latest_obs.operating_context if latest_obs else None

        if is_beyond_eol:
            meta = {
                "last_observation_domain_value": round(x_latest, 6),
                "predicted_eol_domain_value": round(x_latest, 6),
                "extrapolation_distance": 0.0,
                "observed_history_span": round(delta_x, 6),
                "extrapolation_ratio": 0.0,
                "source_observation_count": n_pairs,
                "condition": "ALREADY_AT_OR_BEYOND_EOL",
                "prediction_cutoff_timestamp_ns": cutoff_ns,
                "operating_assumptions": "System state currently satisfies or exceeds configured EOL boundary.",
            }
            return RULPredictionResult(
                prediction_id=pred_id,
                prediction_timestamp_ns=pred_ts,
                created_at_utc=now_utc,
                system_id=trajectory.system_id,
                status=PrognosticStatus.VALID,
                rul_value=0.0,
                rul_unit=unit_x,
                rul_domain=domain,
                uncertainty=None,
                eol_definition=eol_definition,
                method_identity=self._method_identity,
                operating_context=op_ctx,
                diagnostic_message=(
                    f"System degradation ({y_latest:.4f} {unit_y}) is already at or beyond "
                    f"the configured End-of-Life threshold ({y_eol:.4f} {unit_y})."
                ),
                reproducibility_metadata=meta,
            )

        # 5. Compute Historical Degradation Slope m = dy / dx
        if cfg.use_ols_slope and n_pairs >= 2:
            m = self._compute_ols_slope(pairs)
            slope_method = "ORDINARY_LEAST_SQUARES"
        else:
            m = (y_latest - y_0) / delta_x
            slope_method = "ENDPOINT_SLOPE"

        if m is None:
            return self._build_non_valid_result(
                prediction_id=pred_id,
                prediction_timestamp_ns=pred_ts,
                created_at_utc=now_utc,
                system_id=trajectory.system_id,
                status=PrognosticStatus.DATA_QUALITY_FAILED,
                rul_domain=domain,
                rul_unit=unit_x,
                eol_definition=eol_definition,
                diagnostic_message="Could not compute mathematical slope across observation points.",
                observation_count=n_pairs,
            )

        # 6. Validate Direction and Slope Against Threshold
        if is_decreasing:
            # Capacity fade: expected negative slope m < 0
            if m >= -cfg.slope_threshold:
                return self._build_non_valid_result(
                    prediction_id=pred_id,
                    prediction_timestamp_ns=pred_ts,
                    created_at_utc=now_utc,
                    system_id=trajectory.system_id,
                    status=PrognosticStatus.OUT_OF_DOMAIN,
                    rul_domain=domain,
                    rul_unit=unit_x,
                    eol_definition=eol_definition,
                    diagnostic_message=(
                        f"Observed rate of change ({m:.6e} {unit_y}/{unit_x}) is non-negative or flat; "
                        f"trajectory does not approach decreasing EOL threshold ({y_eol:.4f} {unit_y})."
                    ),
                    observation_count=n_pairs,
                )
            deg_rate = -m
            delta_y = y_latest - y_eol
            rul = delta_y / deg_rate
        else:
            # Resistance growth: expected positive slope m > 0
            if m <= cfg.slope_threshold:
                return self._build_non_valid_result(
                    prediction_id=pred_id,
                    prediction_timestamp_ns=pred_ts,
                    created_at_utc=now_utc,
                    system_id=trajectory.system_id,
                    status=PrognosticStatus.OUT_OF_DOMAIN,
                    rul_domain=domain,
                    rul_unit=unit_x,
                    eol_definition=eol_definition,
                    diagnostic_message=(
                        f"Observed rate of change ({m:.6e} {unit_y}/{unit_x}) is non-positive or flat; "
                        f"trajectory does not approach increasing EOL threshold ({y_eol:.4f} {unit_y})."
                    ),
                    observation_count=n_pairs,
                )
            deg_rate = m
            delta_y = y_eol - y_latest
            rul = delta_y / deg_rate

        if math.isnan(rul) or math.isinf(rul) or rul < 0.0:
            return self._build_non_valid_result(
                prediction_id=pred_id,
                prediction_timestamp_ns=pred_ts,
                created_at_utc=now_utc,
                system_id=trajectory.system_id,
                status=PrognosticStatus.DATA_QUALITY_FAILED,
                rul_domain=domain,
                rul_unit=unit_x,
                eol_definition=eol_definition,
                diagnostic_message=f"Calculated RUL result is non-finite or negative ({rul}).",
                observation_count=n_pairs,
            )

        # 7. Assemble Extrapolation Metadata
        predicted_eol_point = x_latest + rul
        extrap_ratio = rul / delta_x if delta_x > 0 else 0.0

        meta = {
            "last_observation_domain_value": round(x_latest, 6),
            "predicted_eol_domain_value": round(predicted_eol_point, 6),
            "extrapolation_distance": round(rul, 6),
            "observed_history_span": round(delta_x, 6),
            "extrapolation_ratio": round(extrap_ratio, 4),
            "degradation_rate": round(deg_rate, 8),
            "degradation_rate_unit": f"{unit_y}/{unit_x}",
            "slope_method": slope_method,
            "source_observation_count": n_pairs,
            "prediction_cutoff_timestamp_ns": cutoff_ns,
            "operating_assumptions": (
                "Deterministic first-order linear extrapolation assuming constant degradation rate "
                "under invariant operating profile equivalent to historical trajectory."
            ),
        }

        return RULPredictionResult(
            prediction_id=pred_id,
            prediction_timestamp_ns=pred_ts,
            created_at_utc=now_utc,
            system_id=trajectory.system_id,
            status=PrognosticStatus.VALID,
            rul_value=rul,
            rul_unit=unit_x,
            rul_domain=domain,
            uncertainty=None,  # Deterministic point estimate; uncertainty not modeled in Level 6.3
            eol_definition=eol_definition,
            method_identity=self._method_identity,
            operating_context=op_ctx,
            diagnostic_message="Linear EOL-crossing RUL estimate computed successfully.",
            reproducibility_metadata=meta,
        )

    def _map_eol_criterion(
        self, criterion_type: EOLCriterionType
    ) -> Optional[Tuple[str, bool, str]]:
        """Maps an EOLCriterionType to (quantity_field_name, is_decreasing, unit)."""
        if criterion_type == EOLCriterionType.CAPACITY_RETENTION_THRESHOLD:
            return ("capacity_retention_fraction", True, "fraction")
        elif criterion_type == EOLCriterionType.ABSOLUTE_CAPACITY_THRESHOLD:
            return ("usable_capacity_ah", True, "Ah")
        elif criterion_type == EOLCriterionType.RESISTANCE_GROWTH_RATIO_THRESHOLD:
            return ("resistance_growth_ratio", False, "ratio")
        elif criterion_type == EOLCriterionType.ABSOLUTE_RESISTANCE_THRESHOLD:
            return ("internal_resistance_r0_ohm", False, "ohm")
        return None

    def _resolve_forecast_domain(
        self,
        trajectory: DegradationTrajectory,
        horizon: Optional[PrognosticHorizon],
        config: EstimatorConfig,
    ) -> Tuple[HorizonDomain, str]:
        """Resolves target forecast domain and unit from horizon, config, or trajectory channels."""
        if horizon is not None:
            return (horizon.domain, horizon.unit)
        if config.default_domain is not None:
            d = config.default_domain
            unit_map = {
                HorizonDomain.CYCLE: "cycles",
                HorizonDomain.TIME: "days",
                HorizonDomain.AMPERE_HOUR_THROUGHPUT: "Ah",
                HorizonDomain.WATT_HOUR_THROUGHPUT: "Wh",
                HorizonDomain.CUSTOM: "units",
            }
            return (d, unit_map.get(d, "units"))

        # Default resolution order: EFC -> TIME (days) -> Ah -> Wh
        if len(trajectory.get_series("equivalent_full_cycles")) >= 2:
            return (HorizonDomain.CYCLE, "cycles")
        if trajectory.duration_seconds > 0.0:
            return (HorizonDomain.TIME, "days")
        if len(trajectory.get_series("discharge_throughput_ah")) >= 2:
            return (HorizonDomain.AMPERE_HOUR_THROUGHPUT, "Ah")
        if len(trajectory.get_series("total_energy_throughput_wh")) >= 2:
            return (HorizonDomain.WATT_HOUR_THROUGHPUT, "Wh")

        return (HorizonDomain.CYCLE, "cycles")

    def _extract_domain_series(
        self,
        trajectory: DegradationTrajectory,
        field_name: str,
        domain: HorizonDomain,
        unit_x: str,
        cutoff_timestamp_ns: Optional[int] = None,
        eol_definition: Optional[EOLDefinition] = None,
    ) -> Tuple[list[Tuple[float, float]], Optional[DegradationObservation]]:
        """Extracts aligned (x_domain, y_quantity) coordinate pairs up to the cutoff timestamp.

        For resistance-growth-ratio criteria:
        1. Uses explicit `resistance_growth_ratio` if present on the observation.
        2. Alternatively, if `internal_resistance_r0_ohm` is present and an explicit
           `baseline_resistance_r0_ohm` (or `baseline_r0_ohm`) is provided in EOLDefinition
           metadata or observation metadata with value > 0, computes $R_0 / R_{0,base}$.
        3. Otherwise returns None (never assumes an arbitrary first sample as baseline).
        """
        pairs: list[Tuple[float, float]] = []
        latest_valid_obs: Optional[DegradationObservation] = None

        # Resolve EOL-level baseline resistance if configured
        eol_baseline_r0: Optional[float] = None
        if eol_definition is not None and eol_definition.metadata:
            b_val = eol_definition.metadata.get("baseline_resistance_r0_ohm") or eol_definition.metadata.get("baseline_r0_ohm")
            if isinstance(b_val, (int, float)) and b_val > 0.0:
                eol_baseline_r0 = float(b_val)

        for obs in trajectory.observations:
            # Enforce temporal causality cutoff
            if cutoff_timestamp_ns is not None and obs.timestamp_ns > cutoff_timestamp_ns:
                continue

            # Extract y value with deterministic provenance handling
            y_val: Optional[float] = None
            if field_name == "resistance_growth_ratio":
                r_ratio = obs.get_value("resistance_growth_ratio")
                if r_ratio is not None:
                    y_val = r_ratio
                else:
                    # Check for explicit baseline resistance
                    r0 = obs.get_value("internal_resistance_r0_ohm")
                    r0_base = eol_baseline_r0
                    if r0_base is None and obs.metadata:
                        obs_b_val = obs.metadata.get("baseline_resistance_r0_ohm") or obs.metadata.get("baseline_r0_ohm")
                        if isinstance(obs_b_val, (int, float)) and obs_b_val > 0.0:
                            r0_base = float(obs_b_val)

                    if r0 is not None and r0_base is not None and r0_base > 0.0:
                        y_val = r0 / r0_base
            else:
                y_val = obs.get_value(field_name)

            if y_val is None:
                continue

            # Extract x coordinate according to target domain
            x_val: Optional[float] = None
            if domain == HorizonDomain.CYCLE:
                x_val = obs.get_value("equivalent_full_cycles")
            elif domain == HorizonDomain.TIME:
                u = unit_x.lower()
                if u in ("seconds", "s", "sec"):
                    x_val = obs.timestamp_ns / 1_000_000_000.0
                elif u in ("hours", "h", "hr"):
                    x_val = obs.timestamp_ns / (3600.0 * 1_000_000_000.0)
                elif u in ("weeks", "w", "week"):
                    x_val = obs.timestamp_ns / (7.0 * 86400.0 * 1_000_000_000.0)
                elif u in ("years", "y", "year"):
                    x_val = obs.timestamp_ns / (365.25 * 86400.0 * 1_000_000_000.0)
                else:  # Default days
                    x_val = obs.timestamp_ns / (86400.0 * 1_000_000_000.0)
            elif domain == HorizonDomain.AMPERE_HOUR_THROUGHPUT:
                x_val = obs.get_value("discharge_throughput_ah")
                if x_val is None:
                    x_val = obs.get_value("charge_throughput_ah")
            elif domain == HorizonDomain.WATT_HOUR_THROUGHPUT:
                x_val = obs.get_value("total_energy_throughput_wh")

            if x_val is not None:
                pairs.append((float(x_val), float(y_val)))
                latest_valid_obs = obs

        return pairs, latest_valid_obs

    @staticmethod
    def _compute_ols_slope(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
        """Calculates Ordinary Least Squares (OLS) descriptive historical trend slope.

        Note: OLS is purely a descriptive historical trend estimator for the transparent linear
        baseline. No empirical RUL model or machine-learning RUL model was implemented.
        """
        n = len(pairs)
        if n < 2:
            return None

        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]

        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))

        if ss_xx <= 0.0:
            return None

        return ss_xy / ss_xx

    def _build_non_valid_result(
        self,
        prediction_id: str,
        prediction_timestamp_ns: int,
        created_at_utc: str,
        system_id: str,
        status: PrognosticStatus,
        rul_domain: HorizonDomain,
        rul_unit: str,
        eol_definition: EOLDefinition,
        diagnostic_message: str,
        observation_count: int = 0,
    ) -> RULPredictionResult:
        """Helper building an immutable non-valid RULPredictionResult with rul_value=None."""
        meta = {
            "source_observation_count": observation_count,
            "failure_reason": status.value,
        }
        return RULPredictionResult(
            prediction_id=prediction_id,
            prediction_timestamp_ns=prediction_timestamp_ns,
            created_at_utc=created_at_utc,
            system_id=system_id,
            status=status,
            rul_value=None,  # Strictly None for non-valid statuses
            rul_unit=rul_unit,
            rul_domain=rul_domain,
            uncertainty=None,
            eol_definition=eol_definition,
            method_identity=self._method_identity,
            operating_context=None,
            diagnostic_message=diagnostic_message,
            reproducibility_metadata=meta,
        )

