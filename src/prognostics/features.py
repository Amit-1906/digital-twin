"""Descriptive Degradation Feature Extraction and Summary Containers.

Extracts immutable, descriptive statistical features, degradation rates, rolling trends,
and operating context statistics from historical degradation trajectories without
implementing predictive modeling or universal degradation laws.
"""

from dataclasses import dataclass, field
import datetime
import math
from typing import Any, Mapping, Optional, Sequence, Tuple
import uuid

from src.prognostics.provenance import (
    PrognosticQuantity,
    PrognosticValueProvenance,
)
from src.prognostics.trajectory import (
    DegradationObservation,
    DegradationTrajectory,
)
from src.prognostics.trajectory_types import (
    FeatureQualityStatus,
    RateDomain,
    TrajectoryOrderingStatus,
    TrendDirection,
)
from src.prognostics.types import ValueOriginType


@dataclass(frozen=True)
class PrognosticFeature:
    """Strongly-typed individual descriptive prognostic feature container."""

    feature_name: str
    value: Optional[float]
    unit: str
    quality_status: FeatureQualityStatus = FeatureQualityStatus.VALID
    rate_domain: RateDomain = RateDomain.NONE
    calculation_method: str = "DIRECT_STATISTICAL"
    calculation_version: str = "1.0.0"
    source_quantity: str = ""
    source_observation_count: int = 0
    diagnostic_message: Optional[str] = None
    provenance: Optional[PrognosticValueProvenance] = None

    def __post_init__(self) -> None:
        if not self.feature_name or not isinstance(self.feature_name, str):
            raise ValueError("feature_name must be a non-empty string.")
        if not isinstance(self.quality_status, FeatureQualityStatus):
            raise TypeError(f"Expected FeatureQualityStatus, got {type(self.quality_status).__name__}.")
        if not isinstance(self.rate_domain, RateDomain):
            raise TypeError(f"Expected RateDomain, got {type(self.rate_domain).__name__}.")

        if self.quality_status == FeatureQualityStatus.VALID:
            if self.value is None:
                raise ValueError("value cannot be None when quality_status is VALID.")
            if not isinstance(self.value, (int, float)) or math.isnan(self.value) or math.isinf(self.value):
                raise ValueError(f"value must be a finite float when VALID, got {self.value}.")
        else:
            if self.value is not None:
                if math.isnan(self.value) or math.isinf(self.value):
                    raise ValueError("Non-finite values (NaN/Inf) are not permitted.")

        if self.provenance is not None and not isinstance(self.provenance, PrognosticValueProvenance):
            raise TypeError(f"Expected PrognosticValueProvenance, got {type(self.provenance).__name__}.")

    @property
    def is_valid(self) -> bool:
        """Returns True if the feature value is present, finite, and marked VALID."""
        return self.quality_status == FeatureQualityStatus.VALID and self.value is not None

    def to_dict(self) -> dict[str, Any]:
        """Serializes prognostic feature to dictionary."""
        return {
            "feature_name": self.feature_name,
            "value": round(float(self.value), 8) if self.value is not None else None,
            "unit": self.unit,
            "quality_status": self.quality_status.value,
            "is_valid": self.is_valid,
            "rate_domain": self.rate_domain.value,
            "calculation_method": self.calculation_method,
            "calculation_version": self.calculation_version,
            "source_quantity": self.source_quantity,
            "source_observation_count": self.source_observation_count,
            "diagnostic_message": self.diagnostic_message,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass(frozen=True)
class DegradationFeatureSet:
    """Immutable bundle of descriptive prognostic features extracted from a DegradationTrajectory."""

    feature_set_id: str
    trajectory_id: str
    system_id: str
    extracted_at_utc: str
    features: Mapping[str, PrognosticFeature] = field(default_factory=dict)
    summary_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_set_id or not isinstance(self.feature_set_id, str):
            raise ValueError("feature_set_id must be a non-empty string.")
        if not self.trajectory_id or not isinstance(self.trajectory_id, str):
            raise ValueError("trajectory_id must be a non-empty string.")
        if not self.system_id or not isinstance(self.system_id, str):
            raise ValueError("system_id must be a non-empty string.")
        if not self.extracted_at_utc or not isinstance(self.extracted_at_utc, str):
            raise ValueError("extracted_at_utc must be a non-empty string.")

        for name, feat in self.features.items():
            if not isinstance(feat, PrognosticFeature):
                raise TypeError(f"Expected PrognosticFeature for '{name}', got {type(feat).__name__}.")

    def get_feature(self, name: str) -> Optional[PrognosticFeature]:
        """Retrieves a named feature container if present in the feature set."""
        return self.features.get(name)

    def get_value(self, name: str) -> Optional[float]:
        """Retrieves scalar value of a named feature if present and VALID, else None."""
        f = self.get_feature(name)
        if f is not None and f.is_valid:
            return f.value
        return None

    def is_feature_valid(self, name: str) -> bool:
        """Returns True if the feature exists and is marked VALID."""
        f = self.get_feature(name)
        return f is not None and f.is_valid

    def valid_features(self) -> list[str]:
        """Returns list of feature names that are currently present and marked VALID."""
        return [k for k, v in self.features.items() if v.is_valid]

    def to_dict(self) -> dict[str, Any]:
        """Serializes degradation feature set to dictionary."""
        return {
            "feature_set_id": self.feature_set_id,
            "trajectory_id": self.trajectory_id,
            "system_id": self.system_id,
            "extracted_at_utc": self.extracted_at_utc,
            "valid_feature_count": len(self.valid_features()),
            "total_feature_count": len(self.features),
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "summary_metadata": dict(self.summary_metadata),
        }


class TrajectoryFeatureExtractor:
    """Stateless, deterministic extractor of descriptive degradation and operating features."""

    def __init__(
        self,
        min_observations: int = 2,
        rolling_window_size: int = 5,
        slope_noise_threshold: float = 1e-7,
    ) -> None:
        if min_observations < 1:
            raise ValueError("min_observations must be >= 1.")
        if rolling_window_size < 2:
            raise ValueError("rolling_window_size must be >= 2.")

        self._min_obs = min_observations
        self._rolling_window = rolling_window_size
        self._slope_thresh = slope_noise_threshold

    def extract_features(
        self,
        trajectory: DegradationTrajectory,
        feature_set_id: Optional[str] = None,
    ) -> DegradationFeatureSet:
        """Extracts all computable descriptive degradation features from an immutable trajectory.

        Args:
            trajectory: Input DegradationTrajectory.
            feature_set_id: Optional unique identifier for the extracted feature set.

        Returns:
            Immutable DegradationFeatureSet container.
        """
        if not isinstance(trajectory, DegradationTrajectory):
            raise TypeError(f"Expected DegradationTrajectory, got {type(trajectory).__name__}.")

        fid = feature_set_id or f"feat_{trajectory.trajectory_id}_{uuid.uuid4().hex[:8]}"
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        features: dict[str, PrognosticFeature] = {}

        # 1. Trajectory Structure & Integrity Features
        n_total = trajectory.total_observations
        duration_s = trajectory.duration_seconds
        duration_days = duration_s / 86400.0 if duration_s > 0 else 0.0

        features["total_observations"] = PrognosticFeature(
            feature_name="total_observations",
            value=float(n_total),
            unit="count",
            quality_status=FeatureQualityStatus.VALID,
            calculation_method="COUNT",
            source_quantity="observations",
            source_observation_count=n_total,
        )

        features["elapsed_time_seconds"] = PrognosticFeature(
            feature_name="elapsed_time_seconds",
            value=duration_s,
            unit="seconds",
            quality_status=FeatureQualityStatus.VALID,
            calculation_method="ENDPOINT_DELTA",
            source_quantity="timestamp_ns",
            source_observation_count=n_total,
        )

        features["elapsed_time_days"] = PrognosticFeature(
            feature_name="elapsed_time_days",
            value=duration_days,
            unit="days",
            quality_status=FeatureQualityStatus.VALID,
            calculation_method="ENDPOINT_DELTA",
            source_quantity="timestamp_ns",
            source_observation_count=n_total,
        )

        # 2. Extract Capacity Trajectory Features
        cap_series = trajectory.get_series("usable_capacity_ah")
        self._extract_quantity_features(
            features=features,
            series=cap_series,
            quantity_name="usable_capacity_ah",
            unit="Ah",
            is_fade=True,
            trajectory=trajectory,
        )

        # 3. Extract Capacity Retention (SOH_C) Features
        soh_c_series = trajectory.get_series("capacity_retention_fraction")
        self._extract_quantity_features(
            features=features,
            series=soh_c_series,
            quantity_name="capacity_retention_fraction",
            unit="fraction",
            is_fade=True,
            trajectory=trajectory,
        )

        # 4. Extract Internal Resistance (R0) Features
        res_series = trajectory.get_series("internal_resistance_r0_ohm")
        self._extract_quantity_features(
            features=features,
            series=res_series,
            quantity_name="internal_resistance_r0_ohm",
            unit="ohm",
            is_fade=False,
            trajectory=trajectory,
        )

        # 5. Extract Resistance Growth Ratio (SOH_R) Features
        res_ratio_series = trajectory.get_series("resistance_growth_ratio")
        self._extract_quantity_features(
            features=features,
            series=res_ratio_series,
            quantity_name="resistance_growth_ratio",
            unit="ratio",
            is_fade=False,
            trajectory=trajectory,
        )

        # 6. Extract Operating Context Statistics
        self._extract_operating_context_features(features, trajectory)

        # Summary metadata
        summary_meta = {
            "ordering_status": trajectory.ordering_status.value,
            "system_id": trajectory.system_id,
            "trajectory_quantity_type": trajectory.quantity_type.value,
        }

        return DegradationFeatureSet(
            feature_set_id=fid,
            trajectory_id=trajectory.trajectory_id,
            system_id=trajectory.system_id,
            extracted_at_utc=now_utc,
            features=features,
            summary_metadata=summary_meta,
        )

    def _extract_quantity_features(
        self,
        features: dict[str, PrognosticFeature],
        series: list[Tuple[int, float]],
        quantity_name: str,
        unit: str,
        is_fade: bool,
        trajectory: DegradationTrajectory,
    ) -> None:
        """Helper computing delta, rate, and trend features for a single continuous degradation quantity."""
        n = len(series)
        prefix = quantity_name

        if n < self._min_obs:
            status = FeatureQualityStatus.UNAVAILABLE if n == 0 else FeatureQualityStatus.INSUFFICIENT_DATA
            features[f"{prefix}_delta"] = PrognosticFeature(
                feature_name=f"{prefix}_delta",
                value=None,
                unit=unit,
                quality_status=status,
                source_quantity=quantity_name,
                source_observation_count=n,
                diagnostic_message=f"Need at least {self._min_obs} valid observations, found {n}.",
            )
            return

        t_0, val_0 = series[0]
        t_end, val_end = series[-1]
        dt_s = (t_end - t_0) / 1_000_000_000.0
        dt_days = dt_s / 86400.0

        # Endpoint Delta (For fade: Q0 - Qend; For growth: Rend - R0)
        delta = (val_0 - val_end) if is_fade else (val_end - val_0)
        features[f"{prefix}_delta"] = PrognosticFeature(
            feature_name=f"{prefix}_delta",
            value=delta,
            unit=unit,
            quality_status=FeatureQualityStatus.VALID,
            calculation_method="ENDPOINT_DELTA",
            source_quantity=quantity_name,
            source_observation_count=n,
        )

        features[f"{prefix}_initial_value"] = PrognosticFeature(
            feature_name=f"{prefix}_initial_value",
            value=val_0,
            unit=unit,
            quality_status=FeatureQualityStatus.VALID,
            calculation_method="INITIAL_VALUE",
            source_quantity=quantity_name,
            source_observation_count=n,
        )

        features[f"{prefix}_latest_value"] = PrognosticFeature(
            feature_name=f"{prefix}_latest_value",
            value=val_end,
            unit=unit,
            quality_status=FeatureQualityStatus.VALID,
            calculation_method="LATEST_VALUE",
            source_quantity=quantity_name,
            source_observation_count=n,
        )

        # Rate per Time (Second and Day)
        if dt_s > 0.0:
            features[f"{prefix}_rate_per_second"] = PrognosticFeature(
                feature_name=f"{prefix}_rate_per_second",
                value=delta / dt_s,
                unit=f"{unit}/s",
                quality_status=FeatureQualityStatus.VALID,
                rate_domain=RateDomain.PER_SECOND,
                calculation_method="ENDPOINT_RATE",
                source_quantity=quantity_name,
                source_observation_count=n,
            )
            features[f"{prefix}_rate_per_day"] = PrognosticFeature(
                feature_name=f"{prefix}_rate_per_day",
                value=delta / dt_days if dt_days > 0.0 else 0.0,
                unit=f"{unit}/day",
                quality_status=FeatureQualityStatus.VALID,
                rate_domain=RateDomain.PER_DAY,
                calculation_method="ENDPOINT_RATE",
                source_quantity=quantity_name,
                source_observation_count=n,
            )
        else:
            features[f"{prefix}_rate_per_second"] = PrognosticFeature(
                feature_name=f"{prefix}_rate_per_second",
                value=None,
                unit=f"{unit}/s",
                quality_status=FeatureQualityStatus.ZERO_DENOMINATOR,
                rate_domain=RateDomain.PER_SECOND,
                source_quantity=quantity_name,
                source_observation_count=n,
                diagnostic_message="Elapsed time span is zero.",
            )

        # Rate per EFC (if EFC series available)
        efc_series = trajectory.get_series("equivalent_full_cycles")
        if len(efc_series) >= 2:
            efc_0 = efc_series[0][1]
            efc_end = efc_series[-1][1]
            d_efc = efc_end - efc_0
            if d_efc > 0.0:
                features[f"{prefix}_rate_per_efc"] = PrognosticFeature(
                    feature_name=f"{prefix}_rate_per_efc",
                    value=delta / d_efc,
                    unit=f"{unit}/EFC",
                    quality_status=FeatureQualityStatus.VALID,
                    rate_domain=RateDomain.PER_EFC,
                    calculation_method="ENDPOINT_RATE",
                    source_quantity=quantity_name,
                    source_observation_count=n,
                )
            else:
                features[f"{prefix}_rate_per_efc"] = PrognosticFeature(
                    feature_name=f"{prefix}_rate_per_efc",
                    value=None,
                    unit=f"{unit}/EFC",
                    quality_status=FeatureQualityStatus.ZERO_DENOMINATOR,
                    rate_domain=RateDomain.PER_EFC,
                    source_quantity=quantity_name,
                    source_observation_count=n,
                    diagnostic_message="EFC delta is zero or non-positive.",
                )

        # Rate per Ah Throughput (if Ah series available)
        ah_series = trajectory.get_series("discharge_throughput_ah") or trajectory.get_series("charge_throughput_ah")
        if len(ah_series) >= 2:
            ah_0 = ah_series[0][1]
            ah_end = ah_series[-1][1]
            d_ah = ah_end - ah_0
            if d_ah > 0.0:
                features[f"{prefix}_rate_per_ah"] = PrognosticFeature(
                    feature_name=f"{prefix}_rate_per_ah",
                    value=delta / d_ah,
                    unit=f"{unit}/Ah",
                    quality_status=FeatureQualityStatus.VALID,
                    rate_domain=RateDomain.PER_AMPERE_HOUR,
                    calculation_method="ENDPOINT_RATE",
                    source_quantity=quantity_name,
                    source_observation_count=n,
                )
            else:
                features[f"{prefix}_rate_per_ah"] = PrognosticFeature(
                    feature_name=f"{prefix}_rate_per_ah",
                    value=None,
                    unit=f"{unit}/Ah",
                    quality_status=FeatureQualityStatus.ZERO_DENOMINATOR,
                    rate_domain=RateDomain.PER_AMPERE_HOUR,
                    source_quantity=quantity_name,
                    source_observation_count=n,
                    diagnostic_message="Throughput delta is zero or non-positive.",
                )

        # Linear Regression (OLS) Trend Metrics (Descriptive summary only)
        if n >= 2 and dt_days > 0.0:
            slope, intercept, r_sq = self._compute_ols_trend(series)
            if slope is not None and r_sq is not None:
                features[f"{prefix}_ols_slope_per_day"] = PrognosticFeature(
                    feature_name=f"{prefix}_ols_slope_per_day",
                    value=slope * 86400.0 * 1_000_000_000.0,  # Convert slope per ns to slope per day
                    unit=f"{unit}/day",
                    quality_status=FeatureQualityStatus.VALID,
                    rate_domain=RateDomain.PER_DAY,
                    calculation_method="ORDINARY_LEAST_SQUARES",
                    source_quantity=quantity_name,
                    source_observation_count=n,
                )
                features[f"{prefix}_ols_r_squared"] = PrognosticFeature(
                    feature_name=f"{prefix}_ols_r_squared",
                    value=r_sq,
                    unit="ratio",
                    quality_status=FeatureQualityStatus.VALID,
                    calculation_method="ORDINARY_LEAST_SQUARES",
                    source_quantity=quantity_name,
                    source_observation_count=n,
                )

        # Rolling Slope Dynamics (if enough observations)
        if n >= self._rolling_window:
            recent_subseries = series[-self._rolling_window:]
            slope_rec, _, _ = self._compute_ols_trend(recent_subseries)
            if slope_rec is not None:
                slope_rec_day = slope_rec * 86400.0 * 1_000_000_000.0
                features[f"{prefix}_rolling_slope_recent_per_day"] = PrognosticFeature(
                    feature_name=f"{prefix}_rolling_slope_recent_per_day",
                    value=slope_rec_day,
                    unit=f"{unit}/day",
                    quality_status=FeatureQualityStatus.VALID,
                    rate_domain=RateDomain.PER_DAY,
                    calculation_method="ROLLING_WINDOW_OLS",
                    source_quantity=quantity_name,
                    source_observation_count=self._rolling_window,
                )

                # Rolling Slope Dispersion / Variability
                rolling_slopes = []
                for idx in range(n - self._rolling_window + 1):
                    window = series[idx : idx + self._rolling_window]
                    sl, _, _ = self._compute_ols_trend(window)
                    if sl is not None:
                        rolling_slopes.append(sl * 86400.0 * 1_000_000_000.0)

                if len(rolling_slopes) >= 2:
                    mean_sl = sum(rolling_slopes) / len(rolling_slopes)
                    var_sl = sum((s - mean_sl) ** 2 for s in rolling_slopes) / (len(rolling_slopes) - 1)
                    std_sl = math.sqrt(max(0.0, var_sl))
                    features[f"{prefix}_rolling_slope_std_dev"] = PrognosticFeature(
                        feature_name=f"{prefix}_rolling_slope_std_dev",
                        value=std_sl,
                        unit=f"{unit}/day",
                        quality_status=FeatureQualityStatus.VALID,
                        calculation_method="SAMPLE_STANDARD_DEVIATION",
                        source_quantity=quantity_name,
                        source_observation_count=len(rolling_slopes),
                    )

    def _extract_operating_context_features(
        self,
        features: dict[str, PrognosticFeature],
        trajectory: DegradationTrajectory,
    ) -> None:
        """Extracts descriptive operating condition metrics (temperatures, current, SOC, voltage)."""
        temp_cell_vals: list[float] = []
        temp_amb_vals: list[float] = []
        current_vals: list[float] = []
        soc_vals: list[float] = []
        voltage_vals: list[float] = []

        for obs in trajectory.observations:
            ctx = obs.operating_context
            if ctx is not None:
                if ctx.cell_temperature_c is not None:
                    temp_cell_vals.append(ctx.cell_temperature_c)
                if ctx.ambient_temperature_c is not None:
                    temp_amb_vals.append(ctx.ambient_temperature_c)
                if ctx.pack_current_a is not None:
                    current_vals.append(ctx.pack_current_a)
                if ctx.soc_fraction is not None:
                    soc_vals.append(ctx.soc_fraction)
                if ctx.pack_voltage_v is not None:
                    voltage_vals.append(ctx.pack_voltage_v)

        # Thermal metrics
        if temp_cell_vals:
            features["mean_cell_temperature_c"] = PrognosticFeature(
                feature_name="mean_cell_temperature_c",
                value=sum(temp_cell_vals) / len(temp_cell_vals),
                unit="degC",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="ARITHMETIC_MEAN",
                source_quantity="cell_temperature_c",
                source_observation_count=len(temp_cell_vals),
            )
            features["max_cell_temperature_c"] = PrognosticFeature(
                feature_name="max_cell_temperature_c",
                value=max(temp_cell_vals),
                unit="degC",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="MAXIMUM",
                source_quantity="cell_temperature_c",
                source_observation_count=len(temp_cell_vals),
            )
            features["min_cell_temperature_c"] = PrognosticFeature(
                feature_name="min_cell_temperature_c",
                value=min(temp_cell_vals),
                unit="degC",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="MINIMUM",
                source_quantity="cell_temperature_c",
                source_observation_count=len(temp_cell_vals),
            )
            features["cell_temperature_range_c"] = PrognosticFeature(
                feature_name="cell_temperature_range_c",
                value=max(temp_cell_vals) - min(temp_cell_vals),
                unit="degC",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="RANGE",
                source_quantity="cell_temperature_c",
                source_observation_count=len(temp_cell_vals),
            )

        if temp_amb_vals:
            features["mean_ambient_temperature_c"] = PrognosticFeature(
                feature_name="mean_ambient_temperature_c",
                value=sum(temp_amb_vals) / len(temp_amb_vals),
                unit="degC",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="ARITHMETIC_MEAN",
                source_quantity="ambient_temperature_c",
                source_observation_count=len(temp_amb_vals),
            )

        # Current & Load metrics
        if current_vals:
            abs_curr = [abs(i) for i in current_vals]
            features["mean_absolute_current_a"] = PrognosticFeature(
                feature_name="mean_absolute_current_a",
                value=sum(abs_curr) / len(abs_curr),
                unit="A",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="MEAN_ABSOLUTE_VALUE",
                source_quantity="pack_current_a",
                source_observation_count=len(current_vals),
            )
            features["max_discharge_current_a"] = PrognosticFeature(
                feature_name="max_discharge_current_a",
                value=max(0.0, max(current_vals)),
                unit="A",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="MAXIMUM",
                source_quantity="pack_current_a",
                source_observation_count=len(current_vals),
            )
            features["max_charge_current_a"] = PrognosticFeature(
                feature_name="max_charge_current_a",
                value=abs(min(0.0, min(current_vals))),
                unit="A",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="PEAK_ABSOLUTE_NEGATIVE",
                source_quantity="pack_current_a",
                source_observation_count=len(current_vals),
            )

        # SOC metrics
        if soc_vals:
            features["mean_soc_fraction"] = PrognosticFeature(
                feature_name="mean_soc_fraction",
                value=sum(soc_vals) / len(soc_vals),
                unit="fraction",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="ARITHMETIC_MEAN",
                source_quantity="soc_fraction",
                source_observation_count=len(soc_vals),
            )
            features["soc_range_fraction"] = PrognosticFeature(
                feature_name="soc_range_fraction",
                value=max(soc_vals) - min(soc_vals),
                unit="fraction",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="RANGE",
                source_quantity="soc_fraction",
                source_observation_count=len(soc_vals),
            )

        # Voltage metrics
        if voltage_vals:
            features["mean_voltage_v"] = PrognosticFeature(
                feature_name="mean_voltage_v",
                value=sum(voltage_vals) / len(voltage_vals),
                unit="V",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="ARITHMETIC_MEAN",
                source_quantity="pack_voltage_v",
                source_observation_count=len(voltage_vals),
            )
            features["voltage_range_v"] = PrognosticFeature(
                feature_name="voltage_range_v",
                value=max(voltage_vals) - min(voltage_vals),
                unit="V",
                quality_status=FeatureQualityStatus.VALID,
                calculation_method="RANGE",
                source_quantity="pack_voltage_v",
                source_observation_count=len(voltage_vals),
            )

    @staticmethod
    def _compute_ols_trend(
        series: Sequence[Tuple[int, float]]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Computes Ordinary Least Squares (OLS) linear slope, intercept, and R^2 for (t_ns, y) pairs."""
        n = len(series)
        if n < 2:
            return None, None, None

        x_vals = [float(p[0]) for p in series]
        y_vals = [float(p[1]) for p in series]

        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)
        ss_yy = sum((y - y_mean) ** 2 for y in y_vals)
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))

        if ss_xx <= 0.0:
            return None, None, None

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

        if ss_yy <= 0.0:
            # Zero variance in y: slope is 0, R^2 is 1.0 (perfect horizontal line)
            r_sq = 1.0
        else:
            r_sq = (ss_xy ** 2) / (ss_xx * ss_yy)
            r_sq = max(0.0, min(1.0, r_sq))

        return slope, intercept, r_sq
