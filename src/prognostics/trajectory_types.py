"""Enumerations and Classifications for Battery Degradation Trajectories and Feature Extraction.

Defines strongly-typed, chemistry-agnostic enumerations for trajectory quantity types,
temporal ordering classifications, degradation rate domains, feature quality states,
and trend directions.
"""

from enum import Enum


class TrajectoryQuantityType(str, Enum):
    """Physical or parametric quantity tracked across a degradation trajectory."""

    CAPACITY = "CAPACITY"                                           # Usable / discharge capacity (Ah)
    CAPACITY_RETENTION = "CAPACITY_RETENTION"                       # SOH_C capacity retention fraction
    INTERNAL_RESISTANCE = "INTERNAL_RESISTANCE"                     # DC internal resistance R0 (Ohm)
    RESISTANCE_GROWTH_RATIO = "RESISTANCE_GROWTH_RATIO"             # R0 / R0_baseline ratio
    EQUIVALENT_FULL_CYCLES = "EQUIVALENT_FULL_CYCLES"               # Cumulative equivalent full cycles (EFC)
    THROUGHPUT_AH = "THROUGHPUT_AH"                                 # Cumulative charge/discharge throughput (Ah)
    THROUGHPUT_WH = "THROUGHPUT_WH"                                 # Cumulative energy throughput (Wh)
    TEMPERATURE = "TEMPERATURE"                                     # Cell/pack operational temperature (degC)
    MULTIVARIATE = "MULTIVARIATE"                                   # Multi-dimensional synchronized degradation snapshot
    CUSTOM = "CUSTOM"                                               # Custom user-defined degradation channel


class TrajectoryOrderingStatus(str, Enum):
    """Temporal ordering integrity and continuity of trajectory observations."""

    STRICTLY_MONOTONIC = "STRICTLY_MONOTONIC"                   # Strictly increasing timestamps (t_{k+1} > t_k)
    MONOTONIC_WITH_DUPLICATES = "MONOTONIC_WITH_DUPLICATES"     # Non-decreasing timestamps with duplicates (t_{k+1} >= t_k)
    RETROGRADE_DETECTED = "RETROGRADE_DETECTED"                 # Timestamp reversal detected (t_{k+1} < t_k)
    EMPTY = "EMPTY"                                             # Trajectory contains zero observations
    SINGLE_POINT = "SINGLE_POINT"                               # Trajectory contains exactly one observation


class RateDomain(str, Enum):
    """Physical or operational denominator domain for degradation rate calculations."""

    PER_SECOND = "PER_SECOND"               # Rate per elapsed wall-clock second (/s)
    PER_HOUR = "PER_HOUR"                   # Rate per operating hour (/h)
    PER_DAY = "PER_DAY"                     # Rate per calendar day (/day)
    PER_EFC = "PER_EFC"                     # Rate per equivalent full cycle (/cycle)
    PER_AMPERE_HOUR = "PER_AMPERE_HOUR"     # Rate per Ampere-hour throughput (/Ah)
    PER_WATT_HOUR = "PER_WATT_HOUR"         # Rate per Watt-hour energy throughput (/Wh)
    NONE = "NONE"                           # Dimensionless or non-rate feature


class FeatureQualityStatus(str, Enum):
    """Validity, completeness, and metrological quality of an extracted prognostic feature."""

    VALID = "VALID"                                 # Feature calculated successfully with sufficient valid samples
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"         # Sample count too small to compute feature
    DATA_QUALITY_FAILED = "DATA_QUALITY_FAILED"     # Input observations failed validation or contain anomalies
    UNAVAILABLE = "UNAVAILABLE"                     # Required quantity was not present in trajectory observations
    ZERO_DENOMINATOR = "ZERO_DENOMINATOR"           # Delta denominator is zero (prevents divide-by-zero)
    UNDEFINED_VARIANCE = "UNDEFINED_VARIANCE"       # Variance or regression slope undefined (zero reference variance)
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"                 # Operating conditions outside extraction domain


class TrendDirection(str, Enum):
    """Descriptive directional classification of an observed degradation slope."""

    DEGRADING = "DEGRADING"         # Negative capacity slope or positive resistance growth slope
    STABLE = "STABLE"               # Slope statistically indistinguishable from zero within noise tolerance
    RECOVERING = "RECOVERING"       # Apparent capacity rebound or resistance drop (e.g. relaxation/temperature effect)
    INDETERMINATE = "INDETERMINATE" # Trend cannot be evaluated due to insufficient data or high noise
