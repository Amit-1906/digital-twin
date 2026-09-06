"""Enumerations and Metrological Classifications for Battery Prognostics and RUL.

Defines strongly-typed, chemistry-agnostic, and hardware-agnostic enumerations
for prognostic lifecycle status, value origins, quantity availability, End-of-Life (EOL)
criteria, prediction horizons, and uncertainty representations.
"""

from enum import Enum


class PrognosticStatus(str, Enum):
    """Execution and validity status of a prognostic estimation or RUL prediction."""

    VALID = "VALID"                                 # Prediction evaluated successfully with valid inputs
    UNINITIALIZED = "UNINITIALIZED"                 # Model or baseline degradation state has not been initialized
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"         # Required degradation/telemetry channels are missing or empty
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"   # Historical observation window too short for trend extrapolation
    DATA_QUALITY_FAILED = "DATA_QUALITY_FAILED"     # Input telemetry failed physics or integrity validation
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"                 # Operating conditions exceed valid parameterization envelope
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"         # Configured prognostic algorithm or model engine is unavailable


class ValueOriginType(str, Enum):
    """Physical, algorithmic, or mathematical origin of prognostic quantities."""

    MEASURED = "MEASURED"   # Direct physical sensor or laboratory instrumentation measurement
    ESTIMATED = "ESTIMATED" # State observer, Kalman filter, or online parameter tracking output
    MODELED = "MODELED"     # Electrochemical, physics-based, or empirical model simulation
    DERIVED = "DERIVED"     # Mathematical calculation or aggregation (e.g. cumulative sum, integral)
    UNKNOWN = "UNKNOWN"     # Origin source is unrecorded or uncharacterized


class QuantityAvailability(str, Enum):
    """Explicit availability status of optional prognostic input quantities."""

    AVAILABLE = "AVAILABLE"                   # Valid value present and verified
    UNAVAILABLE = "UNAVAILABLE"               # Quantity not provided or unmeasured by hardware
    ESTIMATED_INDIRECT = "ESTIMATED_INDIRECT" # Inferred from secondary observable channels
    OUT_OF_RANGE = "OUT_OF_RANGE"             # Measurement rejected due to physical sensor bounds violation


class EOLCriterionType(str, Enum):
    """Configurable End-of-Life (EOL) degradation threshold classification."""

    CAPACITY_RETENTION_THRESHOLD = "CAPACITY_RETENTION_THRESHOLD"         # Relative SOH capacity retention (e.g. SOH_C <= 0.80)
    ABSOLUTE_CAPACITY_THRESHOLD = "ABSOLUTE_CAPACITY_THRESHOLD"           # Absolute usable capacity (e.g. Q_usable <= 2.0 Ah)
    RESISTANCE_GROWTH_RATIO_THRESHOLD = "RESISTANCE_GROWTH_RATIO_THRESHOLD" # Impedance growth multiplier (e.g. R0 / R0_fresh >= 2.0)
    ABSOLUTE_RESISTANCE_THRESHOLD = "ABSOLUTE_RESISTANCE_THRESHOLD"       # Absolute internal resistance (e.g. R0 >= 0.050 Ohm)
    PEAK_POWER_THRESHOLD = "PEAK_POWER_THRESHOLD"                         # Maximum discharge/charge power capability limit
    APPLICATION_SPECIFIC = "APPLICATION_SPECIFIC"                         # Application-level service boundary (e.g. warranty, 2nd life)
    CUSTOM = "CUSTOM"                                                     # Custom user-defined boundary criterion


class ComparisonOperator(str, Enum):
    """Mathematical relational operator used for End-of-Life threshold gating."""

    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"       # <= (e.g. Capacity retention <= 80%)
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL" # >= (e.g. Resistance ratio >= 2.0x)
    LESS_THAN = "LESS_THAN"                         # <
    GREATER_THAN = "GREATER_THAN"                   # >
    EQUALS = "EQUALS"                               # ==


class HorizonDomain(str, Enum):
    """Physical or operational domain in which Remaining Useful Life (RUL) is projected."""

    TIME = "TIME"                                           # Elapsed wall-clock or operating duration (seconds, hours, days, years)
    CYCLE = "CYCLE"                                         # Charge/discharge cycle counts or equivalent full cycles (EFC)
    AMPERE_HOUR_THROUGHPUT = "AMPERE_HOUR_THROUGHPUT"       # Cumulative charge or discharge capacity throughput (Ah)
    WATT_HOUR_THROUGHPUT = "WATT_HOUR_THROUGHPUT"           # Cumulative electrical energy throughput (Wh / kWh)
    CUSTOM = "CUSTOM"                                       # Domain-specific custom projection metric


class TimeUnit(str, Enum):
    """Standardized time units for time-domain prognostic horizons and RUL."""

    SECONDS = "SECONDS"
    MINUTES = "MINUTES"
    HOURS = "HOURS"
    DAYS = "DAYS"
    WEEKS = "WEEKS"
    MONTHS = "MONTHS"
    YEARS = "YEARS"


class UncertaintyType(str, Enum):
    """Mathematical formulation of prognostic prediction uncertainty."""

    CONFIDENCE_BOUNDS = "CONFIDENCE_BOUNDS"                     # Explicit lower and upper bounds at a specified coverage parameter
    STANDARD_DEVIATION = "STANDARD_DEVIATION"                   # Parametric dispersion standard deviation (sigma)
    PERCENTILE_INTERVAL = "PERCENTILE_INTERVAL"                 # Non-parametric empirical percentile interval (e.g. P10 - P90)
    EXPANDED_UNCERTAINTY = "EXPANDED_UNCERTAINTY"               # GUM Guide expanded uncertainty (k-factor coverage)
    DETERMINISTIC_NO_UNCERTAINTY = "DETERMINISTIC_NO_UNCERTAINTY" # Point estimate only; no uncertainty modeled
    UNDEFINED = "UNDEFINED"                                     # Uncertainty was not evaluated or is undefined
