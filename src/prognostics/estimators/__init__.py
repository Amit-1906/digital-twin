"""Prognostic and Remaining Useful Life (RUL) Estimators Subsystem.

Provides transparent, deterministic mathematical estimator baselines for predicting
Remaining Useful Life (RUL) against configured End-of-Life (EOL) degradation criteria.
"""

from src.prognostics.estimators.base import (
    AbstractRULEstimator,
    EstimatorConfig,
)
from src.prognostics.estimators.linear import HistoricalLinearRULEstimator

__all__ = [
    "AbstractRULEstimator",
    "EstimatorConfig",
    "HistoricalLinearRULEstimator",
]
