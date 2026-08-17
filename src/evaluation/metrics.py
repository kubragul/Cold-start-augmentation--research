"""Evaluation metrics for cold-start forecasting experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

NEAR_ZERO_TOLERANCE = 1e-8


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return average absolute forecast error."""
    actual, predicted = _validate_metric_inputs(y_true, y_pred)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return root mean squared forecast error."""
    actual, predicted = _validate_metric_inputs(y_true, y_pred)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mean_absolute_percentage_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return mean absolute percentage error as a percentage.

    MAPE is undefined when true values are zero and unstable when true values
    are very close to zero. This implementation raises a clear error rather
    than silently dropping observations or returning misleading percentages.
    """
    actual, predicted = _validate_metric_inputs(y_true, y_pred)
    near_zero_values = [value for value in actual if abs(value) <= NEAR_ZERO_TOLERANCE]
    if near_zero_values:
        raise ValueError("MAPE is undefined for zero or near-zero true values")

    return sum(abs((a - p) / a) for a, p in zip(actual, predicted)) / len(actual) * 100


def evaluate_forecast(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Evaluate a forecast with complementary academic metrics.

    Multiple metrics are reported because each emphasizes a different aspect of
    forecast quality: MAE gives the average absolute error, RMSE penalizes large
    errors more heavily, and MAPE supports scale-independent interpretation
    when true values are safely away from zero.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def _validate_metric_inputs(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Validate metric inputs and return finite numeric lists."""
    actual = _to_finite_numeric_list(y_true, "y_true")
    predicted = _to_finite_numeric_list(y_pred, "y_pred")

    if not actual:
        raise ValueError("y_true and y_pred must contain at least one value")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")

    return actual, predicted


def _to_finite_numeric_list(values: Sequence[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numeric values, not a string")

    try:
        numeric_values = [float(value) for value in values]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable sequence of numeric values") from exc
    except ValueError as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc

    for value in numeric_values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite values")

    return numeric_values
