"""Simple baseline forecasting models for cold-start experiments."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite


def naive_forecast(train_data: Sequence[float] | Sequence[dict], forecast_horizon: int) -> list[float]:
    """Forecast by repeating the last observed training value.

    This is the minimum credible baseline for the experiment: it asks whether
    augmentation can outperform a model that assumes the most recent observed
    price level persists. The function uses only the training window and never
    receives test-window values, which prevents look-ahead leakage.
    """
    values = _extract_training_values(train_data)
    _validate_forecast_horizon(forecast_horizon)
    return [values[-1]] * forecast_horizon


def moving_average_forecast(
    train_data: Sequence[float] | Sequence[dict],
    forecast_horizon: int,
    k: int = 5,
) -> list[float]:
    """Forecast by repeating the mean of the last ``k`` training observations.

    This baseline smooths short-term noise while remaining intentionally simple.
    It is useful academically because it tests whether augmentation adds value
    beyond a standard local-average rule computed only from the cold-start
    training window.
    """
    values = _extract_training_values(train_data)
    _validate_forecast_horizon(forecast_horizon)
    _validate_k(k)

    window = values[-k:]
    prediction = sum(window) / len(window)
    return [prediction] * forecast_horizon


def linear_trend_forecast(
    train_data: Sequence[float] | Sequence[dict],
    forecast_horizon: int,
) -> list[float]:
    """Forecast by extrapolating a fitted linear trend over the training window.

    The model fits a one-dimensional least-squares line using the training
    observation index as time. It is a deliberately transparent baseline for
    trend-following behavior in cold-start settings and does not inspect any
    future test observations.
    """
    values = _extract_training_values(train_data)
    _validate_forecast_horizon(forecast_horizon)

    if len(values) == 1:
        return [values[0]] * forecast_horizon

    x_values = list(range(len(values)))
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(values) / len(values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    slope = numerator / denominator if denominator else 0.0
    intercept = y_mean - slope * x_mean

    return [intercept + slope * step for step in range(len(values), len(values) + forecast_horizon)]


def _extract_training_values(train_data: Sequence[float] | Sequence[dict]) -> list[float]:
    """Extract numeric target values from a training-only sequence."""
    if not train_data:
        raise ValueError("train_data must contain at least one observation")

    first_item = train_data[0]
    if isinstance(first_item, dict):
        values = []
        for row in train_data:
            if "y" not in row:
                raise ValueError("each training row must contain a 'y' value")
            values.append(_as_finite_float(row["y"]))
    else:
        values = [_as_finite_float(value) for value in train_data]

    if not values:
        raise ValueError("train_data must contain at least one observation")
    return values


def _validate_forecast_horizon(forecast_horizon: int) -> None:
    if not isinstance(forecast_horizon, int):
        raise TypeError("forecast_horizon must be an integer")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive")


def _validate_k(k: int) -> None:
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")


def _as_finite_float(value: object) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("training values must be numeric") from exc
    if not isfinite(numeric_value):
        raise ValueError("training values must be finite")
    return numeric_value
