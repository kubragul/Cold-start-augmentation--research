"""Fail-fast validation of forecast-origin alignment invariants."""

from src.augmentation.statistical_augmentation import (
    augmented_ensemble_forecast,
    generate_endpoint_preserving_histories,
)
from src.models.forecasting_models import naive_forecast


def main() -> None:
    observed = [100.0, 102.0, 101.0, 105.0, 104.0]
    histories = generate_endpoint_preserving_histories(observed, 20, random_seed=42)
    assert all(len(history) == len(observed) for history in histories)
    assert all(history[0] == observed[0] for history in histories)
    assert all(history[-1] == observed[-1] for history in histories)

    forecast = augmented_ensemble_forecast(observed, 28, naive_forecast, 20, 42)
    assert forecast == [observed[-1]] * 28
    print("Alignment checks passed: synthetic histories preserve dates and endpoints.")


if __name__ == "__main__":
    main()
