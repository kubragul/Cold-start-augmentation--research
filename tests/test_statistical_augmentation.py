"""Tests for endpoint-preserving time-series augmentation."""

import unittest

from src.augmentation.statistical_augmentation import (
    augmented_ensemble_forecast,
    generate_endpoint_preserving_histories,
)
from src.models.forecasting_models import linear_trend_forecast, naive_forecast


class EndpointPreservingAugmentationTests(unittest.TestCase):
    def test_histories_keep_length_and_endpoints(self) -> None:
        observed = [10.0, 12.0, 11.0, 15.0, 14.0]
        histories = generate_endpoint_preserving_histories(observed, 7, random_seed=5)

        self.assertEqual(len(histories), 7)
        for history in histories:
            self.assertEqual(len(history), len(observed))
            self.assertAlmostEqual(history[0], observed[0])
            self.assertAlmostEqual(history[-1], observed[-1])

    def test_generation_is_reproducible(self) -> None:
        observed = [10.0, 12.0, 11.0, 15.0, 14.0]
        first = generate_endpoint_preserving_histories(observed, 3, random_seed=42)
        second = generate_endpoint_preserving_histories(observed, 3, random_seed=42)
        self.assertEqual(first, second)

    def test_naive_forecast_cannot_shift_past_real_endpoint(self) -> None:
        observed = [10.0, 12.0, 11.0, 15.0, 14.0]
        forecast = augmented_ensemble_forecast(observed, 4, naive_forecast, 10, 42)
        self.assertEqual(forecast, [14.0] * 4)

    def test_ensemble_returns_requested_horizon(self) -> None:
        observed = [10.0, 12.0, 11.0, 15.0, 14.0]
        forecast = augmented_ensemble_forecast(observed, 6, linear_trend_forecast, 4, 42)
        self.assertEqual(len(forecast), 6)


if __name__ == "__main__":
    unittest.main()
