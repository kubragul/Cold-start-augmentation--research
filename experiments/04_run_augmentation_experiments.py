"""Step 04: run statistical augmentation forecasting experiments."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.statistical_augmentation import generate_statistical_synthetic_series
from src.data.load_finance_data import load_config
from src.evaluation.metrics import evaluate_forecast
from src.models.forecasting_models import (
    linear_trend_forecast,
    moving_average_forecast,
    naive_forecast,
)

ForecastFunction = Callable[[list[float], int], list[float]]
AUGMENTATION_RATIOS = (0.5, 1.0, 2.0)


def configure_logging() -> None:
    """Configure clear console logging for reproducible experiment runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def model_registry() -> dict[str, ForecastFunction]:
    """Return forecasting models used in both baseline and augmentation runs."""
    return {
        "naive": naive_forecast,
        "moving_average": moving_average_forecast,
        "linear_trend": linear_trend_forecast,
    }


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    config = load_config(PROJECT_ROOT / "config.yaml")
    metadata_path = PROJECT_ROOT / "results" / "tables" / "cold_start_scenarios.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"cold-start scenario metadata not found at {metadata_path}; "
            "run experiments/02_create_cold_start_scenarios.py first"
        )

    output_dir = PROJECT_ROOT / config["outputs"]["tables"]
    output_dir.mkdir(parents=True, exist_ok=True)
    augmentation_results_path = output_dir / "statistical_augmentation_results.csv"
    augmentation_predictions_path = output_dir / "statistical_augmentation_predictions.csv"

    scenarios = pd.read_csv(metadata_path)
    available_models = model_registry()
    configured_models = config["forecasting_models"]
    unknown_models = sorted(set(configured_models) - set(available_models))
    if unknown_models:
        raise ValueError(f"unknown forecasting models in config: {unknown_models}")

    random_seed = int(config["random_seed"])
    logger.info("Loaded %d cold-start samples from %s", len(scenarios), metadata_path)
    logger.info("Running statistical augmentation ratios: %s", AUGMENTATION_RATIOS)
    logger.info("Running forecasting models: %s", ", ".join(configured_models))

    result_rows = []
    prediction_rows = []
    processed_sample_count = 0
    failure_count = 0

    for scenario_index, scenario in enumerate(scenarios.to_dict(orient="records")):
        sample_id = scenario["sample_id"]
        try:
            # Metadata paths are stored relative to the project root; joining an
            # already-absolute path is a no-op, so older metadata still loads.
            train = pd.read_csv(PROJECT_ROOT / scenario["train_path"])
            test = pd.read_csv(PROJECT_ROOT / scenario["test_path"])
            train_series = train["y"].tolist()
            y_true = test["y"].tolist()
            forecast_horizon = len(y_true)
        except Exception as exc:
            failure_count += 1
            logger.exception("Failed to load sample %s: %s", sample_id, exc)
            continue

        processed_sample_count += 1

        for ratio_index, augmentation_ratio in enumerate(AUGMENTATION_RATIOS):
            n_synthetic_points = int(round(len(train_series) * augmentation_ratio))
            try:
                # Academic leakage guard: synthetic data are generated from the
                # training series only. The unchanged test series is used later
                # only for forecast evaluation.
                synthetic_values = generate_statistical_synthetic_series(
                    train_series=train_series,
                    n_synthetic_points=n_synthetic_points,
                    random_seed=random_seed + scenario_index * 100 + ratio_index,
                )
                augmented_train_series = train_series + synthetic_values
            except Exception as exc:
                failure_count += 1
                logger.exception(
                    "Sample %s failed during augmentation ratio %.1f: %s",
                    sample_id,
                    augmentation_ratio,
                    exc,
                )
                continue

            for model_name in configured_models:
                try:
                    y_pred = available_models[model_name](augmented_train_series, forecast_horizon)
                    if len(y_pred) != forecast_horizon:
                        raise ValueError(
                            f"model returned {len(y_pred)} predictions for horizon {forecast_horizon}"
                        )

                    metrics = evaluate_forecast(y_true, y_pred)
                    result_rows.append(
                        {
                            "sample_id": sample_id,
                            "ticker": scenario["ticker"],
                            "sector": scenario["sector"],
                            "cold_start_weeks": int(scenario["cold_start_weeks"]),
                            "model": model_name,
                            "augmentation_method": "statistical",
                            "augmentation_ratio": augmentation_ratio,
                            "n_synthetic_points": n_synthetic_points,
                            "MAE": metrics["MAE"],
                            "RMSE": metrics["RMSE"],
                            "MAPE": metrics["MAPE"],
                        }
                    )

                    for date, actual, predicted in zip(test["date"], y_true, y_pred):
                        prediction_rows.append(
                            {
                                "sample_id": sample_id,
                                "ticker": scenario["ticker"],
                                "sector": scenario["sector"],
                                "cold_start_weeks": int(scenario["cold_start_weeks"]),
                                "date": date,
                                "y_true": actual,
                                "y_pred": predicted,
                                "model": model_name,
                                "augmentation_method": "statistical",
                                "augmentation_ratio": augmentation_ratio,
                                "n_synthetic_points": n_synthetic_points,
                            }
                        )
                except Exception as exc:
                    failure_count += 1
                    logger.exception(
                        "Sample %s failed for ratio %.1f and model %s: %s",
                        sample_id,
                        augmentation_ratio,
                        model_name,
                        exc,
                    )

    result_columns = [
        "sample_id",
        "ticker",
        "sector",
        "cold_start_weeks",
        "model",
        "augmentation_method",
        "augmentation_ratio",
        "n_synthetic_points",
        "MAE",
        "RMSE",
        "MAPE",
    ]
    prediction_columns = [
        "sample_id",
        "ticker",
        "sector",
        "cold_start_weeks",
        "date",
        "y_true",
        "y_pred",
        "model",
        "augmentation_method",
        "augmentation_ratio",
        "n_synthetic_points",
    ]
    pd.DataFrame(result_rows, columns=result_columns).to_csv(
        augmentation_results_path,
        index=False,
    )
    pd.DataFrame(prediction_rows, columns=prediction_columns).to_csv(
        augmentation_predictions_path,
        index=False,
    )

    logger.info("Processed %d samples", processed_sample_count)
    logger.info(
        "Wrote %d sample-ratio-model metric rows to %s",
        len(result_rows),
        augmentation_results_path,
    )
    logger.info(
        "Wrote %d prediction rows to %s",
        len(prediction_rows),
        augmentation_predictions_path,
    )
    if failure_count:
        logger.warning("Completed with %d logged failures", failure_count)
    else:
        logger.info("Completed with no sample/augmentation/model failures")


if __name__ == "__main__":
    main()
