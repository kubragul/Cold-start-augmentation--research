"""Step 02: create rolling cold-start forecasting scenarios."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_finance_data import load_config
from src.preprocessing.create_cold_start_scenarios import (
    create_rolling_cold_start_scenarios,
    save_scenario_sample_files,
)


def configure_logging() -> None:
    """Configure clear console logging for reproducible experiment runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    config = load_config(PROJECT_ROOT / "config.yaml")
    input_path = PROJECT_ROOT / "data" / "processed" / "finance_adjusted_close_long.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"cleaned data not found at {input_path}; run experiments/01_download_data.py first"
        )

    logger.info("Reading cleaned long-format data from %s", input_path)
    data = pd.read_csv(input_path)

    metadata, sample_data = create_rolling_cold_start_scenarios(
        data=data,
        cold_start_windows_weeks=config["cold_start_windows_weeks"],
        forecast_horizon_days=int(config["forecast_horizon_days"]),
        rolling_step_days=int(config.get("rolling_step_days", 28)),
    )

    scenarios_dir = PROJECT_ROOT / "data" / "processed" / "cold_start_scenarios"
    metadata = save_scenario_sample_files(
        metadata=metadata,
        sample_data=sample_data,
        output_dir=scenarios_dir,
        path_root=PROJECT_ROOT,
    )

    tables_dir = PROJECT_ROOT / config["outputs"]["tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = tables_dir / "cold_start_scenarios.csv"
    metadata.to_csv(metadata_path, index=False)

    logger.info("Created %d rolling scenario samples", len(metadata))
    logger.info("Saved scenario metadata to %s", metadata_path)
    logger.info("Saved train/test sample CSV files under %s", scenarios_dir)
    logger.info(
        "Scenario counts by cold-start window: %s",
        metadata.groupby("cold_start_weeks").size().to_dict(),
    )
    logger.info(
        "Rolling step between scenario starts: %d trading observations",
        int(config.get("rolling_step_days", 28)),
    )


if __name__ == "__main__":
    main()
