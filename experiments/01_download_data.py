"""Step 01: download and clean finance data for the academic pilot."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_finance_data import (
    create_clean_long_dataset,
    create_data_quality_report,
    download_adjusted_close,
    load_config,
    save_raw_download,
    ticker_sector_map,
)


def configure_logging() -> None:
    """Configure clear, reproducible console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    config_path = PROJECT_ROOT / "config.yaml"
    config = load_config(config_path)
    sectors = ticker_sector_map(config)
    tickers = list(sectors)

    logger.info("Loaded config from %s", config_path)
    logger.info("Configured sectors: %s", ", ".join(config["tickers"].keys()))
    logger.info("Configured tickers: %s", ", ".join(tickers))

    adjusted_close = download_adjusted_close(
        tickers=tickers,
        start_date=config["dataset"]["start_date"],
        end_date=config["dataset"]["end_date"],
        cache_dir=PROJECT_ROOT / "data" / "interim" / "yfinance_cache",
    )

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    tables_dir = PROJECT_ROOT / config["outputs"]["tables"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    save_raw_download(
        adjusted_close=adjusted_close,
        output_path=raw_dir / "yfinance_adjusted_close_wide.csv",
        metadata_path=raw_dir / "yfinance_download_metadata.json",
        config=config,
    )

    quality_report = create_data_quality_report(adjusted_close, sectors)
    quality_report_path = tables_dir / "data_quality_report.csv"
    quality_report.to_csv(quality_report_path, index=False)
    logger.info("Saved data quality report to %s", quality_report_path)

    cleaned_long = create_clean_long_dataset(adjusted_close, sectors)
    cleaned_path = processed_dir / "finance_adjusted_close_long.csv"
    cleaned_long.to_csv(cleaned_path, index=False)
    logger.info("Saved cleaned long-format dataset to %s", cleaned_path)
    logger.info("Cleaned dataset rows: %d", len(cleaned_long))


if __name__ == "__main__":
    main()
