"""Rolling cold-start scenario creation for finance forecasting experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRADING_DAYS_PER_WEEK = 5


def weeks_to_trading_days(weeks: int) -> int:
    """Convert calendar-style research window labels to trading-day counts."""
    return weeks * TRADING_DAYS_PER_WEEK


def create_rolling_cold_start_scenarios(
    data: pd.DataFrame,
    cold_start_windows_weeks: list[int],
    forecast_horizon_days: int,
    rolling_step_days: int = 28,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """Create rolling cold-start train/test samples for every ticker.

    Rolling windows improve academic validity because they evaluate the same
    method across many historical periods. A single first-N-observation split
    can overstate or understate performance if that initial period happens to
    contain unusual volatility, market stress, or unusually smooth trends.
    """
    required_columns = {"date", "ticker", "sector", "y"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"input data is missing required columns: {sorted(missing_columns)}")
    if forecast_horizon_days <= 0:
        raise ValueError("forecast_horizon_days must be positive")
    if rolling_step_days <= 0:
        raise ValueError("rolling_step_days must be positive")

    clean_data = data.copy()
    clean_data["date"] = pd.to_datetime(clean_data["date"])
    clean_data = clean_data.sort_values(["ticker", "date"]).reset_index(drop=True)

    metadata_rows = []
    sample_data = {}

    for ticker, ticker_data in clean_data.groupby("ticker", sort=True):
        ticker_data = ticker_data.sort_values("date").reset_index(drop=True)
        sector = str(ticker_data["sector"].iloc[0])

        for cold_start_weeks in cold_start_windows_weeks:
            train_days = weeks_to_trading_days(int(cold_start_weeks))
            total_window_days = train_days + forecast_horizon_days
            if len(ticker_data) < total_window_days:
                continue

            sample_number = 0
            for start_index in range(0, len(ticker_data) - total_window_days + 1, rolling_step_days):
                train = ticker_data.iloc[start_index : start_index + train_days].copy()
                test = ticker_data.iloc[
                    start_index + train_days : start_index + total_window_days
                ].copy()

                sample_id = f"{ticker}_w{int(cold_start_weeks):02d}_s{sample_number:04d}"
                train_start_date = train["date"].iloc[0].date().isoformat()
                train_end_date = train["date"].iloc[-1].date().isoformat()
                test_start_date = test["date"].iloc[0].date().isoformat()
                test_end_date = test["date"].iloc[-1].date().isoformat()

                train = _format_sample_frame(train, sample_id, "train", int(cold_start_weeks))
                test = _format_sample_frame(test, sample_id, "test", int(cold_start_weeks))

                metadata_rows.append(
                    {
                        "sample_id": sample_id,
                        "ticker": ticker,
                        "sector": sector,
                        "train_start_date": train_start_date,
                        "train_end_date": train_end_date,
                        "test_start_date": test_start_date,
                        "test_end_date": test_end_date,
                        "cold_start_weeks": int(cold_start_weeks),
                    }
                )
                sample_data[sample_id] = (train, test)
                sample_number += 1

    metadata = pd.DataFrame(
        metadata_rows,
        columns=[
            "sample_id",
            "ticker",
            "sector",
            "train_start_date",
            "train_end_date",
            "test_start_date",
            "test_end_date",
            "cold_start_weeks",
        ],
    )
    return metadata, sample_data


def save_scenario_sample_files(
    metadata: pd.DataFrame,
    sample_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: str | Path,
    path_root: str | Path | None = None,
) -> pd.DataFrame:
    """Save each train/test sample as CSV and append paths to metadata.

    When ``path_root`` is given, the recorded ``train_path``/``test_path``
    values are stored relative to it. Recording relative paths keeps the
    scenario metadata portable: absolute paths embed the machine that ran the
    experiment and break reproduction on any other checkout.
    """
    output_dir = Path(output_dir)
    path_root = Path(path_root) if path_root is not None else None
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    metadata = metadata.copy()
    train_paths = []
    test_paths = []

    for sample_id in metadata["sample_id"]:
        train, test = sample_data[sample_id]
        train_path = train_dir / f"{sample_id}_train.csv"
        test_path = test_dir / f"{sample_id}_test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        train_paths.append(str(_record_path(train_path, path_root)))
        test_paths.append(str(_record_path(test_path, path_root)))

    metadata["train_path"] = train_paths
    metadata["test_path"] = test_paths
    return metadata


def _record_path(path: Path, path_root: Path | None) -> Path:
    """Return the path to record in metadata, relative to ``path_root`` if given."""
    if path_root is None:
        return path
    try:
        return path.resolve().relative_to(path_root.resolve())
    except ValueError:
        return path


def _format_sample_frame(
    frame: pd.DataFrame,
    sample_id: str,
    split: str,
    cold_start_weeks: int,
) -> pd.DataFrame:
    """Return a stable sample-level CSV schema."""
    formatted = frame.copy()
    formatted["date"] = formatted["date"].dt.date.astype(str)
    formatted.insert(0, "sample_id", sample_id)
    formatted["split"] = split
    formatted["cold_start_weeks"] = cold_start_weeks
    return formatted[["sample_id", "split", "date", "ticker", "sector", "cold_start_weeks", "y"]]
