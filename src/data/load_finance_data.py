"""Finance data loading utilities for the academic pilot experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# yfinance is imported lazily inside download_adjusted_close: steps 02-08 only
# need load_config from this module and must stay runnable offline without it.

LOGGER = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the experiment configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ticker_sector_map(config: dict[str, Any]) -> dict[str, str]:
    """Return a mapping from ticker to sector using the config ticker groups."""
    mapping = {}
    for sector, tickers in config["tickers"].items():
        for ticker in tickers:
            mapping[ticker] = sector
    return mapping


def download_adjusted_close(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download daily adjusted close prices from yfinance.

    The function requests unadjusted and adjusted fields, then selects
    ``Adj Close`` when available. If yfinance returns only adjusted ``Close``
    values for a ticker, that field is used as a transparent fallback.
    """
    import yfinance as yf

    LOGGER.info(
        "Downloading daily prices for %d tickers from %s to %s",
        len(tickers),
        start_date,
        end_date,
    )
    if cache_dir is not None and hasattr(yf, "set_tz_cache_location"):
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_path))
        LOGGER.info("Using project-local yfinance cache at %s", cache_path)

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise ValueError("yfinance returned no rows for the configured date range")

    adjusted_close = _select_adjusted_close(raw, tickers)
    adjusted_close = adjusted_close.sort_index()
    adjusted_close.index.name = "date"
    return adjusted_close


def save_raw_download(
    adjusted_close: pd.DataFrame,
    output_path: str | Path,
    metadata_path: str | Path,
    config: dict[str, Any],
) -> None:
    """Save raw downloaded prices and metadata for reproducibility."""
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    adjusted_close.to_csv(output_path)
    metadata = {
        "source": config["dataset"]["source"],
        "domain": config["dataset"]["domain"],
        "start_date": str(config["dataset"]["start_date"]),
        "end_date": str(config["dataset"]["end_date"]),
        "tickers": ticker_sector_map(config),
        "price_field": "Adj Close when available; Close fallback if necessary",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved raw adjusted-close matrix to %s", output_path)
    LOGGER.info("Saved raw download metadata to %s", metadata_path)


def create_data_quality_report(
    adjusted_close: pd.DataFrame,
    sectors: dict[str, str],
) -> pd.DataFrame:
    """Create a transparent ticker-level missingness report."""
    rows = []
    total_dates = len(adjusted_close.index)

    for ticker in sectors:
        series = adjusted_close[ticker] if ticker in adjusted_close.columns else pd.Series(dtype=float)
        observed = series.dropna()
        missing_values = int(series.isna().sum()) if ticker in adjusted_close.columns else total_dates
        percentage_missing = (missing_values / total_dates * 100) if total_dates else 0.0
        rows.append(
            {
                "ticker": ticker,
                "sector": sectors[ticker],
                "number_of_observations": int(observed.shape[0]),
                "first_date": observed.index.min().date().isoformat() if not observed.empty else "",
                "last_date": observed.index.max().date().isoformat() if not observed.empty else "",
                "missing_values": missing_values,
                "percentage_missing": round(percentage_missing, 4),
            }
        )

    return pd.DataFrame(rows)


def create_clean_long_dataset(
    adjusted_close: pd.DataFrame,
    sectors: dict[str, str],
) -> pd.DataFrame:
    """Convert wide adjusted-close data to cleaned long format.

    Missing prices are not imputed in this first reproducibility step. They are
    measured in the quality report, then removed from the cleaned modeling file.
    """
    available_tickers = [ticker for ticker in sectors if ticker in adjusted_close.columns]
    missing_tickers = [ticker for ticker in sectors if ticker not in adjusted_close.columns]
    if missing_tickers:
        LOGGER.warning("No downloaded price column found for tickers: %s", ", ".join(missing_tickers))

    long_data = (
        adjusted_close[available_tickers]
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="y")
    )
    missing_rows = int(long_data["y"].isna().sum())
    if missing_rows:
        LOGGER.info("Dropping %d missing ticker-date observations from cleaned dataset", missing_rows)

    long_data = long_data.dropna(subset=["y"]).copy()
    long_data["sector"] = long_data["ticker"].map(sectors)
    long_data["date"] = pd.to_datetime(long_data["date"]).dt.date.astype(str)
    long_data = long_data[["date", "ticker", "sector", "y"]].sort_values(["ticker", "date"])
    return long_data.reset_index(drop=True)


def _select_adjusted_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Select adjusted-close data from yfinance's single or multi-index output."""
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            adjusted_close = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            LOGGER.warning("Adj Close was unavailable; using Close as fallback")
            adjusted_close = raw["Close"].copy()
        else:
            raise ValueError("downloaded data contains neither Adj Close nor Close")
    else:
        if "Adj Close" in raw.columns:
            adjusted_close = raw[["Adj Close"]].copy()
            adjusted_close.columns = [tickers[0]]
        elif "Close" in raw.columns:
            LOGGER.warning("Adj Close was unavailable; using Close as fallback")
            adjusted_close = raw[["Close"]].copy()
            adjusted_close.columns = [tickers[0]]
        else:
            raise ValueError("downloaded data contains neither Adj Close nor Close")

    adjusted_close = adjusted_close.reindex(columns=tickers)
    adjusted_close = adjusted_close.apply(pd.to_numeric, errors="coerce")
    return adjusted_close
