"""Step 07: diagnose where statistical augmentation helps or hurts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_finance_data import load_config

SUMMARY_METRICS = {
    "mean_baseline_MAPE": ("baseline_MAPE", "mean"),
    "mean_augmented_MAPE": ("augmented_MAPE", "mean"),
    "mean_MAPE_difference": ("MAPE_difference", "mean"),
    "improvement_rate": ("improved", "mean"),
}
CASE_COLUMNS = [
    "sample_id",
    "ticker",
    "sector",
    "model",
    "cold_start_weeks",
    "augmentation_ratio",
    "n_synthetic_series",
    "baseline_MAPE",
    "augmented_MAPE",
    "MAPE_difference",
    "percent_change_MAPE",
    "improved",
]
OUTLIER_PERCENT_CHANGE_THRESHOLD = 500.0


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    config = load_config(PROJECT_ROOT / "config.yaml")
    tables_dir = PROJECT_ROOT / config["outputs"]["tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "comparison": tables_dir / "baseline_vs_statistical_comparison.csv",
        "baseline_predictions": tables_dir / "baseline_predictions.csv",
        "augmentation_predictions": tables_dir / "statistical_augmentation_predictions.csv",
        "scenarios": tables_dir / "cold_start_scenarios.csv",
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"required input {name} not found at {path}")

    comparison = pd.read_csv(inputs["comparison"])
    baseline_predictions = pd.read_csv(inputs["baseline_predictions"])
    augmentation_predictions = pd.read_csv(inputs["augmentation_predictions"])
    scenarios = pd.read_csv(inputs["scenarios"])
    logger.info("Loaded %d paired comparison rows", len(comparison))
    logger.info("Loaded %d baseline prediction rows", len(baseline_predictions))
    logger.info("Loaded %d augmentation prediction rows", len(augmentation_predictions))
    logger.info("Loaded %d scenario metadata rows", len(scenarios))

    write_diagnostic_tables(comparison, tables_dir)
    print_plain_english_summary(comparison)


def write_diagnostic_tables(comparison: pd.DataFrame, tables_dir: Path) -> None:
    """Create all requested diagnostic CSV tables."""
    best_improvements = (
        comparison[comparison["MAPE_difference"] < 0]
        .sort_values("MAPE_difference", ascending=True)
        .head(50)[CASE_COLUMNS]
    )
    best_improvements.to_csv(tables_dir / "diagnostic_best_improvements.csv", index=False)

    worst_degradations = (
        comparison[comparison["MAPE_difference"] > 0]
        .sort_values("MAPE_difference", ascending=False)
        .head(50)[CASE_COLUMNS]
    )
    worst_degradations.to_csv(tables_dir / "diagnostic_worst_degradations.csv", index=False)

    improvement_rate_matrix = (
        comparison.groupby(["model", "cold_start_weeks", "augmentation_ratio"], dropna=False)
        .agg(
            number_of_comparisons=("sample_id", "count"),
            improvement_rate=("improved", "mean"),
            mean_MAPE_difference=("MAPE_difference", "mean"),
            mean_percent_change_MAPE=("percent_change_MAPE", "mean"),
        )
        .reset_index()
    )
    improvement_rate_matrix.to_csv(
        tables_dir / "diagnostic_improvement_rate_matrix.csv",
        index=False,
    )

    sector_model_summary = _summary_by(comparison, ["sector", "model", "augmentation_ratio"])
    sector_model_summary.to_csv(tables_dir / "diagnostic_sector_model_summary.csv", index=False)

    ticker_summary = _summary_by(comparison, ["ticker"])
    ticker_summary.to_csv(tables_dir / "diagnostic_ticker_summary.csv", index=False)

    outlier_cases = comparison[
        comparison["percent_change_MAPE"] >= OUTLIER_PERCENT_CHANGE_THRESHOLD
    ].sort_values("percent_change_MAPE", ascending=False)
    outlier_cases[CASE_COLUMNS].to_csv(tables_dir / "diagnostic_outlier_cases.csv", index=False)


def _summary_by(comparison: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = comparison.groupby(group_columns, dropna=False).agg(**SUMMARY_METRICS).reset_index()
    summary.insert(
        len(group_columns),
        "number_of_comparisons",
        comparison.groupby(group_columns, dropna=False)["sample_id"].count().to_numpy(),
    )
    return summary


def print_plain_english_summary(comparison: pd.DataFrame) -> None:
    """Print diagnostic interpretation without proposing new methods."""
    by_ratio = (
        comparison.groupby("augmentation_ratio")
        .agg(
            mean_MAPE_difference=("MAPE_difference", "mean"),
            improvement_rate=("improved", "mean"),
            mean_percent_change_MAPE=("percent_change_MAPE", "mean"),
        )
        .reset_index()
        .sort_values("mean_MAPE_difference")
    )
    by_model = (
        comparison.groupby("model")
        .agg(
            mean_MAPE_difference=("MAPE_difference", "mean"),
            improvement_rate=("improved", "mean"),
        )
        .reset_index()
        .sort_values("mean_MAPE_difference")
    )
    by_sector = (
        comparison.groupby("sector")
        .agg(
            mean_MAPE_difference=("MAPE_difference", "mean"),
            improvement_rate=("improved", "mean"),
        )
        .reset_index()
        .sort_values("mean_MAPE_difference")
    )
    by_window = (
        comparison.groupby("cold_start_weeks")
        .agg(
            mean_MAPE_difference=("MAPE_difference", "mean"),
            improvement_rate=("improved", "mean"),
        )
        .reset_index()
        .sort_values("mean_MAPE_difference")
    )
    outlier_count = int((comparison["percent_change_MAPE"] >= OUTLIER_PERCENT_CHANGE_THRESHOLD).sum())
    improvement_count = int(comparison["improved"].sum())

    best_ratio, worst_ratio = by_ratio.iloc[0], by_ratio.iloc[-1]
    best_model, worst_model = by_model.iloc[0], by_model.iloc[-1]
    best_sector, worst_sector = by_sector.iloc[0], by_sector.iloc[-1]
    best_window, worst_window = by_window.iloc[0], by_window.iloc[-1]

    print("\nPlain-English diagnostic summary")
    print(
        f"Augmentation improved {improvement_count} of {len(comparison)} paired comparisons "
        f"({comparison['improved'].mean() * 100:.2f}%)."
    )
    print(
        f"The best mean effect is at augmentation_ratio={best_ratio['augmentation_ratio']} "
        f"(MAPE difference {best_ratio['mean_MAPE_difference']:.4f}); the weakest is at "
        f"ratio={worst_ratio['augmentation_ratio']} ({worst_ratio['mean_MAPE_difference']:.4f})."
    )
    print(
        "All synthetic histories remain aligned to the real training window and preserve its endpoint."
    )
    print(
        f"By model, the best average effect is for {best_model['model']} "
        f"({best_model['mean_MAPE_difference']:.4f}); the weakest is for {worst_model['model']} "
        f"({worst_model['mean_MAPE_difference']:.4f})."
    )
    print(
        f"By sector, the best average effect is in {best_sector['sector']} "
        f"({best_sector['mean_MAPE_difference']:.4f}); the weakest is in {worst_sector['sector']} "
        f"({worst_sector['mean_MAPE_difference']:.4f})."
    )
    print(
        f"By cold-start length, the best average effect is for {int(best_window['cold_start_weeks'])}-week "
        f"windows and the weakest for {int(worst_window['cold_start_weeks'])}-week windows."
    )
    print(
        "The naive model is expected to be unchanged because every synthetic history shares the real endpoint."
    )
    print(
        f"There are {outlier_count} outlier cases with percent_change_MAPE >= "
        f"{OUTLIER_PERCENT_CHANGE_THRESHOLD:.0f}%."
    )
    print(
        "Subgroup tables should be read alongside the overall paired statistical tests."
    )


if __name__ == "__main__":
    main()
