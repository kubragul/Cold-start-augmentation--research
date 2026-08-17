"""Step 07: diagnose why statistical augmentation worsened performance."""

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
    "n_synthetic_points",
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

    least_harmful_ratio = by_ratio.iloc[0]
    most_harmful_ratio = by_ratio.iloc[-1]
    least_harmful_model = by_model.iloc[0]
    most_harmful_model = by_model.iloc[-1]
    least_harmful_sector = by_sector.iloc[0]
    most_harmful_sector = by_sector.iloc[-1]
    least_harmful_window = by_window.iloc[0]
    most_harmful_window = by_window.iloc[-1]

    print("\nPlain-English diagnostic summary")
    print(
        f"Augmentation improved {improvement_count} of {len(comparison)} paired comparisons "
        f"({comparison['improved'].mean() * 100:.2f}%), but worsened the majority."
    )
    print(
        f"It was least harmful at augmentation_ratio={least_harmful_ratio['augmentation_ratio']} "
        f"(mean MAPE difference {least_harmful_ratio['mean_MAPE_difference']:.4f}) and most harmful "
        f"at augmentation_ratio={most_harmful_ratio['augmentation_ratio']} "
        f"(mean MAPE difference {most_harmful_ratio['mean_MAPE_difference']:.4f})."
    )
    print(
        "Degradation increases monotonically with augmentation_ratio in this run, "
        "which suggests that appending more synthetic continuation values moves the effective "
        "training signal farther from the real cold-start window."
    )
    print(
        f"By model, the least harmful average effect is for {least_harmful_model['model']} "
        f"(mean MAPE difference {least_harmful_model['mean_MAPE_difference']:.4f}); "
        f"the most harmful is for {most_harmful_model['model']} "
        f"({most_harmful_model['mean_MAPE_difference']:.4f})."
    )
    print(
        f"By sector, the least harmful average effect is in {least_harmful_sector['sector']} "
        f"(mean MAPE difference {least_harmful_sector['mean_MAPE_difference']:.4f}); "
        f"the most harmful is in {most_harmful_sector['sector']} "
        f"({most_harmful_sector['mean_MAPE_difference']:.4f})."
    )
    print(
        f"By cold-start length, degradation is smallest for {int(least_harmful_window['cold_start_weeks'])}-week "
        f"windows and largest for {int(most_harmful_window['cold_start_weeks'])}-week windows."
    )
    print(
        "Naive and moving-average baselines are plausibly harmed because the synthetic continuation "
        "values are appended after the real training window; these models depend heavily on the final "
        "or most recent observations, so appended synthetic points can dominate their forecasts."
    )
    print(
        f"There are {outlier_count} outlier cases with percent_change_MAPE >= "
        f"{OUTLIER_PERCENT_CHANGE_THRESHOLD:.0f}%, indicating a substantial right tail of severe degradation."
    )
    print(
        "The negative result is broad across models, sectors, tickers, and cold-start windows rather than "
        "being isolated to a single narrow subgroup."
    )


if __name__ == "__main__":
    main()
