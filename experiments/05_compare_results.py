"""Step 05: compare baseline and statistical augmentation results."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_finance_data import load_config

PAIR_KEYS = ["sample_id", "ticker", "sector", "cold_start_weeks", "model"]
SUMMARY_COLUMNS = [
    "number_of_comparisons",
    "mean_baseline_MAPE",
    "mean_augmented_MAPE",
    "mean_MAPE_difference",
    "mean_percent_change_MAPE",
    "improvement_rate",
]


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
    output_dir = PROJECT_ROOT / config["outputs"]["tables"]
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "baseline_results.csv"
    augmented_path = output_dir / "statistical_augmentation_results.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline results not found at {baseline_path}")
    if not augmented_path.exists():
        raise FileNotFoundError(f"statistical augmentation results not found at {augmented_path}")

    baseline = pd.read_csv(baseline_path)
    augmented = pd.read_csv(augmented_path)
    logger.info("Loaded %d baseline rows from %s", len(baseline), baseline_path)
    logger.info("Loaded %d augmentation rows from %s", len(augmented), augmented_path)

    comparison = create_paired_comparison(baseline, augmented)
    comparison_path = output_dir / "baseline_vs_statistical_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info("Wrote %d paired comparisons to %s", len(comparison), comparison_path)

    write_summary_tables(comparison, output_dir)
    print_plain_english_summary(comparison)


def create_paired_comparison(baseline: pd.DataFrame, augmented: pd.DataFrame) -> pd.DataFrame:
    """Pair baseline rows with each statistical augmentation ratio."""
    baseline_none = baseline[baseline["augmentation_method"] == "none"].copy()
    augmented_statistical = augmented[augmented["augmentation_method"] == "statistical"].copy()

    baseline_columns = PAIR_KEYS + ["MAE", "RMSE", "MAPE"]
    augmented_columns = PAIR_KEYS + [
        "augmentation_ratio",
        "n_synthetic_series",
        "MAE",
        "RMSE",
        "MAPE",
    ]
    paired = augmented_statistical[augmented_columns].merge(
        baseline_none[baseline_columns],
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_augmented", "_baseline"),
        validate="many_to_one",
    )
    if paired.empty:
        raise ValueError("no paired baseline/statistical augmentation rows were found")

    paired["augmentation_method_baseline"] = "none"
    paired["augmentation_method_augmented"] = "statistical"

    for metric in ("MAE", "RMSE", "MAPE"):
        paired[f"baseline_{metric}"] = paired[f"{metric}_baseline"]
        paired[f"augmented_{metric}"] = paired[f"{metric}_augmented"]
        paired[f"{metric}_difference"] = paired[f"augmented_{metric}"] - paired[f"baseline_{metric}"]
    paired["percent_change_MAPE"] = _percent_change(
        paired["baseline_MAPE"],
        paired["MAPE_difference"],
        "MAPE",
    )
    paired["improved"] = paired["augmented_MAPE"] < paired["baseline_MAPE"]

    columns = [
        *PAIR_KEYS,
        "augmentation_method_baseline",
        "augmentation_method_augmented",
        "augmentation_ratio",
        "n_synthetic_series",
        "baseline_MAE",
        "augmented_MAE",
        "MAE_difference",
        "baseline_RMSE",
        "augmented_RMSE",
        "RMSE_difference",
        "baseline_MAPE",
        "augmented_MAPE",
        "MAPE_difference",
        "percent_change_MAPE",
        "improved",
    ]
    return paired[columns].sort_values(PAIR_KEYS + ["augmentation_ratio"]).reset_index(drop=True)


def write_summary_tables(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Write academic interpretation summary tables."""
    summary_specs = {
        "summary_by_cold_start_window.csv": ["cold_start_weeks", "model", "augmentation_ratio"],
        "summary_by_sector.csv": ["sector", "model", "augmentation_ratio"],
        "summary_by_model.csv": ["model", "augmentation_ratio"],
        "summary_by_augmentation_ratio.csv": ["augmentation_ratio", "model"],
    }

    for filename, group_columns in summary_specs.items():
        summary = summarize_comparison(comparison, group_columns)
        output_path = output_dir / filename
        summary.to_csv(output_path, index=False)


def summarize_comparison(comparison: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize paired comparisons while preserving model and ratio factors."""
    summary = (
        comparison.groupby(group_columns, dropna=False)
        .agg(
            number_of_comparisons=("sample_id", "count"),
            mean_baseline_MAPE=("baseline_MAPE", "mean"),
            mean_augmented_MAPE=("augmented_MAPE", "mean"),
            mean_MAPE_difference=("MAPE_difference", "mean"),
            mean_percent_change_MAPE=("percent_change_MAPE", "mean"),
            improvement_rate=("improved", "mean"),
        )
        .reset_index()
    )
    return summary[group_columns + SUMMARY_COLUMNS]


def print_plain_english_summary(comparison: pd.DataFrame) -> None:
    """Print a concise interpretation of augmentation performance."""
    improvement_rate = comparison["improved"].mean() * 100
    mean_difference = comparison["MAPE_difference"].mean()
    mean_percent_change = comparison["percent_change_MAPE"].mean()
    best_ratio = (
        comparison.groupby("augmentation_ratio")["MAPE_difference"].mean().sort_values().index[0]
    )

    direction = "improves" if mean_difference < 0 else "worsens"
    print("\nPlain-English summary:")
    print(
        f"Across all paired sample-model-ratio comparisons, statistical augmentation {direction} "
        f"forecasting performance on average by {abs(mean_difference):.4f} MAPE points "
        f"({mean_percent_change:.2f}% mean percent change)."
    )
    print(f"The paired improvement rate is {improvement_rate:.2f}%.")
    print(f"The lowest average MAPE difference occurs at augmentation_ratio={best_ratio}.")


def _percent_change(baseline_values: pd.Series, differences: pd.Series, metric: str) -> pd.Series:
    if (baseline_values == 0).any():
        raise ValueError(f"cannot compute percent change because baseline_{metric} contains zero")
    return differences / baseline_values * 100


if __name__ == "__main__":
    main()
