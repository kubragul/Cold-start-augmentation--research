"""Step 06: run paired statistical tests on augmentation comparisons."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_finance_data import load_config
from src.evaluation.statistical_tests import (
    benjamini_hochberg_adjust,
    interpret_test_with_adjusted_p,
    run_paired_error_tests,
)

METRICS = ("MAE", "RMSE", "MAPE")
GROUPINGS = (
    ("overall", None),
    ("cold_start_weeks", "cold_start_weeks"),
    ("model", "model"),
    ("augmentation_ratio", "augmentation_ratio"),
    ("sector", "sector"),
)
OUTPUT_COLUMNS = [
    "metric",
    "grouping_variable",
    "group_value",
    "test_name",
    "statistic",
    "p_value",
    "p_value_fdr_bh",
    "mean_baseline_error",
    "mean_augmented_error",
    "mean_difference",
    "percent_change",
    "n_pairs",
    "interpretation",
    "interpretation_fdr_bh",
]


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
    comparison_path = PROJECT_ROOT / config["outputs"]["tables"] / "baseline_vs_statistical_comparison.csv"
    output_dir = PROJECT_ROOT / config["outputs"]["tests"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "significance_tests.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"comparison file not found at {comparison_path}; run experiments/05_compare_results.py first"
        )

    comparison = pd.read_csv(comparison_path)
    logger.info("Loaded %d paired comparison rows from %s", len(comparison), comparison_path)

    results = run_all_grouped_tests(comparison)
    results = add_multiple_testing_correction(results)
    results.to_csv(output_path, index=False)
    logger.info("Wrote %d statistical test rows to %s", len(results), output_path)

    print_requested_mape_outputs(results)
    print_plain_english_interpretation(results)


def run_all_grouped_tests(comparison: pd.DataFrame) -> pd.DataFrame:
    """Run paired tests for every requested metric and grouping."""
    rows = []
    for metric in METRICS:
        baseline_col = f"baseline_{metric}"
        augmented_col = f"augmented_{metric}"
        for grouping_variable, column in GROUPINGS:
            if column is None:
                rows.extend(
                    run_paired_error_tests(
                        baseline_errors=comparison[baseline_col].tolist(),
                        augmented_errors=comparison[augmented_col].tolist(),
                        metric=metric,
                        grouping_variable=grouping_variable,
                        group_value="all",
                    )
                )
            else:
                for group_value, group in comparison.groupby(column, sort=True, dropna=False):
                    rows.extend(
                        run_paired_error_tests(
                            baseline_errors=group[baseline_col].tolist(),
                            augmented_errors=group[augmented_col].tolist(),
                            metric=metric,
                            grouping_variable=grouping_variable,
                            group_value=group_value,
                        )
                    )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def add_multiple_testing_correction(results: pd.DataFrame) -> pd.DataFrame:
    """Add Benjamini-Hochberg FDR-adjusted p-values and interpretation."""
    corrected = results.copy()
    corrected["p_value_fdr_bh"] = benjamini_hochberg_adjust(corrected["p_value"].tolist())
    corrected["interpretation_fdr_bh"] = corrected.apply(
        lambda row: interpret_test_with_adjusted_p(
            mean_difference=float(row["mean_difference"]),
            adjusted_p_value=float(row["p_value_fdr_bh"]),
        ),
        axis=1,
    )
    return corrected


def print_requested_mape_outputs(results: pd.DataFrame) -> None:
    """Print the MAPE-focused outputs requested for review."""
    mape = results[results["metric"] == "MAPE"].copy()
    primary = mape[mape["test_name"] == "wilcoxon_signed_rank"].copy()

    print("\n1. Overall MAPE test results")
    print(mape[mape["grouping_variable"] == "overall"].to_string(index=False))

    print("\n2. MAPE results by cold_start_weeks")
    print(primary[primary["grouping_variable"] == "cold_start_weeks"].to_string(index=False))

    print("\n3. MAPE results by model")
    print(primary[primary["grouping_variable"] == "model"].to_string(index=False))

    print("\n4. MAPE results by augmentation_ratio")
    print(primary[primary["grouping_variable"] == "augmentation_ratio"].to_string(index=False))


def print_plain_english_interpretation(results: pd.DataFrame) -> None:
    """Print a concise interpretation emphasizing the primary Wilcoxon test."""
    overall_mape = results[
        (results["metric"] == "MAPE")
        & (results["grouping_variable"] == "overall")
        & (results["test_name"] == "wilcoxon_signed_rank")
    ].iloc[0]

    print("\n5. Plain-English interpretation")
    if overall_mape["interpretation"] == "significantly_improved":
        print(
            "Using the primary Wilcoxon signed-rank test, statistical augmentation "
            "significantly improves MAPE overall."
        )
    elif overall_mape["interpretation"] == "significantly_worsened":
        print(
            "Using the primary Wilcoxon signed-rank test, statistical augmentation "
            "significantly worsens MAPE overall. This negative result should be reported clearly."
        )
    else:
        print(
            "Using the primary Wilcoxon signed-rank test, there is no statistically significant "
            "overall MAPE difference between baseline and statistical augmentation."
        )

    print(
        f"Overall mean baseline MAPE = {overall_mape['mean_baseline_error']:.4f}; "
        f"mean augmented MAPE = {overall_mape['mean_augmented_error']:.4f}; "
        f"mean difference = {overall_mape['mean_difference']:.4f}; "
        f"p-value = {overall_mape['p_value']:.6g}; "
        f"FDR-adjusted p-value = {overall_mape['p_value_fdr_bh']:.6g}."
    )


if __name__ == "__main__":
    main()
