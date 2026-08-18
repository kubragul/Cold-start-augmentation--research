"""Step 08: generate paper-ready figures and captions."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "interim" / "matplotlib_cache"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_finance_data import load_config
from src.visualization.paper_figures import (
    baseline_mape_by_model_window,
    baseline_vs_augmented_distribution,
    configure_academic_style,
    example_forecast_case,
    improvement_rate_by_model_ratio,
    mape_difference_heatmap,
    augmentation_mape_by_ratio_window,
    save_figure,
    sector_level_mape_difference,
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    configure_academic_style()

    config = load_config(PROJECT_ROOT / "config.yaml")
    tables_dir = PROJECT_ROOT / config["outputs"]["tables"]
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(tables_dir, PROJECT_ROOT / config["outputs"]["tests"])
    captions = []

    figure_specs = [
        (
            "baseline_mape_by_model_window",
            baseline_mape_by_model_window(data["baseline"]),
            "Baseline MAPE by model and cold-start window",
            "Mean baseline MAPE across rolling cold-start samples, grouped by model and training-window length.",
            "Compares baseline difficulty across models and cold-start windows before augmentation.",
        ),
        (
            "augmentation_mape_by_ratio_window",
            augmentation_mape_by_ratio_window(data["augmentation"]),
            "Augmented MAPE by augmentation ratio and cold-start window",
            "Mean MAPE under endpoint-preserving residual-bootstrap augmentation.",
            "Compares ensemble sizes while keeping every forecast origin aligned.",
        ),
        (
            "mape_difference_heatmap",
            mape_difference_heatmap(data["comparison"]),
            "Mean MAPE difference heatmap",
            "Mean MAPE difference, computed as augmented MAPE minus baseline MAPE. Positive values indicate worse augmented performance.",
            "Identifies where statistical continuation augmentation helps or hurts.",
        ),
        (
            "improvement_rate_by_model_ratio",
            improvement_rate_by_model_ratio(data["diagnostic_matrix"]),
            "Improvement rate by model and augmentation ratio",
            "Fraction of paired comparisons in which augmented MAPE was lower than baseline MAPE.",
            "Shows the paired improvement frequency by model and ensemble size.",
        ),
        (
            "sector_level_mape_difference",
            sector_level_mape_difference(data["comparison"]),
            "Sector-level MAPE difference",
            "Mean MAPE difference by sector and augmentation ratio. Positive values indicate degradation under augmentation.",
            "Compares sector-level effects while keeping augmentation ratio separate.",
        ),
        (
            "baseline_vs_augmented_distribution",
            baseline_vs_augmented_distribution(data["comparison"]),
            "Distribution of baseline and augmented MAPE",
            "Boxplots compare the distribution of baseline and augmented MAPE by augmentation ratio, excluding plotted outliers for readability.",
            "Shows distributional shift rather than only average change.",
        ),
        (
            "baseline_vs_augmented_distribution_zoomed",
            baseline_vs_augmented_distribution(data["comparison"], zoomed=True),
            "Distribution of baseline and augmented MAPE, zoomed",
            "Boxplots compare baseline and augmented MAPE with the display limited to 0-40 MAPE. Extreme values are excluded only visually, not analytically.",
            "Shows the central distribution while retaining all outliers in analysis tables.",
        ),
    ]

    failure_case = data["comparison"].sort_values("MAPE_difference", ascending=False).iloc[0]
    improvement_case = data["comparison"][data["comparison"]["MAPE_difference"] < 0].sort_values(
        "MAPE_difference",
        ascending=True,
    ).iloc[0]
    figure_specs.extend(
        [
            (
                "example_forecast_failure_case",
                example_forecast_case(failure_case, data["scenarios"], int(config["random_seed"])),
                "Representative forecast failure case",
                "Example where endpoint-preserving augmentation increased MAPE despite improving the overall mean.",
                "Shows that the average benefit does not extend to every scenario.",
            ),
            (
                "example_forecast_improvement_case",
                example_forecast_case(improvement_case, data["scenarios"], int(config["random_seed"])),
                "Representative forecast improvement case",
                "Example where endpoint-preserving augmentation reduced MAPE.",
                "Illustrates how aligned synthetic-history ensembling can improve a forecast.",
            ),
        ]
    )

    for figure_id, fig, title, caption, message in figure_specs:
        png_path, pdf_path = save_figure(fig, figures_dir, figure_id)
        captions.append(
            {
                "figure_id": figure_id,
                # Recorded relative to the project root so the caption table
                # stays portable across machines and checkouts.
                "figure_file_png": str(_relative_to_root(png_path)),
                "figure_file_pdf": str(_relative_to_root(pdf_path)),
                "figure_title": title,
                "suggested_caption": caption,
                "research_message": message,
            }
        )
        logger.info("Saved %s as PNG and PDF", figure_id)

    captions_frame = pd.DataFrame(captions)
    for filename in ("figure_captions.csv", "figure_captions_revised.csv"):
        captions_path = tables_dir / filename
        captions_frame.to_csv(captions_path, index=False)
        logger.info("Wrote figure captions to %s", captions_path)


def _relative_to_root(path: Path) -> Path:
    """Return ``path`` relative to the project root when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return path


def load_inputs(tables_dir: Path, tests_dir: Path) -> dict[str, pd.DataFrame]:
    required_paths = {
        "baseline": tables_dir / "baseline_results.csv",
        "augmentation": tables_dir / "statistical_augmentation_results.csv",
        "comparison": tables_dir / "baseline_vs_statistical_comparison.csv",
        "summary_window": tables_dir / "summary_by_cold_start_window.csv",
        "summary_model": tables_dir / "summary_by_model.csv",
        "summary_ratio": tables_dir / "summary_by_augmentation_ratio.csv",
        "significance": tests_dir / "significance_tests.csv",
        "diagnostic_matrix": tables_dir / "diagnostic_improvement_rate_matrix.csv",
        "scenarios": tables_dir / "cold_start_scenarios.csv",
    }
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"required input {name} not found at {path}")
    return {name: pd.read_csv(path) for name, path in required_paths.items()}


if __name__ == "__main__":
    main()
