"""Paper-ready visualization utilities for the augmentation study."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/interim/matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.augmentation.statistical_augmentation import (
    augmented_ensemble_forecast,
    generate_endpoint_preserving_histories,
)
from src.models.forecasting_models import (
    linear_trend_forecast,
    moving_average_forecast,
    naive_forecast,
)

DPI = 300
AUGMENTATION_RATIOS = (0.5, 1.0, 2.0)
MODEL_FUNCTIONS = {
    "naive": naive_forecast,
    "moving_average": moving_average_forecast,
    "linear_trend": linear_trend_forecast,
}


def configure_academic_style() -> None:
    """Set a clean, high-contrast academic plotting style."""
    matplotlib.set_loglevel("warning")
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12.5,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, figures_dir: Path, figure_id: str) -> tuple[Path, Path]:
    """Save a figure as both PNG and PDF."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{figure_id}.png"
    pdf_path = figures_dir / f"{figure_id}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def baseline_mape_by_model_window(baseline: pd.DataFrame) -> plt.Figure:
    summary = (
        baseline.groupby(["cold_start_weeks", "model"], as_index=False)["MAPE"]
        .mean()
        .rename(columns={"MAPE": "mean_MAPE"})
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.lineplot(
        data=summary,
        x="cold_start_weeks",
        y="mean_MAPE",
        hue="model",
        marker="o",
        linewidth=2,
        ax=ax,
    )
    ax.set_title("Baseline MAPE by Model and Cold-Start Window")
    ax.set_xlabel("Cold-start window (weeks)")
    ax.set_ylabel("Mean MAPE")
    _legend_outside(ax, "Model")
    return fig


def augmentation_mape_by_ratio_window(augmented: pd.DataFrame) -> plt.Figure:
    summary = (
        augmented.groupby(["augmentation_ratio", "cold_start_weeks", "model"], as_index=False)["MAPE"]
        .mean()
        .rename(columns={"MAPE": "mean_MAPE"})
    )
    grid = sns.relplot(
        data=summary,
        x="augmentation_ratio",
        y="mean_MAPE",
        hue="cold_start_weeks",
        col="model",
        kind="line",
        marker="o",
        linewidth=2,
        height=4.2,
        aspect=0.95,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Augmentation ratio", "Mean MAPE")
    grid.set_titles("{col_name}")
    grid.legend.set_title("Cold-start weeks")
    grid.fig.suptitle("Augmented MAPE by Ensemble Size and Cold-Start Window", y=1.03, fontsize=13)
    return grid.fig


def mape_difference_heatmap(comparison: pd.DataFrame) -> plt.Figure:
    summary = (
        comparison.groupby(["model", "cold_start_weeks", "augmentation_ratio"], as_index=False)[
            "MAPE_difference"
        ]
        .mean()
    )
    summary["window_ratio"] = summary.apply(
        lambda row: f"{int(row['cold_start_weeks'])}w / {row['augmentation_ratio']:.1f}x",
        axis=1,
    )
    pivot = summary.pivot(index="model", columns="window_ratio", values="MAPE_difference")
    ordered_columns = [
        f"{weeks}w / {ratio:.1f}x"
        for weeks in (4, 8, 12)
        for ratio in AUGMENTATION_RATIOS
    ]
    pivot = pivot.reindex(columns=ordered_columns)
    max_abs = float(abs(pivot).max().max())

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Mean MAPE difference"},
        ax=ax,
    )
    ax.set_title(
        "Mean MAPE Difference: Augmented Minus Baseline\n"
        "Positive values indicate worse performance after augmentation."
    )
    ax.set_xlabel("Cold-start window / augmentation ratio")
    ax.set_ylabel("Model")
    return fig


def improvement_rate_by_model_ratio(diagnostic_matrix: pd.DataFrame) -> plt.Figure:
    summary = (
        diagnostic_matrix.groupby(["augmentation_ratio", "model"], as_index=False)[
            "improvement_rate"
        ]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.lineplot(
        data=summary,
        x="augmentation_ratio",
        y="improvement_rate",
        hue="model",
        marker="o",
        linewidth=2,
        ax=ax,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Improvement Rate by Model and Augmentation Ratio")
    ax.set_xlabel("Augmentation ratio")
    ax.set_ylabel("Improvement rate")
    ax.set_ylim(0, 1)
    _legend_outside(ax, "Model")
    return fig


def sector_level_mape_difference(comparison: pd.DataFrame) -> plt.Figure:
    summary = (
        comparison.groupby(["sector", "augmentation_ratio"], as_index=False)["MAPE_difference"]
        .mean()
        .sort_values(["sector", "augmentation_ratio"])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=summary,
        x="sector",
        y="MAPE_difference",
        hue="augmentation_ratio",
        palette="muted",
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(
        "Sector-Level MAPE Difference\n"
        "Positive values indicate worse performance after augmentation."
    )
    ax.set_xlabel("Sector")
    ax.set_ylabel("Mean MAPE difference")
    _legend_outside(ax, "Augmentation ratio")
    return fig


def baseline_vs_augmented_distribution(
    comparison: pd.DataFrame,
    zoomed: bool = False,
) -> plt.Figure:
    baseline_view = comparison[
        ["sample_id", "model", "augmentation_ratio", "baseline_MAPE"]
    ].rename(columns={"baseline_MAPE": "MAPE"})
    baseline_view["condition"] = "Baseline"
    augmented_view = comparison[
        ["sample_id", "model", "augmentation_ratio", "augmented_MAPE"]
    ].rename(columns={"augmented_MAPE": "MAPE"})
    augmented_view["condition"] = "Augmented"
    long_data = pd.concat([baseline_view, augmented_view], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(
        data=long_data,
        x="augmentation_ratio",
        y="MAPE",
        hue="condition",
        showfliers=False,
        palette=["#4c78a8", "#f58518"],
        ax=ax,
    )
    title = "Distributional Shift in MAPE After Statistical Augmentation"
    if zoomed:
        title = (
            "Distributional Shift in MAPE After Statistical Augmentation\n"
            "Display limited to 0-40 MAPE; extreme values remain in the analysis."
        )
        ax.set_ylim(0, 40)
    ax.set_title(title)
    ax.set_xlabel("Augmentation ratio")
    ax.set_ylabel("MAPE")
    _legend_outside(ax, "")
    return fig


def example_forecast_case(
    comparison_row: pd.Series,
    scenarios: pd.DataFrame,
    random_seed: int,
) -> plt.Figure:
    """Plot one aligned synthetic history with baseline and ensemble forecasts."""
    sample_id = comparison_row["sample_id"]
    model_name = comparison_row["model"]
    augmentation_ratio = float(comparison_row["augmentation_ratio"])
    scenario_index = int(scenarios.index[scenarios["sample_id"] == sample_id][0])
    ratio_index = AUGMENTATION_RATIOS.index(augmentation_ratio)
    scenario = scenarios.loc[scenario_index]

    train = pd.read_csv(scenario["train_path"])
    test = pd.read_csv(scenario["test_path"])
    train_series = train["y"].astype(float).tolist()
    test_series = test["y"].astype(float).tolist()
    forecast_horizon = len(test_series)
    n_synthetic_series = int(round(len(train_series) * augmentation_ratio))
    seed = random_seed + scenario_index * 100 + ratio_index
    synthetic_histories = generate_endpoint_preserving_histories(
        train_series,
        n_synthetic_series=n_synthetic_series,
        random_seed=seed,
    )
    baseline_pred = MODEL_FUNCTIONS[model_name](train_series, forecast_horizon)
    augmented_pred = augmented_ensemble_forecast(
        train_series,
        forecast_horizon,
        MODEL_FUNCTIONS[model_name],
        n_synthetic_series,
        seed,
    )

    train_x = list(range(len(train_series)))
    test_x = list(range(len(train_series), len(train_series) + forecast_horizon))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(train_x, train_series, label="Real training data", color="#4c78a8", linewidth=2)
    ax.plot(
        train_x,
        synthetic_histories[0],
        label="Example aligned synthetic history",
        color="#f58518",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(test_x, test_series, label="True test data", color="#54a24b", linewidth=2)
    ax.plot(test_x, baseline_pred, label="Baseline forecast", color="#4c78a8", linestyle=":")
    ax.plot(test_x, augmented_pred, label="Augmented forecast", color="#e45756", linestyle=":")
    ax.axvline(len(train_series) - 0.5, color="black", linewidth=1, alpha=0.7)
    ax.set_title(
        f"{sample_id}: {model_name}, ratio={augmentation_ratio:.1f}, "
        f"MAPE diff={comparison_row['MAPE_difference']:.2f}"
    )
    ax.set_xlabel("Relative trading-day index")
    ax.set_ylabel("Adjusted close")
    _legend_outside(ax, "")
    return fig


def _legend_outside(ax: plt.Axes, title: str) -> None:
    """Place legends outside the plotting area to avoid covering data."""
    ax.legend(
        title=title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )
