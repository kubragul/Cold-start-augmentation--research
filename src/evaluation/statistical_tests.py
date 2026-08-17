"""Paired statistical tests for forecast error comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy import stats

SIGNIFICANCE_LEVEL = 0.05


def run_paired_error_tests(
    baseline_errors: Sequence[float],
    augmented_errors: Sequence[float],
    metric: str,
    grouping_variable: str,
    group_value: object,
) -> list[dict]:
    """Run Wilcoxon signed-rank and paired t-test on matched forecast errors.

    The paired design is essential here because each row compares the same
    cold-start sample under no augmentation and statistical augmentation.
    Wilcoxon is treated as the primary test because error distributions in
    forecasting studies are often skewed or heavy-tailed.
    """
    baseline = _validate_numeric_sequence(baseline_errors, "baseline_errors")
    augmented = _validate_numeric_sequence(augmented_errors, "augmented_errors")
    if len(baseline) != len(augmented):
        raise ValueError("baseline_errors and augmented_errors must have the same length")
    if not baseline:
        raise ValueError("paired tests require at least one pair")

    differences = [aug - base for base, aug in zip(baseline, augmented)]
    mean_baseline_error = sum(baseline) / len(baseline)
    mean_augmented_error = sum(augmented) / len(augmented)
    mean_difference = sum(differences) / len(differences)
    percent_change = _percent_change(mean_baseline_error, mean_difference)

    return [
        _format_result(
            metric=metric,
            grouping_variable=grouping_variable,
            group_value=group_value,
            test_name="wilcoxon_signed_rank",
            statistic_p_value=_wilcoxon_signed_rank(baseline, augmented),
            mean_baseline_error=mean_baseline_error,
            mean_augmented_error=mean_augmented_error,
            mean_difference=mean_difference,
            percent_change=percent_change,
            n_pairs=len(baseline),
        ),
        _format_result(
            metric=metric,
            grouping_variable=grouping_variable,
            group_value=group_value,
            test_name="paired_t_test",
            statistic_p_value=_paired_t_test(baseline, augmented),
            mean_baseline_error=mean_baseline_error,
            mean_augmented_error=mean_augmented_error,
            mean_difference=mean_difference,
            percent_change=percent_change,
            n_pairs=len(baseline),
        ),
    ]


def benjamini_hochberg_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Benjamini-Hochberg FDR-adjusted p-values."""
    values = _validate_numeric_sequence(p_values, "p_values")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("p_values must be between 0 and 1")
    if not values:
        return []

    m = len(values)
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    adjusted_sorted = [0.0] * m
    running_min = 1.0

    for rank in range(m, 0, -1):
        original_index, p_value = indexed[rank - 1]
        adjusted = (p_value * m) / rank
        running_min = min(running_min, adjusted)
        adjusted_sorted[rank - 1] = min(1.0, running_min)

    adjusted_by_original_order = [0.0] * m
    for sorted_position, (original_index, _) in enumerate(indexed):
        adjusted_by_original_order[original_index] = adjusted_sorted[sorted_position]
    return adjusted_by_original_order


def interpret_test(mean_difference: float, p_value: float) -> str:
    """Interpret significance without hiding negative results."""
    if math.isnan(p_value):
        return "no_significant_difference"
    if mean_difference < 0 and p_value < SIGNIFICANCE_LEVEL:
        return "significantly_improved"
    if mean_difference > 0 and p_value < SIGNIFICANCE_LEVEL:
        return "significantly_worsened"
    return "no_significant_difference"


def interpret_test_with_adjusted_p(mean_difference: float, adjusted_p_value: float) -> str:
    """Interpret significance after multiple-testing correction."""
    return interpret_test(mean_difference, adjusted_p_value)


def _format_result(
    metric: str,
    grouping_variable: str,
    group_value: object,
    test_name: str,
    statistic_p_value: tuple[float, float],
    mean_baseline_error: float,
    mean_augmented_error: float,
    mean_difference: float,
    percent_change: float,
    n_pairs: int,
) -> dict:
    statistic, p_value = statistic_p_value
    return {
        "metric": metric,
        "grouping_variable": grouping_variable,
        "group_value": group_value,
        "test_name": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "mean_baseline_error": mean_baseline_error,
        "mean_augmented_error": mean_augmented_error,
        "mean_difference": mean_difference,
        "percent_change": percent_change,
        "n_pairs": n_pairs,
        "interpretation": interpret_test(mean_difference, p_value),
    }


def _wilcoxon_signed_rank(
    baseline_errors: list[float],
    augmented_errors: list[float],
) -> tuple[float, float]:
    differences = [aug - base for base, aug in zip(baseline_errors, augmented_errors)]
    if all(diff == 0 for diff in differences):
        return 0.0, 1.0
    result = stats.wilcoxon(
        augmented_errors,
        baseline_errors,
        zero_method="wilcox",
        alternative="two-sided",
    )
    return float(result.statistic), float(result.pvalue)


def _paired_t_test(
    baseline_errors: list[float],
    augmented_errors: list[float],
) -> tuple[float, float]:
    differences = [aug - base for base, aug in zip(baseline_errors, augmented_errors)]
    if len(differences) < 2 or all(diff == 0 for diff in differences):
        return 0.0, 1.0
    result = stats.ttest_rel(augmented_errors, baseline_errors)
    return float(result.statistic), float(result.pvalue)


def _validate_numeric_sequence(values: Sequence[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numeric values, not a string")
    try:
        numeric_values = [float(value) for value in values]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable sequence of numeric values") from exc
    except ValueError as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc

    for value in numeric_values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite values")
    return numeric_values


def _percent_change(baseline_error: float, mean_difference: float) -> float:
    if baseline_error == 0:
        raise ValueError("cannot compute percent_change with zero mean baseline error")
    return mean_difference / baseline_error * 100
