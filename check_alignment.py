"""Diagnostic: how much of the reported degradation is a time-index artifact?

Reproduces the pipeline on a subsample and compares, for linear_trend:
  A) as-implemented: fit on augmented series, forecast indices n+n_syn .. n+n_syn+h-1
  B) time-aligned:   fit on augmented series, forecast indices n .. n+h-1  (the
     indices the test set actually occupies)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.augmentation.statistical_augmentation import generate_statistical_synthetic_series
from src.evaluation.metrics import mean_absolute_percentage_error
from src.preprocessing.create_cold_start_scenarios import create_rolling_cold_start_scenarios

RATIOS = (0.5, 1.0, 2.0)
H = 28


def fit_line(vals):
    n = len(vals)
    xm = (n - 1) / 2
    ym = sum(vals) / n
    num = sum((i - xm) * (v - ym) for i, v in enumerate(vals))
    den = sum((i - xm) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    return slope, ym - slope * xm


data = pd.read_csv("data/processed/finance_adjusted_close_long.csv")
meta, samples = create_rolling_cold_start_scenarios(data, [4, 8, 12], H, 28)
print(f"scenarios: {len(meta)}")

rows = []
for idx, sid in enumerate(meta["sample_id"]):
    tr, te = samples[sid]
    x = tr["y"].tolist()
    y_true = te["y"].tolist()
    n = len(x)

    s_b, i_b = fit_line(x)
    base = [i_b + s_b * t for t in range(n, n + H)]

    for r_i, ratio in enumerate(RATIOS):
        n_syn = int(round(n * ratio))
        syn = generate_statistical_synthetic_series(x, n_syn, 42 + idx * 100 + r_i)
        aug = x + syn
        s_a, i_a = fit_line(aug)
        as_is = [i_a + s_a * t for t in range(len(aug), len(aug) + H)]
        aligned = [i_a + s_a * t for t in range(n, n + H)]
        rows.append({
            "ratio": ratio,
            "baseline": mean_absolute_percentage_error(y_true, base),
            "as_implemented": mean_absolute_percentage_error(y_true, as_is),
            "time_aligned": mean_absolute_percentage_error(y_true, aligned),
        })

df = pd.DataFrame(rows)
print("\nlinear_trend, mean MAPE:")
print(df.groupby("ratio")[["baseline", "as_implemented", "time_aligned"]].mean().round(3))
print("\nshare of comparisons where augmentation beats baseline:")
print(pd.DataFrame({
    "as_implemented": df.groupby("ratio").apply(
        lambda g: (g.as_implemented < g.baseline).mean(), include_groups=False),
    "time_aligned": df.groupby("ratio").apply(
        lambda g: (g.time_aligned < g.baseline).mean(), include_groups=False),
}).round(4))
