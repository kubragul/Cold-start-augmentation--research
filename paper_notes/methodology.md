# Methodology

## Research question

Can endpoint-preserving statistical augmentation improve 28-trading-day
forecasts when only 4, 8, or 12 weeks of target-series history are available?

## Data and scenarios

The study uses adjusted-close prices for 11 large-cap U.S. equities from 2020
through 2024. Rolling windows create 1,419 paired train/test scenarios. The test
set is never passed to the generator or forecasting models.

## Endpoint-preserving augmentation

A least-squares trend is fitted to each real training history. Training
residuals are sampled with replacement to create alternative histories over
the same indices. A linear correction anchors each synthetic history to the
real first and last prices. Consequently, no synthetic observation lies beyond
the forecast origin.

For augmentation ratios 0.5, 1.0, and 2.0, the number of synthetic histories is
the training length multiplied by the ratio. Naive, five-point moving-average,
and linear-trend forecasts are produced independently for the real history and
each synthetic history, then averaged.

## Evaluation

MAE, RMSE, and MAPE are compared pairwise against unaugmented forecasts for the
same sample and model. Wilcoxon signed-rank is the primary test; paired t-tests
are reported alongside. Benjamini-Hochberg correction controls FDR across
grouped tests. Random seed 42 makes the pipeline deterministic.
