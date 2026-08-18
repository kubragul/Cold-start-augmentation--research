# Preliminary Findings

## 1. Baseline Performance Summary

The baseline experiment evaluated three simple forecasting models across 1,419 rolling cold-start samples. Across all baseline sample-model combinations, the average errors were:

- MAE: 9.4961
- RMSE: 10.8655
- MAPE: 6.4355

Baseline MAPE differed by model. The naive baseline had the lowest average MAPE at 5.3154, followed by moving average at 5.7000 and linear trend at 8.2912. These results suggest that, in this pilot, simple endpoint or local-average baselines were more competitive than extrapolating a linear trend.

## 2. Statistical Augmentation Performance Summary

The statistical continuation augmentation method substantially worsened average forecasting performance. Across all augmented sample-model-ratio combinations, the average errors were:

- MAE: 23.2809
- RMSE: 24.2442
- MAPE: 16.0382

Compared with the matched baseline results, augmentation increased average error by:

- MAE difference: +13.7848
- RMSE difference: +13.3787
- MAPE difference: +9.6027
- Mean percent change in MAPE: +220.7669%

The augmentation improved only 2,670 of 12,771 paired comparisons, an improvement rate of 20.91%. Performance degradation increased with the augmentation ratio. Average MAPE difference was +3.5151 for ratio 0.5, +7.7319 for ratio 1.0, and +17.5611 for ratio 2.0.

## 3. Significance Test Summary

Paired statistical tests were used because each cold-start sample was evaluated under both no augmentation and statistical augmentation. The Wilcoxon signed-rank test was treated as the primary test because forecasting errors may not be normally distributed.

For overall MAPE, the primary Wilcoxon signed-rank test found:

- Mean baseline MAPE: 6.4355
- Mean augmented MAPE: 16.0382
- Mean difference: +9.6027
- p-value: 0.0
- Interpretation: significantly_worsened

The paired t-test produced the same qualitative conclusion. Across the statistical test table, the observed interpretations were significantly worsened rather than significantly improved.

## 4. Diagnostic Analysis Summary

The diagnostic analysis indicates that the failure was broad rather than isolated. Statistical augmentation improved only 20.91% of paired comparisons. The 0.5 augmentation ratio was least harmful, while the 2.0 ratio was most harmful. Degradation increased monotonically as augmentation ratio increased.

By model, moving average was least harmed, while linear trend was most harmed. By sector, consumer stocks were least harmed, while technology stocks were most harmed. The diagnostic analysis also identified 1,660 outlier cases with `percent_change_MAPE >= 500%`, indicating a substantial right tail of severe degradation.

These findings are consistent with a plausible failure mechanism: appending synthetic continuation values after the real training window can distort the endpoint signal used by simple forecasting models. This is especially problematic for naive and moving-average baselines, which depend heavily on the most recent values, and for linear trend models, which may amplify artificial extrapolated trends.

## 5. Main Takeaway

The main finding is that this specific statistical continuation augmentation method significantly worsened forecasting performance in the finance cold-start pilot. The result should not be interpreted as evidence that synthetic augmentation never works. Rather, it shows that a simple trend-plus-Gaussian-noise continuation method can harm forecasting when synthetic points are appended as if they were real post-training observations.

This is a useful negative result. It clarifies that increasing the apparent amount of training data is not sufficient; generated observations must preserve the predictive structure relevant to the forecast horizon.

## 6. Limitations

This is a preliminary pilot, not a final claim about all financial forecasting augmentation methods. The experiment used a limited set of large-cap U.S. equities across four sectors and a fixed 2020-2024 date range. The baseline models were intentionally simple. The augmentation method operated on price levels rather than returns and used a linear trend plus Gaussian residual noise, which may be too restrictive for financial data.

The diagnostic sector findings should also be treated cautiously. Technology stocks were most harmed and consumer stocks least harmed in this pilot, but this should be interpreted as an observed pattern requiring further testing rather than a causal sector-level conclusion.

## 7. Recommended Next Experiment

The next experiment should avoid appending long synthetic continuation blocks that replace the real endpoint signal. A natural next step is residual or bootstrap augmentation, where variation is generated around the observed training sequence without allowing synthetic points to dominate the final training values.

Return-based modeling should also be tested. Generating synthetic returns rather than price levels may better reflect financial time-series structure and reduce artificial level shifts.

After these simpler alternatives are evaluated, more complex methods such as TimeGAN or domain-aware/LLM-assisted augmentation could be explored. Those methods should be introduced only with careful diagnostics showing that generated sequences preserve relevant financial dynamics and do not simply increase data volume.
