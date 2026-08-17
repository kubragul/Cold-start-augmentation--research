# Synthetic Time Series Augmentation for Cold-Start Trend Forecasting: A Finance Pilot Study

## Abstract

Cold-start forecasting is difficult because models must make predictions from very short historical sequences. Synthetic time-series augmentation is an attractive response to this limitation because it appears to increase the amount of training data available for each target series. This pilot study evaluates whether a simple statistical continuation augmentation method improves cold-start forecasting accuracy for financial time series. Using daily adjusted close prices from 11 large-cap U.S. equities across four sectors from 2020 to 2024, we created 1,419 rolling cold-start samples with 4-, 8-, and 12-week training windows and a 28-trading-day forecast horizon. Statistical continuation augmentation estimated a linear trend from the training window, added Gaussian residual noise, and appended synthetic continuation points to the observed training sequence. Contrary to the motivating hypothesis, this augmentation significantly worsened forecasting performance. Overall MAPE increased from 6.4355 under baseline forecasting to 16.0382 after augmentation, with a mean paired MAPE difference of +9.6027 points. The primary Wilcoxon signed-rank test found significantly worsened performance. The result suggests that synthetic data quantity alone is insufficient: generated observations must preserve the predictive structure of the original time series and should not distort endpoint-dependent forecasting signals.

## 1. Introduction

Cold-start forecasting arises when a target time series has limited observed history but still requires short-horizon prediction. This setting is common in practical forecasting applications, including newly listed assets, newly launched products, sparse retail items, and recently observed operational series. The core difficulty is that many forecasting methods rely on historical patterns, but cold-start conditions provide too little target-specific history to estimate those patterns reliably.

Synthetic time-series augmentation offers one possible solution. If plausible additional observations can be generated from the short training window, downstream forecasting models may have more information from which to estimate levels, trends, and short-term variation. This idea is especially appealing in academic experiments because augmentation can be evaluated systematically across repeated cold-start samples.

However, augmentation should not be assumed helpful simply because it increases apparent data volume. Generated observations may introduce artificial trends, distort local levels, or shift the endpoint of the training sequence away from the true observed market path. For forecasting, the relevant question is not whether augmentation creates more data, but whether the generated data preserve the predictive structure needed for the forecast horizon. This pilot study therefore treats augmentation as an empirical question.

## 2. Research Question

The main research question is:

**Does statistical synthetic augmentation improve forecasting accuracy in finance cold-start time series?**

The pilot hypothesis is deliberately cautious: statistical continuation augmentation may help when limited history is available, but its effectiveness depends on whether synthetic points preserve the predictive structure of the series. In particular, appending synthetic continuation points may help if they approximate the near-future trajectory, but may harm if they distort the training endpoint or amplify artificial trend behavior.

## 3. Data

The study uses daily adjusted close prices downloaded from `yfinance`. The pilot dataset covers 11 large-cap U.S. equities across four sectors:

- Technology: AAPL, MSFT, NVDA
- Finance: JPM, BAC, GS
- Consumer: WMT, COST, TGT
- Energy: XOM, CVX

The date range is 2020-01-01 through 2024-12-31. Prices were cleaned into a long-format dataset with date, ticker, sector, and adjusted close target value. Cold-start samples were created with rolling windows rather than using only the first observations of each series. This design reduces dependence on a single historical period and evaluates each method across many market conditions.

The cold-start training windows were 4, 8, and 12 weeks, corresponding to 20, 40, and 60 trading days. Each sample used the next 28 trading days as the forecast horizon. A rolling step of 28 trading days produced 1,419 cold-start samples.

## 4. Methodology

### Rolling Cold-Start Simulation

For each ticker and cold-start window length, the experiment sorted observations by date and generated rolling train-test samples. The training window contained only historical observations available at the forecast origin. The test window contained the following 28 trading days and was used only for evaluation.

### Baseline Forecasting Models

Three simple baseline models were evaluated:

- **Naive:** predicts the last observed training value for all future steps.
- **Moving average:** predicts the mean of the last five training observations.
- **Linear trend:** fits a least-squares linear trend over the training index and extrapolates the next 28 points.

These models were intentionally simple. Their purpose was to establish transparent baseline behavior before evaluating more complex augmentation methods.

### Statistical Continuation Augmentation

The statistical continuation augmentation method used only the training window. For each cold-start sample, it:

1. estimated a linear trend from the observed training values;
2. calculated residuals between observed values and fitted trend values;
3. estimated residual standard deviation;
4. generated synthetic continuation points by extending the fitted trend;
5. added Gaussian noise using the estimated residual standard deviation;
6. appended the synthetic continuation points after the real training window.

Three augmentation ratios were evaluated:

- 0.5x the training length
- 1.0x the training length
- 2.0x the training length

Thus, a 20-observation training window received 10, 20, or 40 synthetic points; a 40-observation window received 20, 40, or 80 synthetic points; and a 60-observation window received 30, 60, or 120 synthetic points.

### Evaluation and Statistical Testing

Forecasts were evaluated with MAE, RMSE, and MAPE. MAE measures average absolute error, RMSE penalizes larger errors more heavily, and MAPE provides scale-independent interpretability when true values are safely away from zero.

The comparison used a paired design. Each sample-model combination was evaluated under no augmentation and under statistical augmentation for each ratio. The Wilcoxon signed-rank test was treated as the primary statistical test because forecasting errors may not be normally distributed. A paired t-test was also reported as a supplementary test. Because multiple grouped tests were run, p-values were additionally corrected using Benjamini-Hochberg false-discovery-rate (FDR) adjustment and both raw and adjusted values were reported.

## 5. Results

Across all baseline sample-model combinations, the average errors were:

- MAE: 9.4961
- RMSE: 10.8655
- MAPE: 6.4355

After statistical continuation augmentation, the average errors were:

- MAE: 23.2809
- RMSE: 24.2442
- MAPE: 16.0382

In the matched paired comparison, augmentation increased average error:

- MAE difference: +13.7848
- RMSE difference: +13.3787
- MAPE difference: +9.6027

Only 2,670 of 12,771 paired comparisons improved, corresponding to an improvement rate of 20.91%. The primary Wilcoxon signed-rank test for overall MAPE returned the interpretation `significantly_worsened`. The least harmful augmentation ratio was 0.5, while the most harmful ratio was 2.0. Diagnostic analysis identified 1,660 outlier cases with `percent_change_MAPE >= 500%`.

The mean paired MAPE difference was +9.6027 points. When percent change is discussed, two quantities should be distinguished. The row-level mean percent change in MAPE was +220.7669%, while the increase relative to the overall mean baseline MAPE was approximately +149.21%. The MAPE-point difference is the more direct summary of the paired error increase.

![Baseline MAPE by model and cold-start window](../results/figures_revised/baseline_mape_by_model_window.png)

Figure 1. Mean baseline MAPE across rolling cold-start samples, grouped by model and training-window length. This figure compares baseline difficulty across models and cold-start windows before augmentation.

![Augmented MAPE by augmentation ratio and cold-start window](../results/figures_revised/augmentation_mape_by_ratio_window.png)

Figure 2. Mean MAPE under statistical continuation augmentation. Larger augmentation ratios generally produce higher errors. This figure shows that larger synthetic continuation blocks worsen performance.

![Mean MAPE difference heatmap](../results/figures_revised/mape_difference_heatmap.png)

Figure 3. Mean MAPE difference, computed as augmented MAPE minus baseline MAPE. Positive values indicate worse augmented performance. This figure identifies where statistical continuation augmentation helps or hurts.

![Improvement rate by model and augmentation ratio](../results/figures_revised/improvement_rate_by_model_ratio.png)

Figure 4. Fraction of paired comparisons in which augmented MAPE was lower than baseline MAPE. This figure shows that augmentation improves only a minority of comparisons.

![Sector-level MAPE difference](../results/figures_revised/sector_level_mape_difference.png)

Figure 5. Mean MAPE difference by sector and augmentation ratio. Positive values indicate degradation under augmentation. This figure compares sector-level degradation while keeping augmentation ratio separate.

![Distribution of baseline and augmented MAPE](../results/figures_revised/baseline_vs_augmented_distribution.png)

Figure 6. Boxplots compare the distribution of baseline and augmented MAPE by augmentation ratio, excluding plotted outliers for readability. This figure shows distributional shift rather than only average change.

![Distribution of baseline and augmented MAPE, zoomed](../results/figures_revised/baseline_vs_augmented_distribution_zoomed.png)

Figure 7. Boxplots compare baseline and augmented MAPE with the display limited to 0-40 MAPE. Extreme values are excluded only visually, not analytically. This figure shows the central distribution while preserving the negative-result interpretation and retaining outliers in analysis tables.

## 6. Diagnostic Analysis

The diagnostic analysis helps explain why the aggregate result was negative. Statistical continuation augmentation improved only 20.91% of paired comparisons, so most comparisons worsened. The degradation increased with augmentation ratio: the 0.5 ratio was least harmful and the 2.0 ratio was most harmful. This dose-response pattern suggests that larger synthetic continuation blocks increasingly dominated or distorted the short real training histories.

By model, moving average was least harmed, while linear trend was most harmed. By sector, consumer stocks were least harmed, while technology stocks were most harmed. These sector-level patterns are descriptive rather than causal, but they suggest that more volatile or regime-sensitive series may be more vulnerable to simple continuation-based augmentation.

The analysis also found 1,660 severe outlier cases with `percent_change_MAPE >= 500%`. These outliers indicate that the method did not merely produce small average degradation; in some samples, synthetic continuation created very large forecast error increases.

![Representative forecast failure case](../results/figures_revised/example_forecast_failure_case.png)

Figure 8. Example where statistical continuation augmentation strongly increased MAPE. Synthetic continuation values visibly shift the forecast away from the true test path. This figure visually explains how continuation augmentation can distort forecasts.

![Representative forecast improvement case](../results/figures_revised/example_forecast_improvement_case.png)

Figure 9. Example where statistical continuation augmentation reduced MAPE. Such cases occur, but they are a minority in this pilot. This figure shows that improvement occurs in some cases but is not dominant.

## 7. Failure Mechanism

The likely failure mechanism is endpoint distortion. The augmentation method appended synthetic continuation points after the real training window. As a result, the final part of the training sequence no longer consisted of observed market prices; it consisted of extrapolated trend values plus Gaussian residual noise. Models trained on this augmented sequence therefore saw a mixture of real data and synthetic continuation values.

Naive and moving-average models are sensitive to this change because they depend heavily on endpoint behavior. The naive model predicts the last training value for every future step. Once synthetic points are appended, that final value is synthetic rather than observed. The moving-average model is also affected because its recent-value window can become dominated by synthetic continuation points. Even if the original real training window contained useful local information, appended synthetic values can overwrite that endpoint signal.

Linear trend can be harmed through trend amplification. The augmentation method first creates a trend-based synthetic continuation, and the linear trend baseline then fits a trend to the augmented sequence. If the synthetic continuation extends an artificial slope, the forecasting model can compound that artificial trend when projecting into the test horizon. This is consistent with the diagnostic finding that linear trend was the most harmed model.

Financial price series may be particularly difficult for simple trend-plus-Gaussian-noise augmentation. Prices are nonstationary and affected by volatility clustering, shocks, sector movements, and firm-specific events. Gaussian residual noise around a short linear trend is a limited approximation, especially when augmentation is performed on price levels rather than returns. The result should not be generalized to all synthetic augmentation methods. It is evidence that this specific statistical continuation method worsened performance in this finance cold-start pilot.

## 8. Limitations

This study is a finance-only pilot. It uses 11 large-cap U.S. equities, four sectors, and a fixed 2020-2024 period. The findings may not generalize to other asset classes, smaller firms, different market regimes, or non-financial domains.

The forecasting models are deliberately simple baselines. This is appropriate for an initial pilot, but the results do not establish how stronger forecasting models would respond to augmentation. The augmentation method also operates on raw price levels rather than returns, which may be poorly matched to financial time-series structure.

Rolling scenarios were created with a fixed step size, so neighboring samples overlap in time. This overlap improves coverage across market regimes but weakens strict independence assumptions in classical significance tests. Therefore, inferential claims should be interpreted with this dependency structure in mind, and future work should include non-overlapping or block-bootstrap robustness checks.

The study does not yet include the Walmart M5 retail dataset, TimeGAN, or LLM/domain-aware augmentation. Sector-level findings should be read descriptively rather than causally. Technology stocks were most harmed and consumer stocks least harmed in this pilot, but additional experiments would be required before making sector-level causal claims.

## 9. Recommended Next Experiment

The next experiment should test augmentation methods that avoid allowing synthetic continuation points to dominate the real endpoint.

First, return-based augmentation should be evaluated. Generating synthetic returns rather than price levels may better respect the statistical structure of financial time series and reduce artificial level shifts.

Second, residual or bootstrap augmentation should be tested. Instead of appending long extrapolated continuations, bootstrap methods can resample residuals or local changes while preserving the observed endpoint and generating plausible variation around the real training path.

Only after these simpler alternatives are evaluated should more complex methods such as TimeGAN or LLM/domain-aware augmentation be introduced. Those methods may be useful, but they should be justified by diagnostics showing that generated sequences preserve relevant temporal and domain structure rather than simply increasing data volume.

## 10. Conclusion

This specific statistical continuation augmentation method significantly worsened forecasting performance in the finance cold-start pilot. Baseline MAPE averaged 6.4355, while augmented MAPE averaged 16.0382, and the primary Wilcoxon signed-rank test classified the overall MAPE result as significantly worsened.

This is a useful negative finding. It shows that synthetic data quantity alone is not enough. Generated observations must preserve the predictive structure of the original time series. In cold-start forecasting, more synthetic continuation data can degrade performance when it distorts endpoint-dependent signals. The conclusion is specific to this continuation-based method: augmentation design must be validated carefully before being treated as beneficial.
