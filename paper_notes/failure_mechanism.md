# Failure Mechanism: Statistical Continuation Augmentation

## Summary of Empirical Finding

The statistical continuation augmentation method worsened forecasting performance in this finance pilot. Across the paired comparisons, augmentation improved only 2,670 of 12,771 sample-model-ratio comparisons, corresponding to an improvement rate of 20.91%. The 0.5 augmentation ratio was the least harmful setting, while the 2.0 ratio was the most harmful. Degradation increased as the augmentation ratio increased, suggesting that adding larger blocks of synthetic continuation points progressively distorted the training signal.

This result was not limited to a single model, sector, or cold-start length. Among the forecasting models, moving average was the least harmed, while linear trend was the most harmed. By sector, consumer stocks were least harmed and technology stocks were most harmed. The diagnostic analysis also found 1,660 outlier cases with `percent_change_MAPE >= 500%`, indicating a substantial right tail of severe degradation. The paired statistical tests found significantly worsened results, including for the primary Wilcoxon signed-rank test on MAPE.

## 1. Distortion From Appending Synthetic Continuation Points

The augmentation strategy generated synthetic observations by estimating a linear trend from the short training window, estimating residual variation, and then appending synthetic continuation points after the real observations. Although this design is simple and interpretable, it changes the effective training distribution seen by the forecasting models.

In the original cold-start setting, each model is trained on a short sequence of real observed prices. After augmentation, however, the final portion of the training sequence no longer consists of market observations. It consists of model-generated values that extrapolate a fitted trend and add Gaussian noise. This means the models are no longer fitting only the empirical cold-start window; they are fitting a mixture of real prices and synthetic continuation values. If the extrapolated continuation deviates from the actual next 28 trading days, the augmented training sequence can move the model away from the true local market path.

The diagnostic pattern supports this explanation. The 0.5 ratio was least harmful, and the 2.0 ratio was most harmful. If synthetic points were consistently informative, larger augmentation blocks might be expected to help or at least stabilize performance. Instead, the monotonic increase in degradation suggests that the synthetic tail increasingly dominated the short real training window.

## 2. Sensitivity of Naive and Moving Average Baselines

Naive and moving average models are especially sensitive to endpoint behavior. The naive model predicts the last observed training value for every forecast step. Once synthetic observations are appended, the "last observed value" is no longer a real market price; it is the final synthetic continuation point. Therefore, even a modest synthetic drift can directly determine the entire forecast.

The moving average model is also affected, although less extremely. It predicts using the mean of the most recent observations. When synthetic points are appended to the end of the training sequence, the moving average window can become dominated by synthetic values rather than real prices. This explains why appended continuation values can harm moving average forecasts even when the original real training window contained useful local information.

In the diagnostics, moving average was the least harmed model on average, but it was still harmed overall. This is consistent with partial smoothing: moving average is less brittle than naive because it averages recent values, but it remains vulnerable when the recent values are synthetic.

## 3. Linear Trend Amplification

The linear trend baseline was the most harmed model. This is plausible because the augmentation method itself creates a trend-based synthetic continuation, and the linear trend model then fits another trend to the augmented sequence. In effect, the method can compound trend extrapolation.

If the synthetic continuation extends an upward or downward fitted slope beyond what the real market subsequently follows, the augmented sequence may create an artificial trend. The linear trend forecaster can then amplify this artificial trajectory when projecting into the test horizon. This mechanism is especially risky in financial price series, where short-term price trends often reverse, flatten, or shift abruptly.

The diagnostic finding that linear trend was most harmed is therefore consistent with a trend amplification failure mode.

## 4. Larger Synthetic Blocks Increase Distortion

The augmentation ratios show a clear dose-response pattern. The 0.5 ratio was least harmful, while the 2.0 ratio was most harmful. This means that generating more synthetic continuation points did not simply add more training information; it increased the weight of generated data relative to observed data.

For a 20-observation cold-start window, the 2.0 ratio appends 40 synthetic points, making the synthetic portion twice as large as the real training sequence. For a model that treats the sequence as observed history, this can substantially alter the estimated level, local average, or trend. The more synthetic continuation values are appended, the farther the effective training endpoint can move away from the real endpoint.

This pattern suggests that, in this pilot, augmentation ratio controls the magnitude of distributional distortion. Larger synthetic blocks create stronger artificial histories and therefore larger forecast errors.

## 5. Limits of Trend Plus Gaussian Noise for Financial Prices

Financial price series are difficult to augment with a simple trend-plus-Gaussian-noise model. Prices are nonstationary, heteroskedastic, and affected by market-wide shocks, firm-specific events, changing volatility regimes, and sector-level dynamics. A short linear trend estimated from a cold-start window may not represent the next month of market behavior.

Gaussian residual noise is also a limited approximation. Financial returns often exhibit heavy tails, volatility clustering, jumps, and asymmetric behavior. When augmentation is performed on price levels rather than returns, synthetic continuation can introduce unrealistic levels or trends. This is particularly problematic when the generated values are appended as if they were observed prices.

The technology sector being most harmed may reflect greater volatility and stronger regime shifts during the study period, although this should be treated as a hypothesis rather than a final causal claim. The consumer sector being least harmed may indicate that more stable price paths are less vulnerable to this augmentation failure mode, but this also requires further analysis.

## 6. Academic Value of the Negative Result

This negative result is academically valuable. It shows that simple synthetic augmentation is not automatically beneficial in cold-start financial forecasting. In this pilot, the statistical continuation method significantly worsened performance, and the result was broad rather than isolated.

The finding helps clarify an important methodological point: augmentation should not be evaluated only by whether it increases the amount of training data. It must be evaluated by whether the generated data preserve the predictive structure relevant to the forecast horizon. More synthetic observations can degrade performance if they alter the local signal that baseline models rely on.

The result also provides a useful baseline failure case. It identifies a mechanism by which augmentation can harm forecasting: generated continuation values can dominate short real histories and shift endpoint-dependent forecasts away from observed market behavior.

## 7. Implications for the Next Method

The next stage should test augmentation methods that reduce endpoint distortion and better respect financial time-series structure.

Residual or bootstrap augmentation is a natural next step. Instead of appending a long synthetic continuation, bootstrap methods can resample residuals or local changes in a way that preserves the observed endpoint and produces alternative plausible paths without replacing the final training values.

Return-based modeling is also important. Generating synthetic returns rather than price levels may better reflect financial dynamics, because returns are often closer to stationary than raw prices. A return-based approach could then reconstruct price paths from the real endpoint, reducing artificial level shifts.

TimeGAN or related sequence models may eventually be useful, but they should be treated carefully. Such models require enough data, careful validation, and explicit checks for whether generated sequences preserve relevant temporal and distributional properties. The present result suggests that more complex generation should be justified by diagnostics rather than assumed to help.

LLM or domain-aware augmentation may also be explored later, especially if it incorporates market context, sector information, or event-aware constraints. However, this should not be introduced until the simpler statistical and return-based baselines are better understood.

Overall, the conclusion is not that synthetic augmentation never works. The more precise conclusion is that this specific statistical continuation augmentation method worsened forecasting performance in this finance cold-start pilot. Future methods should avoid letting synthetic continuation points overwrite or dominate the real endpoint information that simple forecasting baselines depend on.
