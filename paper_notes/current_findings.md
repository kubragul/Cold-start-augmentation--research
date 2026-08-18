# Current Findings After Alignment Correction

The endpoint-preserving experiment contains 12,771 paired
sample-model-ratio comparisons. Mean baseline MAPE is 6.4355 and mean augmented
MAPE is 5.8162, a mean reduction of 0.6193 MAPE points (9.62%).

The primary Wilcoxon signed-rank test reports `significantly_improved` overall
(p = 2.34e-118; BH-FDR-adjusted p = 9.83e-117). The paired t-test gives the same
qualitative conclusion.

By model, linear trend improves from 8.2912 to 6.7405 mean MAPE and moving
average improves from 5.7000 to 5.3928. Naive remains 5.3154 because every
synthetic history shares the observed endpoint. All three augmentation ratios
show significant mean improvement, with mean augmented MAPE of 5.8246, 5.8133,
and 5.8108 for ratios 0.5, 1.0, and 2.0 respectively.

Only 39.90% of individual comparisons improve even though the mean effect is
beneficial. The conclusion is therefore about average paired error, not a claim
that augmentation helps every stock, date, or model.
