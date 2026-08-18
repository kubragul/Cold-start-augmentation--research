# Limitations

- The study covers 11 U.S. equities and 2020–2024; results may not generalize to
  other assets, markets, or regimes.
- Augmentation operates on price levels with a linear-trend residual bootstrap.
  It does not model heavy tails, volatility clustering, jumps, or cross-asset
  dependence explicitly.
- Naive forecasts cannot change because all synthetic histories deliberately
  preserve the real endpoint. This is an alignment invariant, not evidence of
  augmentation benefit for that model.
- Synthetic-history forecasts are correlated because they originate from the
  same short real window. Paired inference is performed across scenarios, but
  overlapping market periods may still introduce dependence.
- Hyperparameters were not selected on a separate validation period. The three
  augmentation ratios should be treated as prespecified sensitivity settings,
  not optimized choices.
- Statistical significance does not establish economic value; transaction
  costs, trading decisions, and risk-adjusted performance are outside scope.
