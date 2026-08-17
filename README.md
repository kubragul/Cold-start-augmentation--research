# Cold-Start Augmentation Research

A reproducible pilot study asking a narrow question: **does simple statistical
augmentation improve forecasting accuracy when only a few weeks of history are
available?**

Using daily adjusted-close prices for 11 large-cap U.S. equities (2020–2024),
the pipeline builds 1,419 rolling cold-start samples (4/8/12-week training
windows, 28-trading-day horizon), forecasts each one with three transparent
baselines, repeats the exercise with synthetic continuation points appended to
the training window at three ratios, and compares the two with paired
statistical tests.

This repository is organized as an academic experiment rather than a software
product. The numbered scripts in `experiments/` define the execution order;
`src/` holds the reusable logic they call.

## Status

The pipeline is fully reproducible end to end: a clean run regenerates every
published number exactly. The **methodology is still under revision** — see
[Known issue: forecast-window alignment](#known-issue-forecast-window-alignment)
before citing any result from this repository.

## Installation

```bash
git clone https://github.com/<your-username>/cold-start-augmentation-research.git
cd cold-start-augmentation-research

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer is required (the code uses `X | Y` type syntax).

## Reproducing the experiment

```bash
./run_all.sh
```

Or step by step:

```bash
python experiments/01_download_data.py              # yfinance -> data/raw, data/processed
python experiments/02_create_cold_start_scenarios.py # rolling train/test samples
python experiments/03_run_baseline_experiments.py    # naive, moving average, linear trend
python experiments/04_run_augmentation_experiments.py # + synthetic continuation, 3 ratios
python experiments/05_compare_results.py             # paired baseline vs augmented table
python experiments/06_run_statistical_tests.py       # Wilcoxon + paired t, BH-FDR corrected
python experiments/07_diagnostic_analysis.py         # where and how it fails
python experiments/08_generate_paper_figures.py      # paper-ready figures
```

Step 01 requires network access. Steps 02–08 are offline and deterministic
(`random_seed: 42` in `config.yaml`).

All experiment parameters — tickers, date range, window lengths, horizon,
augmentation ratios, metrics, output paths — live in `config.yaml`. Nothing is
hard-coded in the scripts.

## Repository layout

```
config.yaml                  all experiment parameters
experiments/                 numbered pipeline steps (entry points)
src/
  data/                      yfinance download + cleaning
  preprocessing/             rolling cold-start scenario construction
  augmentation/              statistical continuation augmentation
  models/                    naive / moving average / linear trend baselines
  evaluation/                MAE, RMSE, MAPE + paired statistical tests
  visualization/             paper figures
data/raw/                    immutable source data          (git-ignored)
data/interim/                caches                          (git-ignored)
data/processed/              cleaned data + per-sample CSVs  (git-ignored)
results/tables/              result tables
results/figures/             paper figures
results/statistical_tests/   significance test output
paper_notes/                 methodology, findings, limitations, manuscript
```

Regenerable artifacts are git-ignored: the repository ships the code and the
small summary tables, and `./run_all.sh` rebuilds the rest (~70 MB) from
scratch.

## Method

**Cold-start samples.** For each ticker and each window length (20/40/60
trading days), rolling windows step forward 28 trading days at a time. Each
sample's training window contains only observations available at the forecast
origin; the next 28 trading days are held out for evaluation only.

**Baselines.** Naive (repeat last value), moving average (mean of last 5), and
linear trend (least-squares extrapolation). All three see the training window
only.

**Augmentation.** A linear trend is fitted to the training window, residual
standard deviation is estimated from training residuals, and synthetic
continuation points are drawn as trend + Gaussian noise and appended after the
real observations. Ratios of 0.5×, 1.0× and 2.0× the training length are
tested. The generator never receives test data.

**Evaluation.** MAE, RMSE and MAPE, compared pairwise per
sample × model × ratio. Wilcoxon signed-rank is the primary test (forecast
errors are skewed); a paired t-test is reported alongside. Because many grouped
tests are run, both raw and Benjamini–Hochberg FDR-adjusted p-values are
reported.

## Result as currently computed

Augmentation made forecasts substantially worse: mean MAPE rose from 6.44 to
16.04 (+9.60 points), only 20.9% of 12,771 paired comparisons improved, and
degradation grew monotonically with the augmentation ratio.

## Known issue: forecast-window alignment

The result above should not yet be read as evidence about synthetic data
quality. In the current implementation the synthetic points are appended to the
end of the training sequence, so the models treat them as observed history and
forecast the *n_synthetic* steps that follow them — a window that starts up to
120 trading days after the test window begins. Baseline and augmented forecasts
are therefore evaluated against the same 28 days while predicting different
calendar periods, and the offset grows with the augmentation ratio.

A controlled check on the linear-trend model isolates the effect. Fitting the
same augmented series but evaluating it over the indices the test set actually
occupies removes the degradation entirely:

| ratio | baseline MAPE | as implemented | time-aligned |
|-------|---------------|----------------|--------------|
| 0.5   | 8.291         | 12.160         | 8.294        |
| 1.0   | 8.291         | 16.846         | 8.284        |
| 2.0   | 8.291         | 26.877         | 8.301        |

The improvement rate moves from 22.5%/16.7%/11.4% to roughly 49%/50%/47% — a
coin flip, i.e. no measurable effect either way. The same offset drives the
naive and moving-average results, since both forecast from the end of the
augmented series.

Resolving this is the next step for the study: either evaluate the synthetic
continuation directly against the test window it overlaps, or restructure
augmentation so the real endpoint is preserved (residual bootstrapping,
return-space generation).

## Citation

If you refer to this work, please cite it as a pilot study with the caveat
above:

```bibtex
@misc{coldstart_augmentation_2026,
  author = {Kubra Gul Ibacik},
  title  = {Synthetic Time Series Augmentation for Cold-Start Trend
            Forecasting: A Finance Pilot Study},
  year   = {2026},
  note   = {Pilot study; methodology under revision},
  url    = {https://github.com/<your-username>/cold-start-augmentation-research}
}
```

## License

MIT — see [LICENSE](LICENSE).

Market data is retrieved from Yahoo Finance via `yfinance` and is subject to
Yahoo's terms of use. No price data is redistributed in this repository.
