# Cold-Start Augmentation Research

A reproducible pilot study asking a narrow question: **does simple statistical
augmentation improve forecasting accuracy when only a few weeks of history are
available?**

Using daily adjusted-close prices for 11 large-cap U.S. equities (2020–2024),
the pipeline builds 1,419 rolling cold-start samples (4/8/12-week training
windows, 28-trading-day horizon), forecasts each one with three transparent
baselines, repeats the exercise with endpoint-preserving residual-bootstrap
histories at three ratios, and compares the two with paired
statistical tests.

This repository is organized as an academic experiment rather than a software
product. The numbered scripts in `experiments/` define the execution order;
`src/` holds the reusable logic they call.

## Status

The pipeline is fully reproducible end to end: a clean run regenerates every
published number exactly. Synthetic histories remain on the observed training
timeline, so every forecast starts at the true test-window origin.

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
python experiments/04_run_augmentation_experiments.py # aligned bootstrap ensemble, 3 ratios
python experiments/05_compare_results.py             # paired baseline vs augmented table
python experiments/06_run_statistical_tests.py       # Wilcoxon + paired t, BH-FDR corrected
python experiments/07_diagnostic_analysis.py         # subgroup diagnostics
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
  augmentation/              endpoint-preserving bootstrap augmentation
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

**Augmentation.** A linear trend is fitted to the training window and its
residuals are resampled with replacement to create alternative histories over
the same timestamps. A linear correction preserves the observed first and last
prices. Each model forecasts separately from the real and synthetic histories;
their forecasts are averaged. Ratios of 0.5×, 1.0× and 2.0× determine the
number of synthetic histories relative to training length. The generator never
receives test data, and no synthetic value is appended beyond the real forecast
origin.

**Evaluation.** MAE, RMSE and MAPE, compared pairwise per
sample × model × ratio. Wilcoxon signed-rank is the primary test (forecast
errors are skewed); a paired t-test is reported alongside. Because many grouped
tests are run, both raw and Benjamini–Hochberg FDR-adjusted p-values are
reported.

## Result as currently computed

Endpoint-preserving augmentation reduced mean MAPE from 6.4355 to 5.8162
(-0.6193 points; -9.62%). The primary paired Wilcoxon test and the paired
t-test both classify the overall change as significantly improved after
Benjamini–Hochberg correction. Linear trend and moving average improve
significantly; naive is unchanged because every synthetic history preserves the
same real endpoint.

## Forecast-window alignment

An earlier pilot appended synthetic continuation values after the observed
training endpoint and then compared forecasts from that later endpoint against
the immediate test window. That time-index mismatch has been removed. The
current implementation creates equal-length alternative histories, preserves
the observed endpoint, and verifies this invariant in automated tests. Baseline
and augmented forecasts now refer to exactly the same 28 trading days.

## Citation

If you refer to this work, please cite it as:

```bibtex
@misc{coldstart_augmentation_2026,
  author = {Kubra Gul Ibacik},
  title  = {Synthetic Time Series Augmentation for Cold-Start Trend
            Forecasting: A Finance Pilot Study},
  year   = {2026},
  note   = {Endpoint-preserving residual-bootstrap augmentation},
  url    = {https://github.com/kubragul/Cold-start-augmentation--research}
}
```

## License

MIT — see [LICENSE](LICENSE).

Market data is retrieved from Yahoo Finance via `yfinance` and is subject to
Yahoo's terms of use. No price data is redistributed in this repository.
