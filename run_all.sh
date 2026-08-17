#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root regardless of where the script is invoked.
cd "$(dirname "$0")"

echo "[1/8] Download and clean data"
python experiments/01_download_data.py

echo "[2/8] Create rolling cold-start scenarios"
python experiments/02_create_cold_start_scenarios.py

echo "[3/8] Run baseline experiments"
python experiments/03_run_baseline_experiments.py

echo "[4/8] Run augmentation experiments"
python experiments/04_run_augmentation_experiments.py

echo "[5/8] Compare baseline vs augmentation"
python experiments/05_compare_results.py

echo "[6/8] Run statistical tests"
python experiments/06_run_statistical_tests.py

echo "[7/8] Run diagnostic analysis"
python experiments/07_diagnostic_analysis.py

echo "[8/8] Generate paper figures"
python experiments/08_generate_paper_figures.py

echo "All steps completed successfully."
echo "Key outputs:"
echo "  - results/tables/"
echo "  - results/statistical_tests/significance_tests.csv"
echo "  - results/figures/"

