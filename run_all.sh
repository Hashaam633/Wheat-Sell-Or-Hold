#!/usr/bin/env bash
# Rerun the analysis from the panels in data/derived to the numbers in results/paper_numbers.json.
#
#   bash run_all.sh                    full run (about 35 minutes on 2 CPU cores)
#   SKIP_FORECASTS=1 bash run_all.sh   keep the stored local forecasts (saves about 8 minutes)
#   REBUILD_PANELS=1 bash run_all.sh   first rebuild the panels from the raw files (download them first, see data/README.md)
#
# The deep-learning and foundation-model forecasts come from colab/wheat_benchmark_colab.ipynb (GPU);
# their outputs are stored in results/colab_raw/ and are not rerun here.
set -euo pipefail
cd "$(dirname "$0")/code"

step() { echo; echo "==> $*"; python "$@"; }

if [ "${REBUILD_PANELS:-0}" = "1" ]; then
  step build_monthly_panel.py      # monthly_panel.csv, cpi_urban_general.csv
  step parse_pbs_wholesale.py      # pbs_wholesale_wheat.csv (official wholesale prices, Table 145)
  step build_colab_panel.py        # colab_panel.csv, panel_flags.csv, panel_series_info.csv
fi

step test_mix_floor.py             # unit tests of the support-price floor mixture (Equation 1)
if [ "${SKIP_FORECASTS:-0}" != "1" ]; then
  step benchmark_local.py          # results/local_forecasts.csv.gz (10 local models)
fi
step leakage_test.py               # deleting the data after an origin leaves every local forecast unchanged
step ingest_colab.py               # results/colab_forecasts.csv.gz (DeepAR, TFT, foundation models)
step score_benchmark.py            # forecast scores, recalibration, model confidence set, Romano-Wolf
step decisions_v3.py               # sell-or-hold decisions and the primary test (H1)
step secondary_v3.py               # H2-H6, cost grid, cash deadlines, other panels, leave-one-season-out, every forecaster
step mean_point_check.py           # the mean instead of the median as the point forecast (exploratory)
step partial_sales_v3.py           # exploratory partial-sales rule
step rule_d_v3.py                  # rule D: copula paths and least-squares Monte Carlo
step sensitivity_v3.py             # repeated WFP values, real terms, without Quetta
step power_v3.py                   # minimum detectable effects (two designs)
step ablation_m5_v3.py             # M5 without the world-price term / without the floor
step placebo_world_price.py        # placebo world prices with scrambled timing (exploratory; about 20 minutes)
step gap_predictive_test.py        # does the price gap predict price changes? all 16 series (exploratory)
step risk_sensitivity.py           # more risk aversion and utility over net value (exploratory, v1.9)
step diagnostics.py                # M5 world-price coefficient, recalibration path, floor decisions, repeated prices (v1.9)
step make_figure1.py               # figures/fig1_prices.png
step make_figures_v3.py            # figures/fig2-fig4
step make_paper_numbers.py         # results/paper_numbers.json (every number in the paper)
step verify_numbers.py             # independent recomputation of key numbers
