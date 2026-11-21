# When does probabilistic forecasting pay? Wheat sell-or-hold decisions in Pakistan

Code and data for the paper *When Does Probabilistic Forecasting Pay? Evidence from Wheat Sell-or-Hold Decisions in Pakistan Across Policy Regimes*.

[![Pre-registration: OSF](https://img.shields.io/badge/OSF-Pre--registered-blue)](https://osf.io/cgj2a)
[![License](https://img.shields.io/badge/License-See%20LICENSE-lightgrey)](LICENSE)

The study asks whether farmers who store wheat after harvest make better sell-or-hold decisions when they act on a full probabilistic price forecast rather than on a single point forecast. It benchmarks twelve forecasting approaches (seasonal benchmarks, SARIMA, exponential smoothing, an error-correction regression, a Markov-switching model, LightGBM, quantile regression forests, DeepAR, a temporal fusion transformer, a regime-aware model and an ensemble), plus four zero-shot foundation models, on monthly wheat prices in Pakistan from 2013 to 2026. Every forecast is turned into sell-or-hold decisions for farmers who differ in risk aversion, cost of credit and need for cash, and the hypotheses fixed in the analysis plan are tested on 13 harvest seasons.

All numbers in the paper are generated from the files in `results/` by `code/make_paper_numbers.py`; none is typed by hand.

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh                     # full rerun, about 35 minutes on 2 CPU cores
SKIP_FORECASTS=1 bash run_all.sh    # keep the stored local forecasts, about 27 minutes
```

The run starts with unit tests of the support-price floor mixture (`code/test_mix_floor.py`) and a leakage test of the local models (`code/leakage_test.py`), and ends with `results/paper_numbers.json` (every number in the paper), the four figures in `figures/`, and an independent check (`code/verify_numbers.py`) that recomputes key numbers from the row-level files with plain Python.

Tested with Python 3.13 on Linux. `run_all.sh` needs a Unix shell; on Windows, use Git Bash or WSL, or run the scripts in `code/` one by one in the order listed in `run_all.sh`.

## Repository layout

| Folder | Contents |
|---|---|
| `code/` | All analysis scripts, in the order given in `run_all.sh`; `paths.py` sets the folders |
| `colab/` | The Colab notebook that produced the DeepAR, TFT and foundation-model forecasts (needs a GPU) |
| `data/raw/` | Source files that may be redistributed (WFP, World Bank, Kaggle) |
| `data/compiled/` | Support prices by harvest year, compiled from the sources listed in the file |
| `data/derived/` | Panels built from the raw files: the inputs to every model |
| `results/` | Forecasts, scores, decisions, tests and `paper_numbers.json` |
| `results/colab_raw/` | Forecast files written by the Colab notebook |
| `results/archive_v1.0/` | Outputs of draft v1.0, kept only for the correction reported in the paper (see below) |
| `results/archive_v1.1/` | Outputs of drafts v1.1–v1.3, kept only for the second correction reported in the paper (see below) |
| `results/archive_v1.5/` | Outputs of drafts v1.4–v1.5, kept only for the third correction reported in the paper (see below) |
| `figures/` | Figures 1 to 4 of the paper |
| `exploratory/` | The earlier exploratory analysis disclosed in the paper (not part of the registered pipeline) |

See `data/README.md` for every data source, its license, and how to download the files that are not included.

## Pipeline

| Step | Script | Output | Time* |
|---|---|---|---|
| Panels (optional) | `build_monthly_panel.py`, `parse_pbs_wholesale.py`, `build_colab_panel.py` | `data/derived/*` | 1 min |
| Unit tests | `test_mix_floor.py` | stops the run if the floor mixture of Equation 1 is wrong | seconds |
| Local forecasts | `benchmark_local.py` | `results/local_forecasts.csv.gz` | 8 min |
| Leakage test | `leakage_test.py` | `results/leakage_test.csv`: forecasts unchanged when data after the origin are deleted | 20 s |
| GPU forecasts | `colab/wheat_benchmark_colab.ipynb` (on Colab) | `results/colab_raw/*` | about 1 h on a T4 |
| Combine GPU forecasts | `ingest_colab.py` | `results/colab_forecasts.csv.gz` | seconds |
| Scores | `score_benchmark.py` | scores, recalibrated forecasts, model confidence set, Romano–Wolf | 1 min |
| Decisions, H1 | `decisions_v3.py` | `decisions_v3_rows.csv.gz`, `primary_test.csv` | 2 min |
| H2–H6 and checks | `secondary_v3.py` | secondary tests, cost grid, cash deadlines, other panels, leave-one-season-out, every forecaster's probabilistic-minus-point gap | 2 min |
| Mean as point forecast (exploratory) | `mean_point_check.py` | the mean instead of the median as the point forecast | 1 min |
| Partial sales | `partial_sales_v3.py` | exploratory partial-sales rule | 20 s |
| Rule D | `rule_d_v3.py` | dynamic programming over copula paths | 2 min |
| Sensitivity | `sensitivity_v3.py` | repeated values, real terms, without Quetta | 15 s |
| Power | `power_v3.py` | minimum detectable effects (two designs) | 2 min |
| Ablation | `ablation_m5_v3.py` | regime-aware model without each component | 1 min |
| Placebo (exploratory) | `placebo_world_price.py` | M5 rebuilt with 404 world-price series whose timing is scrambled | 20 min |
| Direct test (exploratory) | `gap_predictive_test.py` | does the price gap predict price changes? 16 series pooled and one by one, horizons 1–12 | 15 s |
| Risk aversion (exploratory) | `risk_sensitivity.py` | H1 and H6 with γ = 8 and with utility over net value (production costs half of the May value) | 15 s |
| Diagnostics | `diagnostics.py` | M5 world-price coefficient by refit, recalibration factor path, decisions changed by the floor, repeated prices by regime | 15 s |
| Figures | `make_figure1.py`, `make_figures_v3.py` | `figures/fig1`–`fig4` | 20 s |
| Paper numbers | `make_paper_numbers.py`, `verify_numbers.py` | `results/paper_numbers.json` | 10 s |

\*On 2 CPU cores. `benchmark_local.py` uses two worker processes.

Large row-level files (`scored_rows.csv.gz`, `decision_quantiles.csv.gz`, `colab_forecasts.csv.gz`) are not included in the release; `run_all.sh` rebuilds them.

## Notes on reproducibility

- **No look-ahead.** Every model is refit each April on data up to the forecast origin; the world price and the consumer price index enter with a one-month lag; policy inputs (the support price and how far procurement could reach) use only what had been announced by each origin. `code/leakage_test.py` checks the local models by deleting all data after an origin.
- **Exact reproduction.** A fresh copy of this repository, run with `bash run_all.sh` (Python 3.13, the package versions in `requirements.txt`, 2 CPU cores), reproduced every file in `results/`, every figure and every number in `results/paper_numbers.json` exactly (last checked for draft v1.9: 50 result files, 4 figures, 20,600 numbers). Other package versions or platforms may change results in the last digits. Results files are written with full precision, so the paper rounds each number only once.
- **GPU forecasts.** To rerun them, open `colab/wheat_benchmark_colab.ipynb` in Google Colab, choose a T4 GPU runtime and run all cells. The notebook installs its own packages (Chronos, TimesFM, NeuralForecast) and saves each result to the Google Drive folder `wheat_benchmark_outputs`; copy those files to `results/colab_raw/` and run `code/ingest_colab.py`. `code/make_colab_notebook.py` regenerates the notebook from the panel.
- **Correction in draft v1.6.** Up to draft v1.5 the support-price floor of the regime-aware model (M5) was mixed into the forecast by interpolating between point masses, which pulled the quantiles above the median toward the median even when the floor did not bind, so the code did not match Equation 1 of the paper; and the December 2025 and January 2026 forecast origins kept a floor although the government had just said it would set no support price. `benchmark_local.py` now computes the exact mixture (tested by `test_mix_floor.py`) and codes those two origins without a floor. `results/archive_v1.5/` keeps the earlier outputs for the comparison in Table 6 of the paper.
- **Correction in draft v1.4.** Up to draft v1.3 the random search for starting values of the Markov-switching model was not seeded (statsmodels ignores NumPy's global seed) and fits that failed with a numerical error were not retried, so reruns gave slightly different results. `benchmark_local.py` now passes the seed to statsmodels and retries failed fits. `results/archive_v1.1/` keeps the earlier outputs for the comparison in Table 6 of the paper.
- **Correction in draft v1.1.** Draft v1.0 coded the support-price floor of the regime-aware model with information that was not yet public at two points (April 2024 and from October 2025). The code here is the corrected version. `results/archive_v1.0/` keeps the v1.0 forecasts of that model and the v1.0 results that the paper compares against (Table 6 of the paper).
- **Analysis plan.** The analysis plan is registered on OSF; the link is given in the paper.

## License

Code: MIT License (see `LICENSE`). Data files keep the licenses of their sources (see `data/README.md`); please cite the original sources when you use them.

## Citation

If you use this code or the derived data, please cite the paper. Full citation details will be added after publication.
