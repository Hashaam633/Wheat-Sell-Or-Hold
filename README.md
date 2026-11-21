# When Does Probabilistic Forecasting Pay?

**Code, data, and replication package for *"When Does Probabilistic Forecasting Pay? Evidence from Wheat Sell-or-Hold Decisions in Pakistan Across Policy Regimes."***

[![Pre-registration: OSF](https://img.shields.io/badge/OSF-Pre--registered-blue)](https://osf.io/cgj2a)
[![License](https://img.shields.io/badge/License-See%20LICENSE-lightgrey)](LICENSE)

## Overview

This repository benchmarks statistical, machine learning, deep learning, and time-series foundation models (including Chronos-Bolt, Chronos-2, and TimesFM 2.5) on Pakistani wheat and flour price series across shifting policy regimes. Beyond forecast accuracy and calibration, it evaluates when probabilistic forecasts generate economic value under realistic farmer sell-or-hold storage decisions, cash constraints, partial sales, and storage cost grids.

- **Pre-registration:** Open Science Framework — [osf.io/cgj2a](https://osf.io/cgj2a)
- **Replication Entry Point:** [`run_all.sh`](run_all.sh)

---

## Repository Structure

```text
wheat-sell-or-hold/
├── code/                     # End-to-end data, modeling, evaluation, and figure scripts
├── colab/                    # GPU notebook for foundation & deep forecasting models
│   └── wheat_benchmark_colab.ipynb
├── data/
│   ├── raw/                  # Raw WFP, World Bank CMO, and USD/PKR datasets
│   ├── compiled/             # Compiled government wheat support prices
│   ├── derived/              # Cleaned panels, PBS wholesale wheat, and urban CPI
│   └── README.md             # Data dictionary and provenance notes
├── exploratory/              # Earlier exploratory backtests (v1/v2) and summary outputs
├── figures/                  # Publication figures generated from benchmark results
├── results/
│   ├── colab_raw/            # Raw forecast outputs from Chronos, TimesFM, and deep models
│   ├── archive_v1.0/         # Archived outputs from v1.0
│   ├── archive_v1.1/         # Archived outputs from v1.1
│   └── *.csv / *.json        # Primary tests, MCS/StepM, ablations, and paper numbers
├── requirements.txt          # Python dependencies
├── run_all.sh                # Master script to reproduce the pipeline
└── LICENSE
