# Exploratory analysis (before the analysis plan)

These scripts are the earlier exploratory analysis that the paper discloses (Section 6.7 and Table 6): one series (the
Lahore–Multan average of WFP wheat prices), decisions made in May only, and a first version of the regime-aware model.
It motivated hypothesis H6. It is not part of the registered pipeline and is kept for transparency.

| File | Contents |
|---|---|
| `wheat_forecast_backtest.py` | First exploratory backtest (models M0–M3, decision rules D0–D5, cost scenarios) |
| `wheat_backtest_v2.py` | Second exploratory backtest (adds the regime-aware model M5 and recalibration); imports the first script |
| `scores_forecast_v2.csv`, `summary_decisions_v2.csv`, `paired_tests_v2.csv` | Its outputs that the paper reports (read by `code/make_paper_numbers.py`) |

Note: this exploratory version set the reach of the 2026 support price to 0.16, the share of the procurement target
reached by July 2026, which was not known in advance. The registered benchmark in `code/` does not use it.

Run with `python exploratory/wheat_backtest_v2.py`; it writes its outputs into this folder.
