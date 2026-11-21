# Draft v1.4–v1.5 outputs (kept for the correction reported in the paper)

Up to draft v1.5, the support-price floor of the regime-aware model (M5) was mixed into the forecast distribution by
interpolating between point masses. That approximation pulled the quantiles above the median toward the median even
when the floor did not bind (for a forecast with a standard deviation of 0.10, the 0.95 quantile fell from 0.165 to
0.132), so the code did not match Equation 1 of the paper. The December 2025 and January 2026 forecast origins also
kept a floor with reach 0.5, although in late December 2025 the government had said it would leave wheat procurement
and set no support price. The code in this repository computes the exact mixture (checked by `code/test_mix_floor.py`)
and codes those two origins without a floor.

These files are used only to report what the correction changed (Table 6 of the paper) and, for the decisions and
tests, as the "v1.4" side of the earlier Markov-switching correction (compared with `archive_v1.1/`):

| File | Contents |
|---|---|
| `local_forecasts_m5_only.csv.gz` | v1.4–v1.5 M5 forecasts (the other local models did not change) |
| `decisions_v3_rows.csv.gz` | v1.4–v1.5 decisions (values stored with 5 decimals) |
| `primary_test.csv`, `secondary_tests.csv` | v1.4–v1.5 test results |
| `scores_by_tier_regime.csv`, `ablation_scores.csv`, `ablation_tests.csv` | v1.4–v1.5 forecast scores and ablation |
