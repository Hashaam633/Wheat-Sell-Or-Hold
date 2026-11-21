# Draft v1.0 outputs (kept for the correction reported in the paper)

Draft v1.0 coded the support-price floor of the regime-aware model (M5) with information that was not yet public at two
points: at the April 2024 origin it assumed that Punjab would not procure (decided in May 2024), and from October 2025 it
used the reach of the 2026 procurement (about 16 percent of the target, known only in July 2026). The code in this
repository uses the real-time rule described in Section 5.3 of the paper.

These files are used only to report what the correction changed (Table 6 of the paper and the ablation variant `m5_v10`):

| File | Contents |
|---|---|
| `local_forecasts_m5_only.csv.gz` | v1.0 forecasts of M5 (the other models did not change) |
| `decisions_v3_rows.csv.gz` | v1.0 decisions |
| `primary_test.csv`, `scores_by_tier_regime.csv` | v1.0 test results and forecast scores |
