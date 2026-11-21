# Draft v1.1–v1.3 outputs (kept for the correction reported in the paper)

Up to draft v1.3, the random search for starting values of the Markov-switching model (MS-AR) was not seeded: the
statsmodels version used draws these values from its own random generator and ignores NumPy's global seed, so every
rerun gave slightly different MS-AR forecasts. A fit that stopped with a numerical error (`LinAlgError`) was also not
retried, so a rerun could lose that model's forecasts for a season and, because all models are compared on the same rows,
that season for every model. The code in this repository passes the seed to statsmodels and retries failed fits, so every
number can be reproduced exactly.

These files are used only to report what the correction changed (Table 6 of the paper) and the earlier correction of
the support-price floor (v1.0 to v1.1):

| File | Contents |
|---|---|
| `local_forecasts_msar_only.csv.gz` | v1.1–v1.3 MS-AR forecasts (the other models did not change) |
| `decisions_v3_rows.csv.gz` | v1.1–v1.3 decisions |
| `primary_test.csv`, `scores_by_tier_regime.csv` | v1.1–v1.3 test results and forecast scores |
