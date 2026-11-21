"""
Leakage test of the local models (added to the pipeline in v1.6; Section 5.1 of the paper).

For each test case (a series, an annual cutoff and an origin inside that year), every local model is run twice:
  full       on the whole series, as in benchmark_local.py
  truncated  on the series with every month after the origin deleted (the world price is used only through the series'
             own months, so it is cut at the same point)
If a model used any information from after the origin, the two forecasts made at that origin would differ.
Both runs use the same list of origins up to the test origin, so the Monte Carlo draws of the Markov-switching model
are the same. The test also compares the full run with the stored forecasts (results/local_forecasts.csv.gz, which are
rounded to 0.001 rupees).
Output: results/leakage_test.csv (one row per model and case: largest absolute difference in log price).
"""
import numpy as np
import pandas as pd

import benchmark_local as BL
from paths import RESULTS

CASES = [('wfp_wheat_Lahore', 2016, 5), ('wfp_wheat_Karachi', 2022, 7), ('pbs_whsl_Multan', 2024, 3),
         ('fao_flour_Lahore', 2025, 9), ('wfp_wheat_Peshawar', 2025, 10)]      # (series, cutoff year, origin index)
MODELS = dict(BL.UNI)
STORED = pd.read_csv(RESULTS + 'local_forecasts.csv.gz')
QC = [c for c in STORED.columns if c.startswith('q')]


def run(s, cut, orgs, uid):
    res = {name: fn(s, cut, orgs) for name, fn in MODELS.items()}
    res['m5'] = BL.m_m5(s, cut, orgs, uid, res['sarima'])
    return res


rows = []
for uid, year, k in CASES:
    s = BL.LOGP[uid]; cut = pd.Period(f'{year}-04', 'M')
    orgs = BL.origins_for(uid, cut)
    o = orgs[min(k, len(orgs) - 1)]
    upto = [x for x in orgs if x <= o]
    full = run(s, cut, upto, uid)
    trunc = run(s.loc[:o], cut, upto, uid)
    st = STORED[(STORED.unique_id == uid) & (STORED.origin == str(o))]
    for name in full:
        a = np.sort(full[name][o], axis=1); b = np.sort(trunc[name][o], axis=1)
        g = st[st.model == name].sort_values('h')
        stored_diff = (np.abs(np.log(g[QC].to_numpy(float)) - (s[o] + a)).max() if len(g) == BL.H else np.nan)
        rows.append(dict(series=uid, cutoff=str(cut), origin=str(o), model=name,
                         max_abs_diff_truncated=float(np.abs(a - b).max()), max_abs_diff_stored=float(stored_diff)))
T = pd.DataFrame(rows)
T.to_csv(RESULTS + 'leakage_test.csv', index=False, float_format='%.3e')
pd.set_option('display.width', 200)
print(T.groupby('model')[['max_abs_diff_truncated', 'max_abs_diff_stored']].max().to_string())
worst = T.max_abs_diff_truncated.max()
print(f'largest difference after deleting the data after the origin: {worst:.2e} (log points)')
if worst > 1e-8:
    raise SystemExit('leakage test failed: a forecast changed when later data were deleted')
print('LEAKAGE TEST PASSED: no local model uses data from after its origin')
