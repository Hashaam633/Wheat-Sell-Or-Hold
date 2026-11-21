"""
Placebo test for the world-price term of the regime-aware model M5 (exploratory; added after the registered analysis).

M5's decision gain over SARIMA (H6) could come from real information in world wheat prices, or from any slow-moving series
that happens to line up with Pakistani prices in a few seasons. The placebo keeps everything in M5 the same but replaces the
world price with a version whose timing is scrambled: its monthly log changes are shifted in a circle by s months and added
up again from the first month. The placebo series keeps the volatility, persistence and total change of the real one but no
longer moves with Pakistani prices at the right time. Two families:
  usd  the world wheat price in US dollars is scrambled; the actual exchange rate is kept (tests world wheat information)
  pkr  the world price in rupees is scrambled (tests the whole parity term, exchange rate included)
Shifts s = 24 ... T - 24 months, so no placebo is within two years of the real alignment.
For each placebo, M5 is rebuilt for the five primary-panel series from the stored SARIMA forecasts with the same code
(benchmark_local.m_m5, floor included), scored on the comparable rows of the main benchmark, and run through decision rule B
(base costs, gamma 2, probabilistic advice). The statistic is the H6 statistic: the mean over the 13 seasons 2013-2025 of the
season-level CE difference between M5 and SARIMA. The permutation p-value is (1 + #placebos >= real) / (1 + #placebos).
Output: results/placebo_world_price.csv (one row per placebo), results/placebo_world_price_summary.csv
"""
import sys
import time

import numpy as np
import pandas as pd

import benchmark_local as BL
import score_benchmark as SBM
from decisions_v3 import build_cases, to100, u, u_inv, decide, load as load_dec, COSTS
from paths import RESULTS, DERIVED

B = RESULTS
QC = SBM.QC
PRIMARY = [x for x in BL.SERIES if x.startswith('wfp_wheat') and x != 'wfp_wheat_Punjab']
GAMMA = 2
REAL_LW1 = BL.LW1.copy()


def ce(v, g):
    return 100 * (u_inv(u(np.asarray(v), g).mean(), g) - 1)


def scrambled(level, s):
    """Circularly shift the monthly log changes of a price series by s months and add them up again from the first month."""
    x = np.log(level.dropna())
    d = x.diff().dropna().to_numpy()
    out = np.concatenate([[x.iloc[0]], x.iloc[0] + np.cumsum(np.roll(d, s))])
    return pd.Series(np.exp(out), index=x.index).reindex(level.index)


MP = BL.MP
fam_series = {
    'usd': lambda s: np.log(scrambled(MP['wb_wheat_us_hrw_usd_mt'], s) * MP['fx_avg'] / 25).shift(1),
    'pkr': lambda s: np.log(scrambled(MP['world_hrw_rs40'], s)).shift(1),
}

# ---------------------------------------------------------------- stored SARIMA forecasts and the comparable rows
LF = pd.read_csv(B + 'local_forecasts.csv.gz')
LF = LF[LF.model.isin(['sarima', 'm5']) & LF.unique_id.isin(PRIMARY)].copy()
LF['o'] = pd.PeriodIndex(LF.origin, freq='M')
SAR = {}
for uid, g in LF[LF.model == 'sarima'].groupby('unique_id'):
    s = BL.LOGP[uid]
    for cut, gc in g.sort_values(['o', 'h']).groupby(g.o.map(lambda p: pd.Period(f'{BL.season_of(p)}-04', 'M'))):
        SAR[(uid, cut)] = {o: np.log(go[QC].to_numpy(float)) - s[o] for o, go in gc.groupby('o') if len(go) == BL.H}
keys = pd.read_csv(B + 'scored_rows.csv.gz', usecols=['model', 'unique_id', 'o', 'h'])
keys = keys[(keys.model == 'm5') & keys.unique_id.isin(PRIMARY)][['unique_id', 'o', 'h']]
keys['o'] = pd.PeriodIndex(keys.o, freq='M')
panel = pd.read_csv(DERIVED + 'colab_panel.csv'); panel['ym'] = pd.PeriodIndex(pd.to_datetime(panel.ds), freq='M')
price = {(a, p): y for a, p, y in zip(panel.unique_id, panel.ym, panel.y)}
_, price_d, pol = load_dec()
spread, lam = COSTS['base']


def build_m5(lw1):
    BL.LW1 = lw1
    try:
        rows = []
        for (uid, cut), sar in SAR.items():
            s = BL.LOGP[uid]
            for o, Q in BL.m_m5(s, cut, sorted(sar), uid, sar, use_gap=True, use_floor=True).items():
                Q = np.sort(Q, axis=1)
                for h in range(BL.H):
                    rows.append((uid, o, h + 1, *Q[h]))
    finally:
        BL.LW1 = REAL_LW1
    F = pd.DataFrame(rows, columns=['unique_id', 'o', 'h'] + QC)
    return F.merge(keys, on=['unique_id', 'o', 'h'])


def evaluate(F, model):
    """Quantile score (2013-2025) and season-level CE (gamma 2, rule B, base costs, probabilistic advice)."""
    F = F.copy(); F['model'] = model
    p0 = np.array([price[(a, o)] for a, o in zip(F.unique_id, F.o)])
    p1 = np.array([price.get((a, o + int(h)), np.nan) for a, o, h in zip(F.unique_id, F.o, F.h)])
    F['y'] = np.log(p1 / p0)
    S = SBM.score(F.dropna(subset=['y']).copy())
    S['season'] = [SBM.season_of(o) for o in S.o]
    qs = S[S.season <= 2025].qs.mean()
    dq = F[[o.month in range(5, 13) and h <= 13 - o.month for o, h in zip(F.o, F.h)]][['model', 'unique_id', 'o', 'h'] + QC]
    sc = {}
    for (m, uid, s), (R, Fq) in build_cases(dq, price_d).items():
        if 2013 <= s <= 2025:
            r = pol.get(pd.Period(f'{s}-05', 'M')) + spread
            V, k = decide(R, {'prob': {km: np.exp(to100(Fq[km])) for km in Fq}}, GAMMA, r, lam, 'prob', 'B', 8)
            sc.setdefault(s, []).append(V)
    return qs, pd.Series({s: ce(v, GAMMA) for s, v in sc.items()}).sort_index()


# SARIMA (the comparison) and the real M5, rebuilt with the same code
sar_rows = [(uid, o, h + 1, *np.sort(Q, axis=1)[h]) for (uid, cut), sar in SAR.items() for o, Q in sar.items() for h in range(BL.H)]
Fsar = pd.DataFrame(sar_rows, columns=['unique_id', 'o', 'h'] + QC).merge(keys, on=['unique_id', 'o', 'h'])
qs_sar, ce_sar = evaluate(Fsar, 'sarima')
qs_real, ce_real = evaluate(build_m5(REAL_LW1), 'm5')
real = float((ce_real - ce_sar).mean())
print(f'real M5: H6 statistic {real:.4f} (must equal H6 in primary_test.csv), QS {qs_real:.4f}; SARIMA QS {qs_sar:.4f}', flush=True)

T = MP['wb_wheat_us_hrw_usd_mt'].dropna().shape[0] - 1                 # number of monthly changes
step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = []
t0 = time.time()
for fam, make in fam_series.items():
    for sft in range(24, T - 24 + 1, step):
        qs, cs = evaluate(build_m5(make(sft)), 'm5')
        d = cs - ce_sar
        out.append(dict(family=fam, shift_months=sft, h6_stat=float(d.mean()), qs=qs, d2022=d.get(2022), d2025=d.get(2025),
                        without_2022_2025=float(d.drop([2022, 2025]).mean())))
        if len(out) % 20 == 0:
            print(f'{fam} {sft} done ({len(out)}; {time.time() - t0:.0f}s)', flush=True)
P = pd.DataFrame(out)
P.to_csv(B + 'placebo_world_price.csv', index=False, float_format='%.10g')

summ = []
for fam, g in P.groupby('family'):
    n = len(g)
    summ.append(dict(family=fam, n_placebos=n, real_h6=real, real_qs=qs_real, sarima_qs=qs_sar,
                     placebo_mean=g.h6_stat.mean(), placebo_median=g.h6_stat.median(), placebo_p95=g.h6_stat.quantile(0.95),
                     placebo_max=g.h6_stat.max(), share_placebo_positive=(g.h6_stat > 0).mean(),
                     p_perm=(1 + (g.h6_stat >= real - 1e-9).sum()) / (1 + n),
                     placebo_qs_median=g.qs.median(), share_placebo_qs_below_real=(g.qs <= qs_real).mean(),
                     p_perm_qs=(1 + (g.qs <= qs_real + 1e-12).sum()) / (1 + n),
                     n_years_only=int((g.shift_months % 12 == 0).sum()),
                     p_perm_years_only=(1 + (g[g.shift_months % 12 == 0].h6_stat >= real - 1e-9).sum()) / (1 + int((g.shift_months % 12 == 0).sum()))))
S = pd.DataFrame(summ)
S.to_csv(B + 'placebo_world_price_summary.csv', index=False, float_format='%.10g')
pd.set_option('display.width', 250)
print(S.round(4).to_string(index=False))
