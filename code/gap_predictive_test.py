"""
Direct test of the world-price term (exploratory; added after the registered analysis).

Does the gap between the domestic price and the world price (last month's US hard red winter wheat price in rupees)
help predict later domestic price changes? The test uses every forecast origin from April 2013, every horizon from 1 to 12
months and all 16 market series, instead of the 13 harvest seasons of the decision test.

Two nested median forecasts of the log price h months ahead, both refit each April on past data only:
  SARIMA       the median of the stored SARIMA(1,1,0)(0,1,1)12 forecast
  SARIMA+gap   the same plus the world-price pull h * beta * (gap - mean gap) of the regime-aware model M5
               (benchmark_local.m_m5 without the support-price floor), with beta estimated on the training data
Months filled forward over gaps are excluded as origins and targets, as in the main scoring.
For each panel, horizon and period, the squared errors are averaged over series for each origin month; the Clark-West
statistic for nested models is tested with Newey-West standard errors (lag = horizon, because h-step errors overlap).
Out-of-sample R-squared = 1 - MSE(SARIMA+gap) / MSE(SARIMA). Both p-values are one-sided (the alternative is that the
gap term predicts, i.e. lowers the squared error).
v1.6 adds (i) a season-block bootstrap p-value for the Clark-West mean (seasons resampled with replacement, 9,999 draws),
which does not rely on Newey-West errors, and (ii) the same test for each of the 16 series on its own.
Output: results/gap_predictive_tests.csv, results/gap_predictive_by_series.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

import benchmark_local as BL
import score_benchmark as SBM
from paths import RESULTS, DERIVED

B = RESULTS
QC = SBM.QC
MED = QC[9]                                                  # q0.50
SERIES = [x for x in BL.SERIES if x != 'wfp_wheat_Punjab']   # the 16 market series

LF = pd.read_csv(B + 'local_forecasts.csv.gz', usecols=['model', 'unique_id', 'origin', 'h'] + QC)
LF = LF[(LF.model == 'sarima') & LF.unique_id.isin(SERIES)].copy()
LF['o'] = pd.PeriodIndex(LF.origin, freq='M')
flags = pd.read_csv(DERIVED + 'panel_flags.csv'); flags['ym'] = pd.PeriodIndex(pd.to_datetime(flags.ds), freq='M')
imputed = {(a, p) for a, p, f in zip(flags.unique_id, flags.ym, flags.imputed) if f}

rows = []
for uid, g in LF.groupby('unique_id'):
    s = BL.LOGP[uid]
    for cut, gc in g.sort_values(['o', 'h']).groupby(g.o.map(lambda p: pd.Period(f'{BL.season_of(p)}-04', 'M'))):
        sar = {o: np.log(go[QC].to_numpy(float)) - s[o] for o, go in gc.groupby('o') if len(go) == BL.H}
        gap = BL.m_m5(s, cut, sorted(sar), uid, sar, use_gap=True, use_floor=False)
        for o in sar:
            for h in range(1, BL.H + 1):
                t = o + h
                if t not in s.index or (uid, o) in imputed or (uid, t) in imputed or o < pd.Period('2013-04', 'M'):
                    continue
                f_s = s[o] + np.median(sar[o][h - 1]); f_g = s[o] + np.median(gap[o][h - 1])
                rows.append((uid, SBM.tier(uid), o, BL.season_of(o), h, s[t], f_s, f_g))
E = pd.DataFrame(rows, columns=['unique_id', 'tier', 'o', 'season', 'h', 'y', 'f_sar', 'f_gap'])
E['e2_sar'] = (E.y - E.f_sar) ** 2; E['e2_gap'] = (E.y - E.f_gap) ** 2
E['d_cw'] = E.e2_sar - (E.e2_gap - (E.f_sar - E.f_gap) ** 2)
E['d_mse'] = E.e2_sar - E.e2_gap


def nw_t(x, lag):
    """Mean / Newey-West standard error of a time series x (Bartlett weights)."""
    x = np.asarray(x, float); n = len(x); xc = x - x.mean()
    v = (xc @ xc) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (xc[k:] @ xc[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def block_p(by_o, col, nboot=9999, seed=5):
    """One-sided season-block bootstrap p-value for a positive mean of by_o[col] (index = origin month)."""
    v = by_o[col].to_numpy(float); sea = np.array([BL.season_of(o) for o in by_o.index])
    us = np.unique(sea); idx = [np.where(sea == x)[0] for x in us]
    sums = np.array([v[i].sum() for i in idx]); cnts = np.array([len(i) for i in idx], float)
    W = np.random.default_rng(seed).multinomial(len(us), np.full(len(us), 1 / len(us)), size=nboot).astype(float)
    m = (W @ sums) / (W @ cnts); m0 = v.mean()
    return float((1 + ((m - m0) >= m0).sum()) / (1 + nboot))


PERIODS = {'all, 2013-2026': lambda d: d, '2013-2023 only': lambda d: d[d.season <= 2023],
           'without 2022 and 2025': lambda d: d[~d.season.isin([2022, 2025])]}
TIERS = {'all 16 series': lambda d: d, 'primary': lambda d: d[d.tier == 'primary'], 'validation': lambda d: d[d.tier == 'validation'],
         'robustness': lambda d: d[d.tier == 'robustness']}
out = []
for tname, tf in TIERS.items():
    for pname, pf in PERIODS.items():
        for h in range(1, BL.H + 1):
            d = pf(tf(E[E.h == h]))
            by_o = d.groupby('o')[['e2_sar', 'e2_gap', 'd_cw', 'd_mse']].mean().sort_index()
            t_cw = nw_t(by_o.d_cw, h); t_dm = nw_t(by_o.d_mse, h)
            out.append(dict(panel=tname, period=pname, h=h, n_forecasts=len(d), n_origins=len(by_o), n_seasons=d.season.nunique(),
                            mse_sarima=d.e2_sar.mean(), mse_gap=d.e2_gap.mean(), r2_oos=1 - d.e2_gap.mean() / d.e2_sar.mean(),
                            cw_t=t_cw, p_cw=1 - stats.norm.cdf(t_cw), p_cw_block=block_p(by_o, 'd_cw'), dm_t=t_dm, p_dm=1 - stats.norm.cdf(t_dm),
                            p_dm_block=block_p(by_o, 'd_mse'),
                            share_series_seasons_better=float((d.groupby(['unique_id', 'season']).d_mse.mean() > 0).mean())))
T = pd.DataFrame(out)
T.to_csv(B + 'gap_predictive_tests.csv', index=False, float_format='%.10g')

# v1.6: each series on its own (origins are single months of one series, so no averaging across series)
srows = []
for uid, du in E.groupby('unique_id'):
    for pname, pf in PERIODS.items():
        for h in range(1, BL.H + 1):
            d = pf(du[du.h == h]).sort_values('o').set_index('o')
            if len(d) < 24:
                continue
            t_cw = nw_t(d.d_cw, h); t_dm = nw_t(d.d_mse, h)
            srows.append(dict(unique_id=uid, tier=SBM.tier(uid), period=pname, h=h, n_forecasts=len(d),
                              mse_sarima=d.e2_sar.mean(), mse_gap=d.e2_gap.mean(), r2_oos=1 - d.e2_gap.mean() / d.e2_sar.mean(),
                              cw_t=t_cw, p_cw=1 - stats.norm.cdf(t_cw), dm_t=t_dm, p_dm=1 - stats.norm.cdf(t_dm)))
TS = pd.DataFrame(srows)
TS.to_csv(B + 'gap_predictive_by_series.csv', index=False, float_format='%.10g')
summ = TS.groupby(['period', 'h']).agg(series=('unique_id', 'size'), r2_positive=('r2_oos', lambda x: int((x > 0).sum())),
                                       cw_p05=('p_cw', lambda x: int((x < 0.05).sum())), dm_p05=('p_dm', lambda x: int((x < 0.05).sum())))
print(summ.to_string())
pd.set_option('display.width', 250)
print('forecasts:', len(E), 'series:', E.unique_id.nunique(), 'origins:', E.o.nunique())
print(T[T.h.isin([1, 3, 6, 9, 12])][['panel', 'period', 'h', 'n_forecasts', 'r2_oos', 'cw_t', 'p_cw', 'p_dm', 'share_series_seasons_better']].round(4).to_string(index=False))
