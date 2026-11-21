"""
Ablation of the regime-aware model M5 in the v3 pipeline (leak-free local models only).

M5 = SARIMA(1,1,0)(0,1,1)12 + world-price parity pull k*beta*(gap - mean gap) + support-price floor (wheat grain only).
Variants built from the same SARIMA forecasts (local_forecasts.csv.gz) with the same code as benchmark_local.m_m5:
  m5_full     both parts (rebuilt; must reproduce the stored m5)
  m5_nogap    floor only (no world-price information)
  m5_nofloor  world-price pull only
  m5_q0, m5_q1  full model with the reach of announced-but-uncertain floors (April 2024; 2026 season) set to 0 or 1
                instead of 0.5 (sensitivity, v1.1)
  m5_v10      the v1.0 implementation (archived forecasts), which used the 2026 reach from October 2025
Each variant is recalibrated (+R) exactly as in score_benchmark.py, scored on the same comparable rows as the main benchmark,
and run through decision rule B (base costs, gamma 0/1/2/4). The world price enters with a one-month lag (known at the origin).
Output: benchmark/ablation_scores.csv, ablation_scores_regime.csv (all origins incl. 2026), ablation_decisions.csv, ablation_tests.csv
"""
import itertools

import numpy as np
import pandas as pd

import benchmark_local as BL
import score_benchmark as SBM
from decisions_v3 import build_cases, to100, u, u_inv, value, decide, tier, GAMMAS, COSTS, load as load_dec

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
QC = SBM.QC; QL = SBM.QL


def m5_variant(s, cut, origins, uid, sar, use_gap, use_floor, reach=None):
    """Same code as the stored model (benchmark_local.m_m5); reach overrides the reach of announced-but-uncertain floors."""
    old = BL.UNCERTAIN_REACH
    if reach is not None:
        BL.UNCERTAIN_REACH = reach
    try:
        res = BL.m_m5(s, cut, origins, uid, sar, use_gap=use_gap, use_floor=use_floor)
    finally:
        BL.UNCERTAIN_REACH = old
    return {o: np.sort(Q, axis=1) for o, Q in res.items()}


LF = pd.read_csv(B + 'local_forecasts.csv.gz')
LF = LF[LF.model.isin(['sarima', 'm5'])].copy()
LF['o'] = pd.PeriodIndex(LF.origin, freq='M')
rows = []
for uid, g in LF[LF.model == 'sarima'].groupby('unique_id'):
    s = BL.LOGP[uid]
    g = g.sort_values(['o', 'h'])
    for cut, gc in g.groupby(g.o.map(lambda p: pd.Period(f'{BL.season_of(p)}-04', 'M'))):
        sar = {o: np.log(go[QC].to_numpy(float)) - s[o] for o, go in gc.groupby('o') if len(go) == BL.H}
        orgs = sorted(sar)
        for name, ug, uf, rq in [('m5_full', True, True, None), ('m5_nogap', False, True, None), ('m5_nofloor', True, False, None),
                                 ('m5_q0', True, True, 0.0), ('m5_q1', True, True, 1.0)]:
            res = m5_variant(s, cut, orgs, uid, sar, ug, uf, rq)
            for o, Q in res.items():
                lvl = np.exp(s[o] + Q)
                for h in range(BL.H):
                    rows.append((name, uid, str(o), h + 1, *lvl[h]))
V = pd.DataFrame(rows, columns=['model', 'unique_id', 'origin', 'h'] + QC)
V10 = pd.read_csv(ARCHIVE + 'local_forecasts_m5_only.csv.gz')
V10 = V10[V10.model == 'm5'].assign(model='m5_v10')[V.columns]
V = pd.concat([V, V10], ignore_index=True)

# check: the rebuilt full model reproduces the stored m5
chk = V[V.model == 'm5_full'].merge(LF[LF.model == 'm5'][['unique_id', 'origin', 'h'] + QC], on=['unique_id', 'origin', 'h'], suffixes=('', '_st'))
rel = np.abs(np.log(chk[QC].to_numpy(float) / chk[[c + '_st' for c in QC]].to_numpy(float)))
print('rebuild check: rows', len(chk), 'max |log diff|', rel.max().round(6), 'mean', rel.mean().round(8))

# ---------------------------------------------------------------- score on the same comparable rows as the main benchmark
panel = pd.read_csv(DERIVED + 'colab_panel.csv'); panel['ym'] = pd.PeriodIndex(pd.to_datetime(panel.ds), freq='M')
flags = pd.read_csv(DERIVED + 'panel_flags.csv'); flags['ym'] = pd.PeriodIndex(pd.to_datetime(flags.ds), freq='M')
price = {(a, p): y for a, p, y in zip(panel.unique_id, panel.ym, panel.y)}
imputed = {(a, p) for a, p, f in zip(flags.unique_id, flags.ym, flags.imputed) if f}
F = pd.concat([V, LF[LF.model.isin(['sarima', 'm5'])][V.columns]], ignore_index=True)
F['o'] = pd.PeriodIndex(F.origin, freq='M')
p0 = np.array([price.get((a, o), np.nan) for a, o in zip(F.unique_id, F.o)])
tgt = [o + int(h) for o, h in zip(F.o, F.h)]
p1 = np.array([price.get((a, t), np.nan) for a, t in zip(F.unique_id, tgt)])
F['masked'] = [((a, o) in imputed) or ((a, t) in imputed) for a, o, t in zip(F.unique_id, F.o, tgt)]
F[QC] = np.sort(np.log(F[QC].to_numpy(float) / p0[:, None]), axis=1); F['y'] = np.log(p1 / p0)
F = F[['model', 'unique_id', 'o', 'h'] + QC + ['y', 'masked']]
F = F[F.o >= pd.Period('2010-01', 'M')]
F = SBM.recalibrate(F)
F = F[~F.masked & F.y.notna() & (F.o >= pd.Period('2013-04', 'M'))].copy()
F = SBM.score(F)
F['season'] = [SBM.season_of(o) for o in F.o]; F['regime'] = [SBM.regime_of(o) for o in F.o]; F['tier'] = [SBM.tier(a) for a in F.unique_id]
keys = pd.read_csv(B + 'scored_rows.csv.gz', usecols=['model', 'unique_id', 'o', 'h'])
keys = keys[keys.model == 'm5'][['unique_id', 'o', 'h']]; keys['o'] = pd.PeriodIndex(keys.o, freq='M')
F = F.merge(keys, on=['unique_id', 'o', 'h'])
S = F[F.season <= 2025].groupby(['tier', 'model']).agg(qs=('qs', 'mean'), cov80=('cov80', 'mean'), n=('qs', 'size')).reset_index()
SR = F[(F.tier == 'primary') & (F.season <= 2025)].groupby(['regime', 'model']).qs.mean().unstack(0)
S.to_csv(B + 'ablation_scores.csv', index=False, float_format='%.10g')
SRall = F[F.tier == 'primary'].groupby(['model', 'regime']).agg(qs=('qs', 'mean'), cov80=('cov80', 'mean'), n=('qs', 'size')).reset_index()
SRall.to_csv(B + 'ablation_scores_regime.csv', index=False, float_format='%.10g')

# ---------------------------------------------------------------- decisions (rule B, base costs)
dq = F[[o.month in range(5, 13) and h <= 13 - o.month for o, h in zip(F.o, F.h)]][['model', 'unique_id', 'o', 'h'] + QC].copy()
_, price_d, pol = load_dec()
cases = build_cases(dq, price_d)
spread, lam = COSTS['base']
drows = []
for (m, uid, s), (R, Fq) in cases.items():
    if not 2013 <= s <= 2025:
        continue
    r = pol.get(pd.Period(f'{s}-05', 'M')) + spread
    DIST = {'point': {km: np.exp(Fq[km][:, 9:10]) for km in Fq}, 'prob': {km: np.exp(to100(Fq[km])) for km in Fq}}
    for g in GAMMAS:
        for form in ['point', 'prob']:
            Vv, k = decide(R, DIST, g, r, lam, form, 'B', 8)
            drows.append((m, form, g, uid, tier(uid), s, Vv, k))
Dd = pd.DataFrame(drows, columns=['model', 'form', 'gamma', 'unique_id', 'tier', 'season', 'V', 'k'])
Dd.to_csv(B + 'ablation_decisions.csv', index=False, float_format='%.8f')


def ce(v, g):
    return 100 * (u_inv(u(np.asarray(v), g).mean(), g) - 1)


def signflip(x):
    x = np.asarray(x); sg = np.array(list(itertools.product([-1.0, 1.0], repeat=len(x))))
    return float(((sg * x).mean(1) >= x.mean() - 1e-12).mean())


tests = []
for tr in ['primary', 'validation', 'robustness']:
    for g in [2, 0, 4]:
        d = Dd[(Dd.tier == tr) & (Dd.gamma == g) & (Dd.form == 'prob')]
        sc = {m: d[d.model == m].groupby('season').V.apply(lambda v: ce(v, g)) for m in d.model.unique()}
        for a, b in [('m5_full', 'sarima'), ('m5_nofloor', 'sarima'), ('m5_nogap', 'sarima'), ('m5_full', 'm5_nogap'), ('m5_full', 'm5_nofloor'),
                     ('m5_full+R', 'sarima+R'), ('m5_nofloor+R', 'sarima+R'), ('m5_nogap+R', 'sarima+R'),
                     ('m5_q0', 'sarima'), ('m5_q1', 'sarima'), ('m5_v10', 'sarima'), ('m5_q0+R', 'sarima+R'), ('m5_q1+R', 'sarima+R'), ('m5_v10+R', 'sarima+R')]:
            x = (sc[a] - sc[b]).dropna()
            tests.append(dict(tier=tr, gamma=g, contrast=f'{a} - {b}', seasons=len(x), mean_diff_pp=x.mean(), p_one_sided=signflip(x),
                              wins=int((x > 1e-9).sum()), losses=int((x < -1e-9).sum()),
                              without_2022_2025=x.drop([2022, 2025], errors='ignore').mean(),
                              d2022=x.get(2022, np.nan), d2025=x.get(2025, np.nan)))
T = pd.DataFrame(tests); T.to_csv(B + 'ablation_tests.csv', index=False, float_format='%.10g')
pd.set_option('display.width', 250)
print(S[S.tier == 'primary'].round(4).to_string(index=False)); print(SRall.pivot(index='model', columns='regime', values='qs').round(4).to_string())
print(T[T.gamma == 2].round(3).to_string(index=False))
print(Dd[(Dd.tier == 'primary') & (Dd.gamma == 2)].groupby(['model', 'form']).V.apply(lambda v: ce(v, 2)).unstack().round(2))
