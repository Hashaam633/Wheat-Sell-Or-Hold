"""
Collect every number and table used in the paper from the analysis outputs, so no result is typed by hand.
Reads results/*.csv|json (and results/archive_v1.0/, archive_v1.1/ and archive_v1.5/ for the three corrections), the panels
in data/ and the exploratory outputs; writes results/paper_numbers.json. From v1.6 every results file is written with full
precision, so values here are rounded only to 6 decimals and the paper rounds once, at display.
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, ARCHIVE11, ARCHIVE15, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS; A10 = ARCHIVE; A11 = ARCHIVE11; A15 = ARCHIVE15
from decisions_v3 import u, u_inv, to100  # noqa: E402

MAIN = ['snaive', 'm0', 'sarima', 'ets', 'ecm', 'msar', 'lgbm', 'qrf', 'deepar', 'tft', 'm5', 'ens']
PRIMARY = [f'wfp_wheat_{c}' for c in ['Lahore', 'Multan', 'Karachi', 'Peshawar', 'Quetta']]
LABEL = {'snaive': 'Seasonal naive', 'm0': 'Seasonal climatology (M0)', 'sarima': 'SARIMA', 'ets': 'Damped exponential smoothing',
         'ecm': 'Error-correction regression', 'msar': 'Markov-switching AR', 'lgbm': 'LightGBM quantile', 'qrf': 'Quantile regression forest',
         'deepar': 'DeepAR', 'tft': 'Temporal Fusion Transformer', 'm5': 'Regime-aware (M5)', 'ens': 'Equal-weight ensemble', 'rw': 'Random walk',
         'chronos_bolt_base': 'Chronos-Bolt (base)', 'chronos2': 'Chronos-2', 'chronos2_worldcov': 'Chronos-2 + world price', 'timesfm25': 'TimesFM 2.5'}
N = {'label': LABEL}
D6 = 6


def r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), D6)


def recs(df):
    return df.round(D6).to_dict('records')


def signflip(x):
    x = np.asarray(x); sgn = np.array(list(itertools.product([-1.0, 1.0], repeat=len(x))))
    return float(((sgn * x).mean(1) >= x.mean() - 1e-12).mean())


def ce(v, g=2):
    return 100 * (u_inv(u(np.asarray(v), g).mean(), g) - 1)


# ---------------------------------------------------------------- data facts
panel = pd.read_csv(DERIVED + 'colab_panel.csv'); panel['ym'] = pd.PeriodIndex(pd.to_datetime(panel.ds), freq='M')
info = pd.read_csv(DERIVED + 'panel_series_info.csv')
N['data'] = {'n_series_total': int(len(info)), 'n_market_series': int((info.unique_id != 'wfp_wheat_Punjab').sum()), 'series': info.to_dict('records')}
cmp = {}
for c in ['Lahore', 'Multan', 'Peshawar']:
    a = panel[panel.unique_id == f'wfp_wheat_{c}'].set_index('ym').y
    b = panel[panel.unique_id == f'pbs_whsl_{c}'].set_index('ym').y
    j = pd.concat([a, b], axis=1, keys=['w', 'p']).dropna()
    cmp[c] = dict(premium_pct=r(100 * (j.w / j.p - 1).mean()), corr_dlog=r(np.log(j).diff().corr().iloc[0, 1]), months=len(j))
N['data']['wfp_vs_pbs'] = cmp
rep = {}
for c in ['Lahore', 'Multan', 'Karachi', 'Peshawar', 'Quetta']:
    s = panel[panel.unique_id == f'wfp_wheat_{c}'].y.reset_index(drop=True)
    rep[c] = dict(n=len(s), repeats=int((s.diff().abs() < 1e-9).sum()))
N['data']['repeats'] = rep
N['data']['imputed_months'] = int(pd.read_csv(DERIVED + 'panel_flags.csv').imputed.sum())

# ---------------------------------------------------------------- forecast quality
T = pd.read_csv(B + 'scores_by_tier_regime.csv')
P = T[T.tier == 'primary']
fc = []
for m in MAIN + ['rw']:
    row = {'model': m, 'label': LABEL[m]}
    for reg in ['All', 'R1', 'T2024', 'R2', 'R3']:
        x = P[(P.model == m) & (P.regime == reg)]; xr = P[(P.model == m + '+R') & (P.regime == reg)]
        row[f'qs_{reg}'] = r(x.qs.iloc[0]); row[f'qsR_{reg}'] = r(xr.qs.iloc[0])
        row[f'cov80_{reg}'] = r(x.cov80.iloc[0]); row[f'cov80R_{reg}'] = r(xr.cov80.iloc[0])
        row[f'mae_{reg}'] = r(x.mae.iloc[0]); row[f'cov90_{reg}'] = r(x.cov90.iloc[0])
    row['n'] = int(P[(P.model == m) & (P.regime == 'All')].n.iloc[0])
    v = T[(T.tier == 'validation') & (T.regime == 'All')]
    row['qs_validation'] = r(v[v.model == m].qs.iloc[0]); row['qsR_validation'] = r(v[v.model == m + '+R'].qs.iloc[0])
    f = T[(T.tier == 'robustness') & (T.regime == 'All')]
    row['qs_flour'] = r(f[f.model == m].qs.iloc[0])
    fc.append(row)
N['fc'] = fc
_fc = pd.DataFrame(fc).set_index('model').loc[MAIN]
N['rank_validation'] = dict(spearman=r(_fc.qs_All.corr(_fc.qs_validation, method='spearman')),
                            rank_primary={m: int(v) for m, v in _fc.qs_All.rank().items()},
                            rank_validation={m: int(v) for m, v in _fc.qs_validation.rank().items()},
                            spearman_flour=r(_fc.qs_All.corr(_fc.qs_flour, method='spearman')))
mcs = pd.read_csv(B + 'mcs_primary.csv'); stepm = pd.read_csv(B + 'stepm_primary.csv')
N['mcs_all'] = recs(mcs.rename(columns={'Model name': 'model'})[['model', 'mcs_pvalue', 'mean_qs', 'in_90pct_mcs']])
N['mcs'] = recs(mcs[mcs.in_90pct_mcs].rename(columns={'Model name': 'model'})[['model', 'mcs_pvalue', 'mean_qs']])
N['stepm'] = recs(stepm[stepm.beats_m0_stepm_10pct][['model', 'gain_vs_m0_pct']])
N['stepm_all'] = recs(stepm[['model', 'gain_vs_m0_pct', 'beats_m0_stepm_10pct']])
N['foundation'] = recs(pd.read_csv(B + 'foundation_clean_window.csv'))
N['foundation_all'] = {m: dict(qs=r(P[(P.model == m) & (P.regime == 'All')].qs.iloc[0]), cov80=r(P[(P.model == m) & (P.regime == 'All')].cov80.iloc[0]))
                       for m in ['chronos_bolt_base', 'chronos2', 'chronos2_worldcov', 'timesfm25']}
N['abl_scores'] = recs(pd.read_csv(B + 'ablation_scores.csv'))
N['abl_regime'] = recs(pd.read_csv(B + 'ablation_scores_regime.csv'))

# ---------------------------------------------------------------- decisions, main table
S = pd.read_csv(B + 'decisions_v3_summary.csv')
base = S[(S.tier == 'primary') & (S.gamma == 2) & (S.costs == 'base') & (S.rule == 'B') & (S.kbar == 8)]
dec = []
for m in MAIN + ['rw']:
    g = lambda mm, f, col: base[(base.model == mm) & (base.form == f)][col].iloc[0]
    dec.append(dict(model=m, label=LABEL[m], ce_point=r(g(m, 'point', 'ce_gain_pct')), ce_prob=r(g(m, 'prob', 'ce_gain_pct')),
                    ce_probR=r(g(m + '+R', 'prob', 'ce_gain_pct')), share_loss_probR=r(g(m + '+R', 'prob', 'share_loss')),
                    cvar_probR=r(g(m + '+R', 'prob', 'cvar20_pct')), vcr_probR=r(g(m + '+R', 'prob', 'vcr')),
                    mean_probR=r(g(m + '+R', 'prob', 'mean_gain_pct')), hold_probR=r(g(m + '+R', 'prob', 'share_hold'))))
N['dec'] = dec
X = pd.read_csv(B + 'decisions_v3_rows.csv.gz')
Xp = X[(X.tier == 'primary') & X.season.between(2013, 2025) & (X.gamma == 2) & (X.rule == 'B') & (X.costs == 'base') & (X.kbar == 8)]
orc = Xp[(Xp.model == 'ens+R') & (Xp.form == 'prob')]
N['dec_n_market_seasons'] = int(len(orc))
ms = orc.groupby('unique_id').season.agg(['min', 'count'])
N['market_seasons'] = {u_.replace('wfp_wheat_', ''): dict(first=int(a), n=int(b)) for u_, (a, b) in ms.iterrows()}
N['oracle_mean_gain'] = r(100 * (orc.V_oracle.mean() - 1))
tm = {}
for lab, ss in [('2013-2023', range(2013, 2024)), ('2024', [2024]), ('2025', [2025])]:
    d = Xp[(Xp.model == 'ens+R') & Xp.season.isin(list(ss))]
    tm[lab] = {f: r(d[d.form == f].k.mean()) for f in ['point', 'prob']}
N['timing_ensR'] = tm
allf = Xp[Xp.model.isin(MAIN + [m + '+R' for m in MAIN])]
pv = allf.pivot_table(index=['model', 'unique_id', 'season'], columns='form', values='k').dropna()
pv['period'] = np.where(pv.index.get_level_values('season') <= 2023, 'R1', np.where(pv.index.get_level_values('season') == 2024, '2024', '2025'))
N['same_decision_share_all24'] = {k: r((g.point == g.prob).mean()) for k, g in pv.groupby('period')}
N['earlier_sale_share_all24'] = {k: r((g.prob < g.point).mean()) for k, g in pv.groupby('period')}
N['later_sale_share_all24'] = {k: r((g.prob > g.point).mean()) for k, g in pv.groupby('period')}
pp = {}
for s in [2024, 2025]:
    ch = []
    for uid, g in panel[panel.unique_id.isin(PRIMARY)].groupby('unique_id'):
        g = g.set_index('ym').y
        may, jan = pd.Period(f'{s}-05', 'M'), pd.Period(f'{s + 1}-01', 'M')
        if may in g.index and jan in g.index:
            ch.append(100 * (g[jan] / g[may] - 1))
    pp[s] = dict(may_to_jan_mean_pct=r(np.mean(ch)), min=r(min(ch)), max=r(max(ch)))
N['price_change_may_jan'] = pp

# pooled (all market-seasons as one lottery) versus season-level gap, with a season bootstrap of the pooled gap
def pooled_gap_boot(d_prob, d_point, g=2, B_=5000, seed=11):
    sea = sorted(d_prob.season.unique())
    up = d_prob.assign(uu=u(d_prob.V.to_numpy(), g)).groupby('season').uu.agg(['sum', 'size']).reindex(sea)
    uq = d_point.assign(uu=u(d_point.V.to_numpy(), g)).groupby('season').uu.agg(['sum', 'size']).reindex(sea)
    rng = np.random.default_rng(seed); W = rng.multinomial(len(sea), np.full(len(sea), 1 / len(sea)), size=B_).astype(float)
    cp = 100 * (u_inv((W @ up['sum'].to_numpy()) / (W @ up['size'].to_numpy()), g) - 1)
    cq = 100 * (u_inv((W @ uq['sum'].to_numpy()) / (W @ uq['size'].to_numpy()), g) - 1)
    full = ce(d_prob.V, g) - ce(d_point.V, g)
    lo, hi = np.percentile(cp - cq, [5, 95])
    return dict(gap=r(full), ci90_lo=r(lo), ci90_hi=r(hi), share_boot_le0=r(((cp - cq) <= 0).mean()))
N['pooled_gap_ensR'] = pooled_gap_boot(Xp[(Xp.model == 'ens+R') & (Xp.form == 'prob')], Xp[(Xp.model == 'ens+R') & (Xp.form == 'point')])

# forecast tails: lowest price ratio implied anywhere in the 100-level distributions used for decisions
DQ = pd.read_csv(B + 'decision_quantiles.csv.gz')
QC = [c for c in DQ.columns if c.startswith('q')]
low = np.exp(to100(DQ[QC].to_numpy(float))[:, 0])
bad = low < 0.03                                         # V_k could be <= 0 at k = 8 only if the price ratio is below about 0.024
N['tail'] = dict(n_rows=int(len(DQ)), n_bad=int(bad.sum()), share_bad=r(bad.mean()),
                 models_bad=sorted(DQ.model[bad].unique().tolist()),
                 key_models_bad=int(DQ[bad & DQ.model.isin(['ens', 'ens+R', 'm5', 'm5+R', 'sarima', 'sarima+R'])].shape[0]))

# ---------------------------------------------------------------- tests and robustness tables
N['primary_test_rows'] = recs(pd.read_csv(B + 'primary_test.csv'))
N['secondary'] = recs(pd.read_csv(B + 'secondary_tests.csv'))
N['key_secondary'] = json.load(open(B + 'key_numbers_secondary.json'))
N['power'] = json.load(open(B + 'key_numbers_power.json'))
N['tier_tests'] = recs(pd.read_csv(B + 'tier_tests.csv'))
N['tier_ce'] = recs(pd.read_csv(B + 'tier_ce_table.csv'))
N['cost_grid'] = recs(pd.read_csv(B + 'cost_grid.csv'))
N['cash'] = recs(pd.read_csv(B + 'cash_constraint.csv'))
N['partial_summary'] = recs(pd.read_csv(B + 'partial_sales_summary.csv'))
N['partial_tests'] = recs(pd.read_csv(B + 'partial_sales_tests.csv'))
N['rule_d'] = recs(pd.read_csv(B + 'rule_d_tests.csv'))
N['sensitivity'] = recs(pd.read_csv(B + 'sensitivity_tests.csv'))
N['key_sensitivity'] = json.load(open(B + 'key_numbers_sensitivity.json'))
N['ablation'] = recs(pd.read_csv(B + 'ablation_tests.csv'))
N['loso'] = recs(pd.read_csv(B + 'loso.csv'))
N['calib_value'] = recs(pd.read_csv(B + 'calibration_value.csv'))

sce = lambda m, f: Xp[(Xp.model == m) & (Xp.form == f)].groupby('season').V.apply(lambda v: ce(v.to_numpy()))
d6 = sce('m5', 'prob') - sce('sarima', 'prob')
rest = d6.drop([2022, 2025])
N['m5_vs_sarima_by_season'] = {int(k): r(v) for k, v in d6.items()}
N['m5_vs_sarima_other11'] = dict(mean=r(rest.mean()), wins=int((rest > 1e-9).sum()), losses=int((rest < -1e-9).sum()), ties=int((rest.abs() <= 1e-9).sum()))
N['ensR_gap_by_season'] = {int(k): r(v) for k, v in (sce('ens+R', 'prob') - sce('ens+R', 'point')).items()}

AD = pd.read_csv(B + 'ablation_decisions.csv'); ad = AD[(AD.gamma == 2)]
N['abl_ce'] = {f'{t}|{m}|{f}': r(ce(g.V.to_numpy())) for (t, m, f), g in ad.groupby(['tier', 'model', 'form'])}
RD = pd.read_csv(B + 'rule_d_rows.csv.gz'); rd = RD[(RD.gamma == 2) & (RD.tier == 'primary')]
scd = lambda m, col: rd[rd.model == m].groupby('season')[col].apply(lambda v: ce(v.to_numpy()))
d = (scd('m5', 'V_D') - scd('sarima', 'V_D')).dropna()
N['rule_d_m5_vs_sarima'] = dict(mean=r(d.mean()), p=r(signflip(d)), wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()))

# ---------------------------------------------------------------- 2026 season so far
MP = pd.read_csv(DERIVED + 'monthly_panel.csv'); MP['ym'] = pd.PeriodIndex(MP['month'], freq='M')
may26 = pd.Period('2026-05', 'M')
lat = {}
for c in ['Lahore', 'Multan', 'Karachi', 'Peshawar', 'Quetta']:
    g = panel[panel.unique_id == f'wfp_wheat_{c}'].set_index('ym').y
    if may26 in g.index:
        lat[c] = dict(last=str(g.index.max()), change_pct=r(100 * (g.iloc[-1] / g[may26] - 1)))
N['may2026'] = {'price_change_to_last': lat}

# ---------------------------------------------------------------- v0.5 exploratory analysis (disclosed in Section 6.7)
F2 = pd.read_csv(EXPLORATORY + 'scores_forecast_v2.csv'); S2 = pd.read_csv(EXPLORATORY + 'summary_decisions_v2.csv'); P2 = pd.read_csv(EXPLORATORY + 'paired_tests_v2.csv')
al = F2[F2.period == 'All (2013-2025)'].set_index('model')
s2 = S2[(S2.scenario == 'base') & (S2.period == 'All (2013-2025)')]
p2 = P2[(P2.scenario == 'base') & (P2.model == 'M5') & (P2.rule == 'D3') & (P2.period == 'All (2013-2025)')]
N['v05'] = dict(crps_m5=r(al.loc['M5', 'crps_logpts']), crps_m1=r(al.loc['M1', 'crps_logpts']),
                gain_m5_d3=r(s2[(s2.model == 'M5') & (s2.rule == 'D3')].mean_gain_pct.iloc[0]),
                gain_d1=r(s2[s2.rule == 'D1'].mean_gain_pct.iloc[0]), p_m5d3_vs_d1=r(p2.p_vs_D1.iloc[0]))

# ---------------------------------------------------------------- v1.0 results before the M5 floor correction (archive)
pt10 = pd.read_csv(A10 + 'primary_test.csv').set_index('test')
T10 = pd.read_csv(A10 + 'scores_by_tier_regime.csv'); T10 = T10[T10.tier == 'primary']
N['v10'] = dict(h1=r(pt10.loc['H1 primary: ens+R prob vs point, gamma 2, rule B', 'mean_diff_pp']),
                h6=r(pt10.loc['H6: M5 vs SARIMA (prob), gamma 2, rule B', 'mean_diff_pp']),
                m5_qs_all=r(T10[(T10.model == 'm5') & (T10.regime == 'All')].qs.iloc[0]),
                m5_qs_r3=r(T10[(T10.model == 'm5') & (T10.regime == 'R3')].qs.iloc[0]))

# correction of the M5 floor (v1.1): how many decisions changed relative to v1.0 (both archived)
A_ = pd.read_csv(A10 + 'decisions_v3_rows.csv.gz')
X11 = pd.read_csv(A11 + 'decisions_v3_rows.csv.gz')
k_ = ['model', 'form', 'gamma', 'costs', 'rule', 'kbar', 'unique_id', 'season']
mm_ = A_.merge(X11, on=k_, suffixes=('_a', '_b'))
chg = mm_[(mm_.V_a - mm_.V_b).abs() > 1e-9]
ptv11 = pd.read_csv(A11 + 'primary_test.csv').set_index('test')
N['correction'] = dict(n_changed=int(len(chg)), n_total=int(len(mm_)), seasons=sorted(int(x) for x in chg.season.unique()),
                       models=sorted(chg.model.unique().tolist()),
                       h6b_v10=r(pt10.loc['H6b: M5+R vs SARIMA+R (prob), gamma 2, rule B', 'mean_diff_pp']),
                       h6b_v11=r(ptv11.loc['H6b: M5+R vs SARIMA+R (prob), gamma 2, rule B', 'mean_diff_pp']))

# correction of the MS-AR fit (v1.4): seeded start-value search and retried failed fits, relative to v1.1-v1.3
X15 = pd.read_csv(A15 + 'decisions_v3_rows.csv.gz')
mm2 = X11.merge(X15, on=k_, how='outer', suffixes=('_a', '_b'), indicator=True)
both2 = mm2[mm2._merge == 'both']
chg2 = both2[((both2.V_a - both2.V_b).abs() > 1e-9) | (both2.k_a != both2.k_b)]
ptv14 = pd.read_csv(A15 + 'primary_test.csv').set_index('test')
f14 = pd.read_csv(B + 'local_failures.csv')
N['correction_msar'] = dict(n_rows_v13=int(len(X11)), n_rows_v14=int(len(X15)), n_common=int(len(both2)), n_changed=int(len(chg2)),
                            n_changed_primary=int((chg2.tier_a == 'primary').sum()) if 'tier_a' in chg2 else None,
                            models=sorted(chg2.model.unique().tolist()), only_v13=int((mm2._merge == 'left_only').sum()),
                            only_v14=int((mm2._merge == 'right_only').sum()),
                            h1_v13=r(ptv11.loc['H1 primary: ens+R prob vs point, gamma 2, rule B', 'mean_diff_pp']),
                            h1_v14=r(ptv14.loc['H1 primary: ens+R prob vs point, gamma 2, rule B', 'mean_diff_pp']),
                            p_h1_v13=r(ptv11.loc['H1 primary: ens+R prob vs point, gamma 2, rule B', 'p_one_sided']),
                            p_h1_v14=r(ptv14.loc['H1 primary: ens+R prob vs point, gamma 2, rule B', 'p_one_sided']),
                            h6_v13=r(ptv11.loc['H6: M5 vs SARIMA (prob), gamma 2, rule B', 'mean_diff_pp']),
                            h6_v14=r(ptv14.loc['H6: M5 vs SARIMA (prob), gamma 2, rule B', 'mean_diff_pp']),
                            msar_failures_v14=int((f14.model == 'msar').sum()),
                            failures_v14=recs(f14))

# correction of the M5 floor mixture and of the December 2025 - January 2026 floor (v1.6), relative to v1.4-v1.5
mm3 = X15.merge(X, on=k_, how='outer', suffixes=('_a', '_b'), indicator=True)
both3 = mm3[mm3._merge == 'both']
chg3 = both3[((both3.V_a - both3.V_b).abs() > 1e-5) | (both3.k_a != both3.k_b)]   # v1.5 rows were stored with 5 decimals
chg3k = both3[both3.k_a != both3.k_b]
pt15 = pd.read_csv(A15 + 'primary_test.csv').set_index('test'); pt16 = pd.read_csv(B + 'primary_test.csv').set_index('test')
sec15 = pd.read_csv(A15 + 'secondary_tests.csv').set_index('hypothesis'); sec16 = pd.read_csv(B + 'secondary_tests.csv').set_index('hypothesis')
T15 = pd.read_csv(A15 + 'scores_by_tier_regime.csv'); T15 = T15[T15.tier == 'primary']
T16 = T[T.tier == 'primary']
q15 = lambda m, reg, col='qs': float(T15[(T15.model == m) & (T15.regime == reg)][col].iloc[0])
q16 = lambda m, reg, col='qs': float(T16[(T16.model == m) & (T16.regime == reg)][col].iloc[0])
H1n = 'H1 primary: ens+R prob vs point, gamma 2, rule B'; H6n = 'H6: M5 vs SARIMA (prob), gamma 2, rule B'
H6r = 'H6: CE(M5) - CE(SARIMA), prob, gamma 2, rule B'; H1l = 'H1 literal (mean realised V over markets)'
L15 = pd.read_csv(A15 + 'local_forecasts_m5_only.csv.gz'); L16 = pd.read_csv(B + 'local_forecasts.csv.gz'); L16 = L16[L16.model == 'm5']
QCl = [c for c in L16.columns if c.startswith('q')]
lm = L15.merge(L16, on=['model', 'unique_id', 'origin', 'h'], suffixes=('_a', '_b'))
ld = np.abs(np.log(lm[[c + '_a' for c in QCl]].to_numpy(float)) - np.log(lm[[c + '_b' for c in QCl]].to_numpy(float))).max(1)
N['correction_v16'] = dict(n_common=int(len(both3)), n_changed=int(len(chg3)), n_changed_month=int(len(chg3k)),
                           models=sorted(chg3.model.unique().tolist()), models_month=sorted(chg3k.model.unique().tolist()),
                           n_changed_primary_month=int((chg3k.tier_a == 'primary').sum()) if 'tier_a' in chg3k else None,
                           m5_rows=int(len(lm)), m5_rows_changed=int((ld > 1e-6).sum()), m5_max_change_log=r(ld.max()),
                           h1_v15=r(pt15.loc[H1n, 'mean_diff_pp']), h1_v16=r(pt16.loc[H1n, 'mean_diff_pp']),
                           p_h1_v15=r(pt15.loc[H1n, 'p_one_sided']), p_h1_v16=r(pt16.loc[H1n, 'p_one_sided']),
                           h1lit_v15=r(sec15.loc[H1l, 'estimate']), h1lit_v16=r(sec16.loc[H1l, 'estimate']),
                           h6_v15=r(pt15.loc[H6n, 'mean_diff_pp']), h6_v16=r(pt16.loc[H6n, 'mean_diff_pp']),
                           p_h6_v15=r(pt15.loc[H6n, 'p_one_sided']), p_h6_v16=r(pt16.loc[H6n, 'p_one_sided']),
                           p_h6_rw_v15=r(sec15.loc[H6r, 'p_romano_wolf']), p_h6_rw_v16=r(sec16.loc[H6r, 'p_romano_wolf']),
                           m5_qs_v15=r(q15('m5', 'All')), m5_qs_v16=r(q16('m5', 'All')),
                           m5_cov80_v15=r(q15('m5', 'All', 'cov80')), m5_cov80_v16=r(q16('m5', 'All', 'cov80')),
                           m5_qs_r3_v15=r(q15('m5', 'R3')), m5_qs_r3_v16=r(q16('m5', 'R3')),
                           ens_qs_v15=r(q15('ens', 'All')), ens_qs_v16=r(q16('ens', 'All')))

# ---------------------------------------------------------------- v1.6: leakage test, every forecaster's prob - point, mean point, per-series direct test
LK = pd.read_csv(B + 'leakage_test.csv')
N['leakage'] = dict(n_cases=int(LK[['series', 'origin']].drop_duplicates().shape[0]), models=sorted(LK.model.unique().tolist()),
                    max_abs_diff_truncated=float(LK.max_abs_diff_truncated.max()), max_abs_diff_stored=float(LK.max_abs_diff_stored.max()))
N['prob_point_by_model'] = recs(pd.read_csv(B + 'prob_point_by_model.csv'))
N['mean_point'] = recs(pd.read_csv(B + 'mean_point_check.csv'))
GS = pd.read_csv(B + 'gap_predictive_by_series.csv')
N['gap_by_series'] = {f'{p_}|{h_}': dict(series=int(len(g)), r2_positive=int((g.r2_oos > 0).sum()), cw_p05=int((g.p_cw < 0.05).sum()),
                                        dm_p05=int((g.p_dm < 0.05).sum()), r2_min=r(g.r2_oos.min()), r2_max=r(g.r2_oos.max()))
                      for (p_, h_), g in GS.groupby(['period', 'h'])}

# ---------------------------------------------------------------- world-parity gap at May origins (domestic vs last month's world price)
MPi = pd.read_csv(DERIVED + 'monthly_panel.csv').set_index('month')
N['parity_gap_may_pct'] = {y: r(100 * (MPi.loc[f'{y}-05', 'wfp_punjab_rs40'] / MPi.loc[f'{y}-04', 'world_hrw_rs40'] - 1)) for y in range(2013, 2027)}

# ---------------------------------------------------------------- data notes: FX overlap and PBS unit change
_mp = pd.read_csv(DERIVED + 'monthly_panel.csv'); _mp = _mp[_mp.fx_source == 'SBP']   # SBP monthly means (fx_avg)
sbp = pd.Series(_mp.fx_avg.to_numpy(float), index=pd.PeriodIndex(_mp.month, freq='M'))
kg = pd.read_csv(RAW + 'USD_To_PKR_1992_kaggle.csv', encoding='utf-8-sig'); kg.columns = [c.strip() for c in kg.columns]
kg['ym'] = pd.to_datetime(kg['Date'].str.strip(), format='%d/%m/%Y').dt.to_period('M'); kg['Rate'] = pd.to_numeric(kg['Rate'], errors='coerce')
j = pd.concat([kg.groupby('ym').Rate.mean(), sbp], axis=1, keys=['k', 's']).dropna()
dd = 100 * (j.k / j.s - 1)
N['fx_overlap'] = dict(start=str(j.index.min()), end=str(j.index.max()), mean_pct=r(dd.mean()), max_abs_pct=r(dd.abs().max()))
pw = pd.read_csv(DERIVED + 'pbs_wholesale_wheat.csv'); lh = pw[pw.market == 'Lahore'].set_index('month').rs40
N['pbs_join'] = dict(lahore_jun2024=r(lh.get('2024-06')), lahore_jul2024=r(lh.get('2024-07')))

# ---------------------------------------------------------------- v1.2: calibration diagnostics (80% and 90% coverage, PIT), primary panel, all origins
SRc = pd.read_csv(B + 'scored_rows.csv.gz', usecols=['model', 'tier', 'cov80', 'cov90', 'pit'])
SRc = SRc[(SRc.tier == 'primary') & SRc.model.isin(MAIN + [m + '+R' for m in MAIN])].dropna(subset=['pit'])
cal = SRc.groupby('model').agg(cov80=('cov80', 'mean'), cov90=('cov90', 'mean'), pit_mean=('pit', 'mean'), pit_sd=('pit', 'std'),
                               pit_tails=('pit', lambda x: ((x < 0.05) | (x > 0.95)).mean()),
                               pit_lo=('pit', lambda x: (x < 0.05).mean()), pit_hi=('pit', lambda x: (x > 0.95).mean()), n=('pit', 'size')).reset_index()
N['calibration'] = recs(cal)

# ---------------------------------------------------------------- v1.2: the support price as a lower bound beyond harvest (seasons with routine procurement)
SPd = pd.read_csv(COMPILED + 'support_prices.csv').set_index('harvest_year')
pun = MPi['wfp_punjab_rs40'].dropna(); pun.index = pd.PeriodIndex(pun.index, freq='M')
rows_ = []
for p_, v in pun.items():
    s_ = p_.year if p_.month >= 4 else p_.year - 1
    if 2006 <= s_ <= 2023 and s_ in SPd.index and SPd.loc[s_, 'punjab_procurement'] == 'yes':
        rows_.append(dict(month=p_.month, ratio=v / SPd.loc[s_, 'support_price_rs40']))
RR = pd.DataFrame(rows_)
late = RR[~RR.month.isin([4, 5, 6])]; harv = RR[RR.month.isin([4, 5, 6])]
N['floor_beyond_harvest'] = dict(n_months=int(len(RR)), share_below_all=r((RR.ratio < 1).mean()), p5_all=r(RR.ratio.quantile(0.05)),
                                 share_below_late=r((late.ratio < 1).mean()), p5_late=r(late.ratio.quantile(0.05)),
                                 share_below_harvest=r((harv.ratio < 1).mean()), p5_harvest=r(harv.ratio.quantile(0.05)))

# ---------------------------------------------------------------- v1.2: why recalibration lowered M5's decision value
dR = (sce('m5+R', 'prob') - sce('m5', 'prob')).dropna()
N['m5_recal_effect'] = dict(mean=r(dR.mean()), by_season={int(k): r(v) for k, v in dR.items()},
                            mean_without_2025=r(dR.drop(2025).mean()), worst_season=int(dR.idxmin()), worst=r(dR.min()),
                            pooled_m5=r(ce(Xp[(Xp.model == 'm5') & (Xp.form == 'prob')].V)), pooled_m5R=r(ce(Xp[(Xp.model == 'm5+R') & (Xp.form == 'prob')].V)),
                            pooled_m5_wo2025=r(ce(Xp[(Xp.model == 'm5') & (Xp.form == 'prob') & (Xp.season != 2025)].V)),
                            pooled_m5R_wo2025=r(ce(Xp[(Xp.model == 'm5+R') & (Xp.form == 'prob') & (Xp.season != 2025)].V)),
                            k_2025_m5=r(Xp[(Xp.model == 'm5') & (Xp.form == 'prob') & (Xp.season == 2025)].k.mean()),
                            k_2025_m5R=r(Xp[(Xp.model == 'm5+R') & (Xp.form == 'prob') & (Xp.season == 2025)].k.mean()),
                            cvar_m5=r(base[(base.model == 'm5') & (base.form == 'prob')].cvar20_pct.iloc[0]), cvar_m5R=r(base[(base.model == 'm5+R') & (base.form == 'prob')].cvar20_pct.iloc[0]),
                            loss_m5=r(base[(base.model == 'm5') & (base.form == 'prob')].share_loss.iloc[0]), loss_m5R=r(base[(base.model == 'm5+R') & (base.form == 'prob')].share_loss.iloc[0]))

# ---------------------------------------------------------------- v1.2: the same comparison under rule D (copula paths), primary panel, gamma 2
RD_ = pd.read_csv(B + 'rule_d_rows.csv.gz')
RD_ = RD_[(RD_.tier == 'primary') & (RD_.gamma == 2) & RD_.model.isin(['m5', 'm5+R']) & RD_.season.between(2013, 2025)]
ceD = RD_.groupby(['model', 'season']).V_D.apply(ce).unstack(0)
dD = ceD['m5+R'] - ceD['m5']
N['m5_recal_effect_D'] = dict(by_season={int(k): r(v) for k, v in dD.items()}, mean=r(dD.mean()),
                              k_D={m: {int(s): r(RD_[(RD_.model == m) & (RD_.season == s)].k_D.mean()) for s in (2022, 2024, 2025)} for m in ['m5', 'm5+R']})

# ---------------------------------------------------------------- v1.5: exploratory checks of the world-price term
PLs = pd.read_csv(B + 'placebo_world_price_summary.csv')
PL = pd.read_csv(B + 'placebo_world_price.csv')
N['placebo'] = {row['family']: {k: (r(v) if isinstance(v, float) else (int(v) if isinstance(v, (int, np.integer)) else v))
                                for k, v in row.items()} for row in PLs.to_dict('records')}
for fam in N['placebo']:
    g = PL[PL.family == fam]
    N['placebo'][fam].update(n_ge_real=int((g.h6_stat >= N['placebo'][fam]['real_h6'] - 1e-9).sum()),
                             shift_min=int(g.shift_months.min()), shift_max=int(g.shift_months.max()),
                             placebo_without_2022_2025_median=r(g.without_2022_2025.median()),
                             share_half_2022=r((g.d2022 >= N['m5_vs_sarima_by_season'][2022] / 2).mean()),
                             share_half_2025=r((g.d2025 >= N['m5_vs_sarima_by_season'][2025] / 2).mean()),
                             share_half_both=r(((g.d2022 >= N['m5_vs_sarima_by_season'][2022] / 2) & (g.d2025 >= N['m5_vs_sarima_by_season'][2025] / 2)).mean()),
                             max_d2022=r(g.d2022.max()), max_d2025=r(g.d2025.max()))
GT = pd.read_csv(B + 'gap_predictive_tests.csv')
N['gap_test'] = recs(GT)

# ---------------------------------------------------------------- v1.2: unit of the 2024-25 official wholesale series (divided by 2.5)
unit = {}
for c in ['Lahore', 'Multan', 'Peshawar']:
    w_ = panel[panel.unique_id == f'wfp_wheat_{c}'].set_index('ym').y; p_ = panel[panel.unique_id == f'pbs_whsl_{c}'].set_index('ym').y
    j_ = pd.concat([w_, p_], axis=1, keys=['w', 'p']).dropna()
    unit[c] = dict(ratio_fy25=r((j_.w / j_.p).loc['2024-07':'2025-06'].mean()), ratio_fy22_24=r((j_.w / j_.p).loc['2021-07':'2024-06'].mean()))
N['official_unit_check'] = unit

# ---------------------------------------------------------------- v1.9: checks added after the second outside review
N['risk'] = recs(pd.read_csv(B + 'risk_sensitivity.csv'))
N['risk_same_month'] = recs(pd.read_csv(B + 'risk_sensitivity_same_month.csv'))
N['diagnostics'] = json.load(open(B + 'diagnostics.json'))
_sec = pd.read_csv(B + 'secondary_tests.csv')
N['secondary_basic_ci'] = {row.hypothesis: dict(lo=r(2 * row.estimate - row.ci90_hi), hi=r(2 * row.estimate - row.ci90_lo))
                           for row in _sec.itertuples() if pd.notna(row.ci90_lo)}   # basic bootstrap interval: inverts the recentered test

# ---------------------------------------------------------------- write strict JSON (NaN -> null)
txt = json.dumps(N, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
txt = txt.replace(': NaN', ': null').replace(': Infinity', ': null').replace(': -Infinity', ': null')
open(RESULTS + 'paper_numbers.json', 'w').write(txt)
print('results/paper_numbers.json written')
for k in ['dec_n_market_seasons', 'market_seasons', 'oracle_mean_gain', 'pooled_gap_ensR', 'tail', 'v10', 'correction', 'timing_ensR', 'm5_vs_sarima_other11']:
    print(k, N[k])
