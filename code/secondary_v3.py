"""
Secondary analyses for the wheat sell-or-hold paper (pre-analysis plan, Section 6), v3.

Input : benchmark/decisions_v3_rows.csv.gz (from decisions_v3.py), benchmark/scored_rows.csv.gz (from score_benchmark.py)
Output: benchmark/secondary_tests.csv        H2-H6 with Romano-Wolf stepdown (season bootstrap) + H1 reading check
        benchmark/calibration_value.csv      model x regime cells: 80% coverage and probabilistic-minus-point CE gap
        benchmark/cost_grid.csv              CE gain and value capture ratio across the cost grid
        benchmark/cash_constraint.csv        rule C (forced sale by Jul / Sep / Nov) vs rule B
        benchmark/tier_tests.csv             H1 and H6 contrasts on the validation and robustness panels
        benchmark/loso.csv                   headline numbers with each season left out
        benchmark/key_numbers_secondary.json
        benchmark/prob_point_by_model.csv    v1.6: probabilistic minus point for each of the 24 forecasters (gamma 2, rule B)
Seasons 2013-2025 (validation panel: 2013-2024, PBS ends June 2025). Confirmatory models exclude the foundation group and the random walk.
From v1.6 every test also reports the 90% interval from inverting the exact sign-flip test and the number of seasons
in which the difference is not zero.
"""
import itertools
import json

import numpy as np
import pandas as pd

from decisions_v3 import u, u_inv, GAMMAS, signflip_ci

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
MAIN = ['snaive', 'm0', 'sarima', 'ets', 'ecm', 'msar', 'lgbm', 'qrf', 'deepar', 'tft', 'm5', 'ens']
FORECASTERS = MAIN + [m + '+R' for m in MAIN]
SPREADS = {'s0': 0, 'base': 5, 's10': 10, 's15': 15}
NBOOT = 9999
ALPHA = 0.05


def regime3(s):
    return 'R1' if s <= 2023 else ('T2024' if s == 2024 else '2025')


# ---------------------------------------------------------------- season-level building blocks
X = pd.read_csv(B + 'decisions_v3_rows.csv.gz')
X = X[X.season.between(2013, 2025)].copy()
X['uV'] = np.nan
for g in GAMMAS:
    m = X.gamma == g
    X.loc[m, 'uV'] = u(X.loc[m, 'V'].to_numpy(), g)
KEYS = ['tier', 'model', 'form', 'gamma', 'costs', 'rule', 'kbar', 'season']
SS = X.groupby(KEYS).agg(sumu=('uV', 'sum'), sumV=('V', 'sum'), sumO=('V_oracle', 'sum'), n=('V', 'size')).reset_index()
SS['ce'] = np.nan
for g in GAMMAS:
    m = SS.gamma == g
    SS.loc[m, 'ce'] = u_inv(SS.loc[m, 'sumu'] / SS.loc[m, 'n'], g) - 1
SS['mg'] = SS.sumV / SS.n - 1
SS = SS.set_index(KEYS).sort_index()


def blk(model, form, g, rule='B', kbar=8, costs='base', tr='primary'):
    """Season-level block (index = season) for one setting."""
    return SS.loc[(tr, model, form, g, costs, rule, kbar)]


def pooled_ce(model, form, g, rule='B', kbar=8, costs='base', tr='primary', drop=None):
    d = X[(X.tier == tr) & (X.model == model) & (X.form == form) & (X.gamma == g) & (X.rule == rule) & (X.kbar == kbar) & (X.costs == costs)]
    if drop is not None:
        d = d[d.season != drop]
    return 100 * (u_inv(u(d.V.to_numpy(), g).mean(), g) - 1)


def signflip(x):
    x = np.asarray(x); sg = np.array(list(itertools.product([-1.0, 1.0], repeat=len(x))))
    return float(((sg * x).mean(1) >= x.mean() - 1e-12).mean())


def boot_ci(x, B_=5000):
    rng = np.random.default_rng(1); x = np.asarray(x)
    m = x[rng.integers(0, len(x), (B_, len(x)))].mean(1); return np.percentile(m, [5, 95])


def contrast(a, b, measure='ce'):
    """Season-level difference a - b (in percentage points)."""
    return 100 * (a[measure] - b[measure]).dropna()


SEASONS = list(range(2013, 2026))
rng = np.random.default_rng(20261121)
W = rng.multinomial(len(SEASONS), np.full(len(SEASONS), 1 / len(SEASONS)), size=NBOOT).astype(float)   # bootstrap season counts
W1 = np.ones((1, len(SEASONS)))


def wmean(d, Wm):
    v = d.reindex(SEASONS).to_numpy()
    ok = ~np.isnan(v)
    return (Wm[:, ok] @ v[ok]) / Wm[:, ok].sum(1)


def gap_series(model, g, rule='B', kbar=8, costs='base', tr='primary', measure='ce'):
    return contrast(blk(model, 'prob', g, rule, kbar, costs, tr), blk(model, 'point', g, rule, kbar, costs, tr), measure)


# ---------------------------------------------------------------- calibration cells (H4b)
SR = pd.read_csv(B + 'scored_rows.csv.gz', usecols=['model', 'unique_id', 'o', 'h', 'cov80', 'season', 'tier'])
SR = SR[(SR.tier == 'primary') & SR.model.isin(FORECASTERS) & SR.season.between(2013, 2025)]
om = pd.PeriodIndex(SR.o, freq='M').month
SR['decision_rel'] = (om >= 5) & (om <= 12) & (SR.h <= 13 - om)
COVD = SR[SR.decision_rel].groupby(['model', 'season']).cov80.agg(['sum', 'size'])
COVA = SR.groupby(['model', 'season']).cov80.agg(['sum', 'size'])
REG = {r: [i for i, s in enumerate(SEASONS) if regime3(s) == r] for r in ['R1', 'T2024', '2025']}


def cells(Wm, cov=COVD, g=2, rule='B'):
    """Returns (miscal, gap) arrays of shape (len(Wm), n_cells)."""
    mis, gap = [], []
    for m in FORECASTERS:
        hits = cov.loc[m]['sum'].reindex(SEASONS).fillna(0).to_numpy(); cnt = cov.loc[m]['size'].reindex(SEASONS).fillna(0).to_numpy()
        bp, bq = blk(m, 'prob', g, rule), blk(m, 'point', g, rule)
        su_p, n_p = bp.sumu.reindex(SEASONS).fillna(0).to_numpy(), bp.n.reindex(SEASONS).fillna(0).to_numpy()
        su_q, n_q = bq.sumu.reindex(SEASONS).fillna(0).to_numpy(), bq.n.reindex(SEASONS).fillna(0).to_numpy()
        for r, idx in REG.items():
            w = Wm[:, idx]
            with np.errstate(invalid='ignore', divide='ignore'):
                c = (w @ hits[idx]) / (w @ cnt[idx])
                ce_p = u_inv((w @ su_p[idx]) / (w @ n_p[idx]), g)
                ce_q = u_inv((w @ su_q[idx]) / (w @ n_q[idx]), g)
            mis.append(np.abs(c - 0.80)); gap.append(100 * (ce_p - ce_q))
    return np.array(mis).T, np.array(gap).T


def rowcorr(a, b):
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        ok = ~np.isnan(a[i]) & ~np.isnan(b[i])
        if ok.sum() > 3:
            out[i] = np.corrcoef(a[i, ok], b[i, ok])[0, 1]
    return out


# ---------------------------------------------------------------- value capture ratio (H5)
def vcr_curve(Wm, model='ens+R', form='prob', g=2, rule='B'):
    v = []
    for c in SPREADS:
        b = blk(model, form, g, rule, 8, c)
        num = Wm @ (b.sumV - b.n).reindex(SEASONS).fillna(0).to_numpy()
        den = Wm @ (b.sumO - b.n).reindex(SEASONS).fillna(0).to_numpy()
        v.append(num / den)
    v = np.array(v).T                                       # (B, 4)
    x = np.array(list(SPREADS.values()), float); xc = x - x.mean()
    return v, (v - v.mean(1, keepdims=True)) @ xc / (xc @ xc)


# ---------------------------------------------------------------- the family of secondary hypotheses
d2 = gap_series('ens+R', 0, rule='A')
gB = {g: gap_series('ens+R', g, rule='B') for g in GAMMAS}
gA = {g: gap_series('ens+R', g, rule='A') for g in GAMMAS}
xg = np.array(GAMMAS, float); xgc = xg - xg.mean()
slope_B = pd.concat(gB, axis=1).dropna() @ (xgc / (xgc @ xgc))
slope_A = pd.concat(gA, axis=1).dropna() @ (xgc / (xgc @ xgc))
d4a = (gap_series('ens+R', 2) - gap_series('ens', 2)).dropna()
d6 = contrast(blk('m5', 'prob', 2), blk('sarima', 'prob', 2))


def family(Wm):
    """theta_j for every hypothesis; alternative is theta_j > 0."""
    th = {}
    th['H2 (TOST upper): gap < +1 pp, gamma 0, rule A'] = 1.0 - wmean(d2, Wm)
    th['H2 (TOST lower): gap > -1 pp, gamma 0, rule A'] = wmean(d2, Wm) + 1.0
    th['H3: gap rises with gamma (slope, pp per unit gamma), rule B'] = wmean(slope_B, Wm)
    th['H4a: gap(ens+R) - gap(ens), gamma 2, rule B'] = wmean(d4a, Wm)
    mis, gp = cells(Wm)
    th['H4b: -corr(|coverage - 0.80|, gap) across model x regime'] = -rowcorr(mis, gp)
    _, sl = vcr_curve(Wm)
    th['H5: -slope of VCR on spread (per pp), ens+R prob'] = -sl
    th['H6: CE(M5) - CE(SARIMA), prob, gamma 2, rule B'] = wmean(d6, Wm)
    return th


th_hat = {k: float(v[0]) for k, v in family(W1).items()}
th_star = family(W)
names = list(th_hat)
TS = np.column_stack([th_star[k] for k in names])
se = np.nanstd(TS, axis=0, ddof=1)
t_hat = np.array([th_hat[k] for k in names]) / se
t_star = (TS - np.array([th_hat[k] for k in names])) / se
t_star = np.where(np.isnan(t_star), -np.inf, t_star)

# Romano-Wolf stepdown adjusted p-values
order = np.argsort(-t_hat)
p_adj = np.empty(len(names)); running = 0.0
for pos, j in enumerate(order):
    rest = order[pos:]
    p = float((t_star[:, rest].max(1) >= t_hat[j]).mean())
    running = max(running, p); p_adj[j] = running
p_raw = (t_star >= t_hat[None, :]).mean(0)

rows = []
for j, k in enumerate(names):
    lo, hi = np.nanpercentile(TS[:, j], [5, 95])
    rows.append(dict(hypothesis=k, estimate=th_hat[k], boot_se=se[j], t=t_hat[j], ci90_lo=lo, ci90_hi=hi,
                     p_boot_raw=p_raw[j], p_romano_wolf=p_adj[j], reject_at_5pct=bool(p_adj[j] < ALPHA)))
FAM = pd.DataFrame(rows)

# descriptive numbers that go with the family
desc = {
    'H2_gap_gamma0_ruleA_pp': float(d2.mean()),
    'H3_gap_by_gamma_ruleB_pp': {g: float(gB[g].mean()) for g in GAMMAS},
    'H3_gap_by_gamma_ruleA_pp': {g: float(gA[g].mean()) for g in GAMMAS},
    'H3_slope_ruleA_pp': float(slope_A.mean()),
    'H5_vcr_by_spread': dict(zip(SPREADS.values(), vcr_curve(W1)[0][0].round(4).tolist())),
    'H5_vcr_by_spread_point': dict(zip(SPREADS.values(), vcr_curve(W1, form='point')[0][0].round(4).tolist())),
    'H4b_corr_all_rows_coverage': float(rowcorr(*cells(W1, cov=COVA))[0]),
}

# H1 reading check: the plan says "season-level mean difference (averaged over markets within each season)".
# decisions_v3 pooled markets with the utility function (CE over markets); the literal reading averages realised V.
h1_ce = gap_series('ens+R', 2)
h1_mean = gap_series('ens+R', 2, measure='mg')
h1_rows = []
for lab, d in [('H1 as run (CE over markets within season)', h1_ce), ('H1 literal (mean realised V over markets)', h1_mean)]:
    lo, hi = boot_ci(d); slo, shi = signflip_ci(d)
    h1_rows.append(dict(hypothesis=lab, estimate=d.mean(), ci90_lo=lo, ci90_hi=hi, ci90_signflip_lo=slo, ci90_signflip_hi=shi,
                        n_nonzero=int((d.abs() > 1e-9).sum()), p_signflip_one_sided=signflip(d),
                        wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()), without_2024=d.drop(2024).mean()))
H1C = pd.DataFrame(h1_rows)
pd.concat([FAM, H1C]).to_csv(B + 'secondary_tests.csv', index=False, float_format='%.10g')

# ---------------------------------------------------------------- calibration-to-value cells (figure 6)
mis, gp = cells(W1)
cv = []
i = 0
for m in FORECASTERS:
    for r in REG:
        bp = blk(m, 'prob', 2); n_r = int(bp.n.reindex([SEASONS[k] for k in REG[r]]).sum())
        cv.append(dict(model=m.replace('+R', ''), recal=m.endswith('+R'), regime=r, miscal=mis[0, i], gap_pp=gp[0, i], n=n_r))
        i += 1
CV = pd.DataFrame(cv)
# add signed coverage and the probabilistic CE gain itself
covd = COVD.reset_index(); covd['regime'] = covd.season.map(regime3)
cc = covd.groupby(['model', 'regime'])[['sum', 'size']].sum(); cc = (cc['sum'] / cc['size']).rename('cov80').reset_index()
cc['recal'] = cc.model.str.endswith('+R'); cc['model'] = cc.model.str.replace('+R', '', regex=False)
CV = CV.merge(cc, on=['model', 'recal', 'regime'])
pr = X[(X.tier == 'primary') & (X.gamma == 2) & (X.rule == 'B') & (X.costs == 'base') & X.model.isin(FORECASTERS)].copy()
pr['regime3'] = pr.season.map(regime3)
ceg = pr.groupby(['model', 'form', 'regime3']).V.apply(lambda v: 100 * (u_inv(u(v.to_numpy(), 2).mean(), 2) - 1)).unstack('form').reset_index()
ceg['recal'] = ceg.model.str.endswith('+R'); ceg['model'] = ceg.model.str.replace('+R', '', regex=False)
ceg = ceg.rename(columns={'regime3': 'regime', 'prob': 'ce_prob_pp', 'point': 'ce_point_pp'})
CV = CV.merge(ceg, on=['model', 'recal', 'regime'])
CV.to_csv(B + 'calibration_value.csv', index=False, float_format='%.10g')
corr_by_regime = {r: float(np.corrcoef(CV[CV.regime == r].miscal, CV[CV.regime == r].gap_pp)[0, 1]) for r in REG}
corr_prob_ce = float(np.corrcoef(CV.miscal, CV.ce_prob_pp)[0, 1])

# ---------------------------------------------------------------- cost grid (H5) and cash constraint (rule C)
KEY = ['ens+R', 'm5+R', 'sarima+R', 'm0+R', 'ens', 'm5', 'sarima', 'm0']
cg = []
for m, g, rule, c in itertools.product(KEY, [0, 2], ['A', 'B'], ['s0', 'base', 's10', 's15', 'loss1pct']):
    row = dict(model=m, gamma=g, rule=rule, costs=c)
    for form in ['point', 'prob']:
        b = blk(m, form, g, rule, 8, c)
        row[f'ce_{form}_pp'] = pooled_ce(m, form, g, rule, 8, c)
        row[f'vcr_{form}'] = float((b.sumV - b.n).sum() / (b.sumO - b.n).sum())
    d = gap_series(m, g, rule, 8, c)
    row['gap_pp'] = d.mean(); row['gap_p_one_sided'] = signflip(d)
    cg.append(row)
CG = pd.DataFrame(cg); CG.to_csv(B + 'cost_grid.csv', index=False, float_format='%.10g')

cc_rows = []
for m, g, kb in itertools.product(KEY, GAMMAS, [2, 4, 6, 8]):
    rule = 'B' if kb == 8 else 'C'
    row = dict(model=m, gamma=g, sell_by={2: 'July', 4: 'September', 6: 'November', 8: 'January (rule B)'}[kb], kbar=kb)
    for form in ['point', 'prob']:
        row[f'ce_{form}_pp'] = pooled_ce(m, form, g, rule, kb)
    d = gap_series(m, g, rule, kb)
    lo, hi = boot_ci(d)
    row.update(gap_pp=d.mean(), gap_ci90_lo=lo, gap_ci90_hi=hi, gap_p_one_sided=signflip(d), wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()))
    cc_rows.append(row)
CC = pd.DataFrame(cc_rows); CC.to_csv(B + 'cash_constraint.csv', index=False, float_format='%.10g')

# ---------------------------------------------------------------- other panels
tt = []
for tr in ['primary', 'validation', 'robustness']:
    for lab, a, b_, meas in [('H1: ens+R prob - point', ('ens+R', 'prob'), ('ens+R', 'point'), 'ce'),
                             ('H1 literal: ens+R prob - point (mean realised V)', ('ens+R', 'prob'), ('ens+R', 'point'), 'mg'),
                             ('ens prob - point (no recalibration)', ('ens', 'prob'), ('ens', 'point'), 'ce'),
                             ('H6: M5 - SARIMA (prob)', ('m5', 'prob'), ('sarima', 'prob'), 'ce'),
                             ('M5+R - SARIMA+R (prob)', ('m5+R', 'prob'), ('sarima+R', 'prob'), 'ce'),
                             ('M5+R prob - M0+R prob', ('m5+R', 'prob'), ('m0+R', 'prob'), 'ce')]:
        d = contrast(blk(*a, 2, tr=tr), blk(*b_, 2, tr=tr), meas)
        lo, hi = boot_ci(d); slo, shi = signflip_ci(d)
        tt.append(dict(tier=tr, contrast=lab, seasons=len(d), mean_diff_pp=d.mean(), ci90_lo=lo, ci90_hi=hi,
                       ci90_signflip_lo=slo, ci90_signflip_hi=shi, n_nonzero=int((d.abs() > 1e-9).sum()),
                       p_one_sided=signflip(d), wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()),
                       without_2024=d.drop(2024, errors='ignore').mean()))
TT = pd.DataFrame(tt); TT.to_csv(B + 'tier_tests.csv', index=False, float_format='%.10g')

tier_table = []
for tr in ['primary', 'validation', 'robustness']:
    for m in MAIN:
        for mm in [m, m + '+R']:
            tier_table.append(dict(tier=tr, model=mm, ce_point_pp=pooled_ce(mm, 'point', 2, tr=tr), ce_prob_pp=pooled_ce(mm, 'prob', 2, tr=tr)))
pd.DataFrame(tier_table).to_csv(B + 'tier_ce_table.csv', index=False, float_format='%.10g')

# ---------------------------------------------------------------- leave one season out
lo_rows = []
heads = {'H1 gap (ens+R prob - point), pp': gap_series('ens+R', 2),
         'H1 literal gap, pp': h1_mean,
         'H4a recalibration effect on gap, pp': d4a,
         'H6 M5 - SARIMA (prob), pp': d6,
         'M5+R - SARIMA+R (prob), pp': contrast(blk('m5+R', 'prob', 2), blk('sarima+R', 'prob', 2))}
for lab, d in heads.items():
    for s in [None] + SEASONS:
        dd = d if s is None else d.drop(s, errors='ignore')
        lo_rows.append(dict(quantity=lab, left_out='none' if s is None else s, value=dd.mean(), p_one_sided=signflip(dd)))
for m, form in [('ens+R', 'prob'), ('ens+R', 'point'), ('m5+R', 'prob'), ('m5', 'prob'), ('sarima', 'prob'), ('m0+R', 'prob')]:
    for s in [None] + SEASONS:
        lo_rows.append(dict(quantity=f'CE gain {m} {form}, pp', left_out='none' if s is None else s, value=pooled_ce(m, form, 2, drop=s), p_one_sided=np.nan))
LO = pd.DataFrame(lo_rows); LO.to_csv(B + 'loso.csv', index=False, float_format='%.10g')
loso_summary = LO[LO.left_out != 'none'].groupby('quantity').value.agg(['min', 'max'])
loso_summary['full'] = LO[LO.left_out == 'none'].set_index('quantity').value
loso_summary['without_2024'] = LO[LO.left_out == 2024].set_index('quantity').value

# ---------------------------------------------------------------- v1.6: probabilistic minus point for every forecaster (calibration claim)
pp_rows = []
for m in FORECASTERS:
    for meas in ['ce', 'mg']:
        d = gap_series(m, 2, measure=meas)
        slo, shi = signflip_ci(d)
        pp_rows.append(dict(model=m, measure=meas, seasons=len(d), mean_gap_pp=d.mean(), ci90_signflip_lo=slo, ci90_signflip_hi=shi,
                            p_one_sided_prob_better=signflip(d), p_one_sided_point_better=signflip(-d),
                            wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()), ties=int((d.abs() <= 1e-9).sum()),
                            ce_prob_pooled=pooled_ce(m, 'prob', 2), ce_point_pooled=pooled_ce(m, 'point', 2)))
PPM = pd.DataFrame(pp_rows); PPM.to_csv(B + 'prob_point_by_model.csv', index=False, float_format='%.10g')

# ---------------------------------------------------------------- v1.6: markets per season, and the headline contrasts without 2022 and 2025
mps = {}
for tr in ['primary', 'validation', 'robustness']:
    b = blk('ens+R', 'prob', 2, tr=tr)
    mps[tr] = {int(k): int(v) for k, v in b.n.items()}
wo = {}
for lab, d in [('H6 M5 - SARIMA (prob)', d6), ('M5+R - SARIMA+R (prob)', contrast(blk('m5+R', 'prob', 2), blk('sarima+R', 'prob', 2))),
               ('H1 ens+R prob - point (CE)', gap_series('ens+R', 2)), ('H1 literal (mean V)', h1_mean)]:
    dd = d.drop([2022, 2025], errors='ignore')
    slo, shi = signflip_ci(dd)
    wo[lab] = dict(mean=float(dd.mean()), p_one_sided=signflip(dd), ci90_signflip_lo=slo, ci90_signflip_hi=shi, seasons=int(len(dd)),
                   wins=int((dd > 1e-9).sum()), losses=int((dd < -1e-9).sum()))

# ---------------------------------------------------------------- key numbers
KN = {'markets_per_season': mps, 'without_2022_2025': wo, 'family': FAM.set_index('hypothesis')[['estimate', 'ci90_lo', 'ci90_hi', 'p_romano_wolf', 'reject_at_5pct']].to_dict('index'),
      'h1_check': H1C.set_index('hypothesis').to_dict('index'), 'descriptive': desc,
      'calibration_value': {'corr_all': float(np.corrcoef(CV.miscal, CV.gap_pp)[0, 1]), 'corr_by_regime': corr_by_regime,
                            'corr_miscal_vs_prob_ce': corr_prob_ce, 'n_cells': len(CV)},
      'loso': loso_summary.reset_index().to_dict('records'), 'nboot': NBOOT}
json.dump(KN, open(B + 'key_numbers_secondary.json', 'w'), indent=1, default=float)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(FAM.round(3).to_string(index=False)); print(); print(H1C.round(3).to_string(index=False)); print()
print(json.dumps(desc, indent=1, default=float)); print('calibration-value', KN['calibration_value'])
print(); print(TT.round(2).to_string(index=False)); print(); print(loso_summary.round(2).to_string())
