"""
Risk-aversion sensitivity (added in v1.9 after an outside review; not in the analysis plan).

The main analysis applies CRRA utility to V, the discounted value of the stored grain relative to the May price (about 1).
A farmer who must cover production costs out of this revenue faces larger relative swings in net income, which raises
the effective risk aversion over V. Variants, all with rule B, base costs, primary panel, seasons 2013-2025:
  base_g2    : CRRA on V with gamma = 2 (the main analysis, recomputed here so every variant uses the same code);
  gross_g8   : CRRA on V with gamma = 8;
  net_c50_g2 : CRRA on V - c with c = 0.5 (production costs equal to half the May value of the grain), gamma = 2;
  net_c50_g4 : the same with gamma = 4.
Near V = 1 the relative risk aversion with respect to V is gamma / (1 - c), so 4 and 8 for the two net variants.
c = 0.75 is not used: realised V falls below 0.75 in a few market-seasons, where CRRA over net value is undefined.
Certainty equivalents are in the units of V: CE = c + u^-1(mean u(V - c)), and gains are CE - 1 as in the main analysis.

Outputs: results/risk_sensitivity.csv (season-level tests and pooled CE gains for each variant) and
         results/risk_sensitivity_same_month.csv (share of market-seasons in which the probabilistic and the point version
         of each of the 24 forecasters choose the same month, by period).
"""
import numpy as np
import pandas as pd

from paths import RESULTS
from decisions_v3 import load, build_cases, to100, value, u, u_inv, signflip_p, signflip_ci, COSTS, PRIMARY

B = RESULTS
MAIN = ['snaive', 'm0', 'sarima', 'ets', 'ecm', 'msar', 'lgbm', 'qrf', 'deepar', 'tft', 'm5', 'ens']
FORECASTERS = MAIN + [m + '+R' for m in MAIN]
VARIANTS = {'base_g2': (2, 0.0), 'gross_g8': (8, 0.0), 'net_c50_g2': (2, 0.5), 'net_c50_g4': (4, 0.5)}
LABEL = {'base_g2': 'CRRA on gross value, gamma 2 (main analysis)', 'gross_g8': 'CRRA on gross value, gamma 8',
         'net_c50_g2': 'CRRA on net value (costs half of the May value), gamma 2',
         'net_c50_g4': 'CRRA on net value (costs half of the May value), gamma 4'}


def util(V, g, c):
    return u(np.asarray(V, float) - c, g)


def ce(V, g, c):
    return c + u_inv(util(V, g, c).mean(), g)


def decide_B(R, dist, g, c, r, lam, kbar=8):
    """Rule B (rolling monthly sell-or-wait, sale by January) with utility u(V - c)."""
    for km in range(kbar):
        now = util([value(R[km], km, r, lam)], g, c).mean()
        fut = max(util(value(R[km] * dist[km][h - 1], km + h, r, lam), g, c).mean() for h in range(1, kbar - km + 1))
        if now >= fut:
            return value(R[km], km, r, lam), km
    return value(R[kbar], kbar, r, lam), kbar


if __name__ == '__main__':
    D, price, pol = load()
    D = D[D.unique_id.isin(PRIMARY) & D.model.isin(FORECASTERS)]
    cases = build_cases(D, price)
    spread, lam = COSTS['base']
    rows = []
    for (m, uid, s), (R, F) in cases.items():
        if not 2013 <= s <= 2025:
            continue
        r = pol.get(pd.Period(f'{s}-05', 'M')) + spread
        DIST = {'point': {km: np.exp(F[km][:, 9:10]) for km in F}, 'prob': {km: np.exp(to100(F[km])) for km in F}}
        for vname, (g, c) in VARIANTS.items():
            for form in ['point', 'prob']:
                V, k = decide_B(R, DIST[form], g, c, r, lam)
                rows.append((vname, m, form, uid, s, V, k))
    X = pd.DataFrame(rows, columns=['variant', 'model', 'form', 'unique_id', 'season', 'V', 'k'])

    out = []
    for vname, (g, c) in VARIANTS.items():
        d = X[X.variant == vname]

        def season_ce(model, form):
            q = d[(d.model == model) & (d.form == form)]
            return q.groupby('season').V.apply(lambda v: 100 * (ce(v.to_numpy(), g, c) - 1))

        def test(label, a, b):
            x = (a - b).dropna(); lo, hi = signflip_ci(x.to_numpy())
            out.append(dict(variant=vname, variant_label=LABEL[vname], gamma=g, cost_share=c, contrast=label, seasons=len(x),
                            mean_diff_pp=x.mean(), p_one_sided=signflip_p(x.to_numpy()), ci90_signflip_lo=lo, ci90_signflip_hi=hi,
                            wins=int((x > 1e-9).sum()), losses=int((x < -1e-9).sum()), n_nonzero=int((x.abs() > 1e-9).sum())))
        test('H1: ens+R prob - point (CE)', season_ce('ens+R', 'prob'), season_ce('ens+R', 'point'))
        mv = lambda model, form: 100 * d[(d.model == model) & (d.form == form)].groupby('season').V.mean()
        test('H1 literal: ens+R prob - point (mean realised V)', mv('ens+R', 'prob'), mv('ens+R', 'point'))
        test('H6: M5 - SARIMA (prob, CE)', season_ce('m5', 'prob'), season_ce('sarima', 'prob'))
        for model, form in [('ens+R', 'prob'), ('ens+R', 'point'), ('m5', 'prob'), ('sarima', 'prob')]:
            q = d[(d.model == model) & (d.form == form)]
            out.append(dict(variant=vname, variant_label=LABEL[vname], gamma=g, cost_share=c, contrast=f'pooled CE gain {model} {form}',
                            seasons=q.season.nunique(), mean_diff_pp=100 * (ce(q.V.to_numpy(), g, c) - 1)))
        out.append(dict(variant=vname, variant_label=LABEL[vname], gamma=g, cost_share=c, contrast='mean sale month ens+R prob - point',
                        seasons=13, mean_diff_pp=d[(d.model == 'ens+R') & (d.form == 'prob')].k.mean() - d[(d.model == 'ens+R') & (d.form == 'point')].k.mean()))
    T = pd.DataFrame(out)
    T.to_csv(B + 'risk_sensitivity.csv', index=False, float_format='%.10g')

    pv = X.pivot_table(index=['variant', 'model', 'unique_id', 'season'], columns='form', values='k').reset_index()
    pv['period'] = np.where(pv.season <= 2023, '2013-2023', np.where(pv.season == 2024, '2024', '2025'))
    S = pv.groupby(['variant', 'period']).apply(lambda q: pd.Series({'same_month_share': (q.prob == q.point).mean(),
                                                                       'prob_earlier_share': (q.prob < q.point).mean(), 'n': len(q)})).reset_index()
    E = pv[pv.model == 'ens+R'].groupby(['variant', 'period']).apply(lambda q: pd.Series({'same_month_share': (q.prob == q.point).mean(),
                                                                                          'prob_earlier_share': (q.prob < q.point).mean(), 'n': len(q)})).reset_index()
    S['forecasters'] = 'all 24'; E['forecasters'] = 'ens+R'
    pd.concat([S, E]).to_csv(B + 'risk_sensitivity_same_month.csv', index=False, float_format='%.10g')
    print(T.round(3).to_string(index=False))
    print(pd.concat([S, E]).round(3).to_string(index=False))
