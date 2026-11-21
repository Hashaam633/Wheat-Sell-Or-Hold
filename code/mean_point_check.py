"""
Exploratory check (added in v1.6, not in the analysis plan): is the median the right point forecast?

The plan's point version treats the median forecast as certain. A reviewer noted that for a right-skewed price
distribution the mean is above the median, so part of any probabilistic-minus-point difference can come from the choice
of location rather than from the shape of the distribution. Here a third version treats the mean of the same predictive
distribution (the 100-level version used by the probabilistic rule) as certain.
Because the payoff V is linear in the price ratio, the probabilistic version for a risk-neutral farmer (gamma = 0) makes
exactly the same decisions as the mean version; at gamma = 0 the plan's H2 therefore compares the mean with the median.

Primary panel, seasons 2013-2025, base costs, rules A and B, gamma 0, 1, 2, 4, the forecasters of the confirmatory tests.
Season-level CE differences (markets combined through the certainty equivalent, as in primary_test.csv), exact one-sided
sign-flip test and the interval from inverting it.
Output: results/mean_point_check.csv
"""
import numpy as np
import pandas as pd

from decisions_v3 import load, build_cases, decide, to100, u, u_inv, tier, COSTS, GAMMAS, signflip_p, signflip_ci
from paths import RESULTS

B = RESULTS
MODELS = ['ens+R', 'ens', 'm5', 'm5+R', 'sarima', 'sarima+R', 'm0+R']

D, price, pol = load()
D = D[D.model.isin(MODELS)].copy()
cases = build_cases(D, price)
spread, lam = COSTS['base']
rows = []
for (m, uid, s), (R, F) in cases.items():
    if tier(uid) != 'primary' or not 2013 <= s <= 2025:
        continue
    r = pol.get(pd.Period(f'{s}-05', 'M')) + spread
    prob = {km: np.exp(to100(F[km])) for km in F}
    DIST = {'point': {km: np.exp(F[km][:, 9:10]) for km in F}, 'prob': prob,
            'mean': {km: prob[km].mean(axis=1, keepdims=True) for km in F}}
    for g in GAMMAS:
        for rule in ['A', 'B']:
            for form in ['point', 'mean', 'prob']:
                V, k = decide(R, DIST, g, r, lam, form, rule, 8)
                rows.append((m, form, g, rule, uid, s, V, k))
X = pd.DataFrame(rows, columns=['model', 'form', 'gamma', 'rule', 'unique_id', 'season', 'V', 'k'])


def season_ce(d, g):
    return d.groupby('season').V.apply(lambda v: 100 * (u_inv(u(v.to_numpy(), g).mean(), g) - 1))


out = []
for (m, g, rule), d in X.groupby(['model', 'gamma', 'rule']):
    ce = {f: season_ce(d[d.form == f], g) for f in ['point', 'mean', 'prob']}
    kk = d.pivot_table(index=['unique_id', 'season'], columns='form', values='k')
    for a, b in [('prob', 'point'), ('mean', 'point'), ('prob', 'mean')]:
        x = (ce[a] - ce[b]).dropna()
        lo, hi = signflip_ci(x)
        out.append(dict(model=m, gamma=g, rule=rule, contrast=f'{a} - {b}', seasons=len(x), mean_diff_pp=x.mean(),
                        ci90_signflip_lo=lo, ci90_signflip_hi=hi, p_one_sided=signflip_p(x), p_one_sided_negative=signflip_p(-x),
                        n_nonzero=int((x.abs() > 1e-9).sum()), wins=int((x > 1e-9).sum()), losses=int((x < -1e-9).sum()),
                        same_month_share=float((kk[a] == kk[b]).mean()),
                        ce_a_pooled=100 * (u_inv(u(d[d.form == a].V.to_numpy(), g).mean(), g) - 1),
                        ce_b_pooled=100 * (u_inv(u(d[d.form == b].V.to_numpy(), g).mean(), g) - 1)))
T = pd.DataFrame(out)
T.to_csv(B + 'mean_point_check.csv', index=False, float_format='%.10g')
pd.set_option('display.width', 250)
print(T[T.model.isin(['ens+R', 'm5', 'sarima'])].round(3).to_string(index=False))
chk = T[(T.gamma == 0) & (T.contrast == 'prob - mean')]
print('gamma 0, prob vs mean: share of identical sale months', chk.same_month_share.round(4).tolist())
