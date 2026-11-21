"""
Exploratory extension (not in the pre-analysis plan): partial sales.

Rule P (rolling, fractional). Each month km = 0..7 (May..December) the farmer holds a share w of the crop and has already
banked A (present value, relative to the May price). She chooses the share theta in {0, 0.1, ..., 1} of the remaining crop to
sell now and, for the rest, the best single future month h, maximising E[u(A + w*theta*V_now + w*(1-theta)*V_h)], where V_h
is drawn from the forecast distribution. Whatever is left in January is sold then. Realised outcome evaluated at actual prices.
With a point forecast the objective is monotone in theta, so rule P reduces exactly to rule B (all or nothing): the
probabilistic-minus-point gap under rule P therefore measures what the distribution adds when sales can be split.

Mechanical benchmarks (no forecast): sell everything in May; equal ninths every month May..January; thirds in May, September, January.
Output: benchmark/partial_sales_rows.csv.gz, partial_sales_summary.csv, partial_sales_tests.csv
"""
import itertools

import numpy as np
import pandas as pd

from decisions_v3 import load, build_cases, to100, u, u_inv, value, decide, tier, GAMMAS, COSTS

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
MODELS = ['ens+R', 'ens', 'm5+R', 'm5', 'sarima+R', 'sarima', 'm0+R']
THETA = np.round(np.arange(0, 1.0001, 0.1), 2)


def decide_partial(R, DIST, g, r, lam, form):
    A, w, path = 0.0, 1.0, []
    for km in range(8):
        vnow = value(R[km], km, r, lam)
        dist = DIST[form][km]
        best_eu, best_th = -np.inf, 1.0
        for h in range(1, 9 - km):
            vfut = value(R[km] * dist[h - 1], km + h, r, lam)                       # draws
            eu = u(A + w * (THETA[:, None] * vnow + (1 - THETA[:, None]) * vfut[None, :]), g).mean(1)
            j = int(np.argmax(eu))
            if eu[j] > best_eu + 1e-12 or (abs(eu[j] - best_eu) <= 1e-12 and THETA[j] > best_th):
                best_eu, best_th = eu[j], THETA[j]
        # selling everything now is always available: compare explicitly (ties -> sell)
        if u(np.array([A + w * vnow]), g).mean() >= best_eu - 1e-12:
            best_th = 1.0
        A += w * best_th * vnow; path.append(round(w * best_th, 4)); w *= (1 - best_th)
        if w <= 1e-9:
            return A, path
    A += w * value(R[8], 8, r, lam); path.append(round(w, 4))
    return A, path


def mechanical(R, r, lam, shares):
    return sum(sh * value(R[k], k, r, lam) for k, sh in shares.items())


if __name__ == '__main__':
    D, price, pol = load()
    D = D[D.model.isin(MODELS)].copy()
    cases = build_cases(D, price)
    print('cases', len(cases))
    spread, lam = COSTS['base']
    rows = []
    for (m, uid, s), (R, F) in cases.items():
        if tier(uid) not in ('primary', 'validation', 'robustness') or not 2013 <= s <= 2025:
            continue
        r = pol.get(pd.Period(f'{s}-05', 'M')) + spread
        DIST = {'point': {km: np.exp(F[km][:, 9:10]) for km in F}, 'prob': {km: np.exp(to100(F[km])) for km in F}}
        for g in GAMMAS:
            for form in ['point', 'prob']:
                VP, path = decide_partial(R, DIST, g, r, lam, form)
                VB, kB = decide(R, DIST, g, r, lam, form, 'B', 8)
                n_sales = sum(1 for p in path if p > 1e-9)
                rows.append((m, form, g, uid, s, VP, VB, n_sales, path[0] if path else 0.0))
        if m == MODELS[0]:                                     # mechanical rules once per market-season
            for name, sh in [('sell all in May', {0: 1.0}), ('equal ninths May-Jan', {k: 1 / 9 for k in range(9)}),
                             ('thirds May/Sep/Jan', {0: 1 / 3, 4: 1 / 3, 8: 1 / 3})]:
                for g in GAMMAS:
                    v = mechanical(R, r, lam, sh)
                    rows.append((name, 'mechanical', g, uid, s, v, np.nan, len(sh), sh.get(0, 0.0)))
    P = pd.DataFrame(rows, columns=['model', 'form', 'gamma', 'unique_id', 'season', 'V_partial', 'V_ruleB', 'n_sales', 'share_sold_may'])
    P['tier'] = P.unique_id.map(tier)
    P.to_csv(B + 'partial_sales_rows.csv.gz', index=False, compression='gzip', float_format='%.8f')

    def ce(v, g):
        return 100 * (u_inv(u(np.asarray(v), g).mean(), g) - 1)

    summ = []
    for (tr, m, form, g), d in P.groupby(['tier', 'model', 'form', 'gamma']):
        summ.append(dict(tier=tr, model=m, form=form, gamma=g, ce_partial_pp=ce(d.V_partial, g),
                         ce_ruleB_pp=ce(d.V_ruleB, g) if form != 'mechanical' else np.nan,
                         mean_sales=d.n_sales.mean(), share_split=(d.n_sales > 1).mean(), n=len(d)))
    S = pd.DataFrame(summ); S.to_csv(B + 'partial_sales_summary.csv', index=False, float_format='%.10g')

    def signflip(x):
        x = np.asarray(x); sg = np.array(list(itertools.product([-1.0, 1.0], repeat=len(x))))
        return float(((sg * x).mean(1) >= x.mean() - 1e-12).mean())

    def season_ce(d, col, g):
        return d.groupby('season')[col].apply(lambda v: ce(v, g))

    tests = []
    for tr in ['primary', 'validation', 'robustness']:
        for m in MODELS:
            for g in GAMMAS:
                d = P[(P.tier == tr) & (P.gamma == g)]
                a = season_ce(d[(d.model == m) & (d.form == 'prob')], 'V_partial', g)
                b = season_ce(d[(d.model == m) & (d.form == 'point')], 'V_partial', g)
                c = season_ce(d[(d.model == m) & (d.form == 'prob')], 'V_ruleB', g)
                for lab, x in [('P prob - P point (= B point)', a - b), ('P prob - B prob', a - c)]:
                    x = x.dropna()
                    tests.append(dict(tier=tr, model=m, gamma=g, contrast=lab, seasons=len(x), mean_diff_pp=x.mean(),
                                      p_one_sided=signflip(x), wins=int((x > 1e-9).sum()), losses=int((x < -1e-9).sum()),
                                      without_2024=x.drop(2024, errors='ignore').mean()))
    T = pd.DataFrame(tests); T.to_csv(B + 'partial_sales_tests.csv', index=False, float_format='%.10g')
    pd.set_option('display.width', 250)
    print(T[T.tier == 'primary'].round(3).to_string(index=False))
    print(S[(S.tier == 'primary')].round(3).to_string(index=False))
