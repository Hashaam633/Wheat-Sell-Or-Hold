"""
Decision analysis v3 (pre-analysis plan, Sections 5-6): point vs probabilistic sell-or-hold decisions.

Input : benchmark/decision_quantiles.csv.gz (forecasts at origins May-December, horizons up to January), colab_panel.csv,
        monthly_panel.csv (policy rate)
Output: benchmark/decisions_v3_rows.csv.gz (one row per model x form x setting x market x season)
        benchmark/decisions_v3_summary.csv, primary_test.csv, calibration_to_value.csv, key_numbers_decisions.json

Payoff (relative to the May price): V_k = [R_k (1 - lam)^k - c k] / (1 + r)^(k/12), k = months after May (0..8).
r = SBP policy rate in May of the season + spread. CRRA utility u(V) = V^(1-g)/(1-g) (log for g = 1).
Point form: the median forecast treated as certain. Probabilistic form: 19 quantiles interpolated to 100 levels
(linear extrapolation beyond 0.05 and 0.95 using the two outermost quantiles).
Rules: A single sale chosen in May; B rolling-horizon monthly sell-or-wait (must sell by January);
       C rule B with a forced sale by July (2), September (4) or November (6).
Evaluation: realised V at actual prices. CE over a set of outcomes = u^-1(mean u(V)); CE gain = CE - 1.
Tests: exact one-sided sign-flip test over seasons; 90% season-bootstrap interval (as in the plan) and, from v1.6, the 90%
interval from inverting the sign-flip test (signflip_ci), which always agrees with the test.
"""
import itertools
import json

import numpy as np
import pandas as pd

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
QL = np.round(np.arange(0.05, 0.951, 0.05), 2); QC = [f'q{q:.2f}' for q in QL]
TAU = (np.arange(100) + 0.5) / 100
PRIMARY = [f'wfp_wheat_{c}' for c in ['Lahore', 'Multan', 'Karachi', 'Peshawar', 'Quetta']]
GAMMAS = [0, 1, 2, 4]
COSTS = {'base': (0.05, 0.003), 's0': (0.00, 0.003), 's10': (0.10, 0.003), 's15': (0.15, 0.003), 'loss1pct': (0.05, 0.010)}
C_STORE = 0.003


def signflip_p(x):
    """One-sided exact sign-flip p-value for a positive mean (all 2^n sign patterns)."""
    x = np.asarray(x, float); sg = np.array(list(itertools.product([-1.0, 1.0], repeat=len(x))))
    return float(((sg * x).mean(1) >= x.mean() - 1e-12).mean())


def signflip_ci(x, level=0.90):
    """Interval from inverting the exact sign-flip test (added in v1.6): the shifts delta that neither one-sided test at
    (1 - level) / 2 rejects for x - delta. The one-sided p-value for x - delta is (1 + #{subsets S: mean of x over S <= delta}) / 2^n,
    so the bounds are order statistics of the 2^n - 1 subset means; the lower bound is above 0 exactly when the one-sided
    test at (1 - level) / 2 rejects."""
    x = np.asarray(x, float); n = len(x)
    S = np.array(list(itertools.product([0.0, 1.0], repeat=n)))[1:]
    m = np.sort((S @ x) / S.sum(1))
    K = int(np.floor((1 - level) / 2 * 2 ** n - 1)) + 1
    return float(m[K - 1]), float(m[-K])


def tier(u):
    if u in PRIMARY: return 'primary'
    if u.startswith('pbs_whsl'): return 'validation'
    if u.startswith('fao_flour'): return 'robustness'
    return 'descriptive'


def regime_of_season(s):
    return 'R1' if s <= 2023 else ('T2024' if s == 2024 else ('R2/R3 2025' if s == 2025 else 'R3 2026'))


def to100(q):
    """19 quantiles (rows of a 2-D array) -> 100 levels, with linear extrapolation in the tails."""
    q = np.sort(np.atleast_2d(q), axis=1)
    out = np.vstack([np.interp(TAU, QL, row) for row in q])
    lo = TAU < 0.05; hi = TAU > 0.95
    out[:, lo] = q[:, [0]] - (0.05 - TAU[lo])[None, :] * ((q[:, 1] - q[:, 0]) / 0.05)[:, None]
    out[:, hi] = q[:, [-1]] + (TAU[hi] - 0.95)[None, :] * ((q[:, -1] - q[:, -2]) / 0.05)[:, None]
    return out


def u(V, g):
    V = np.maximum(V, 1e-6)
    return np.log(V) if g == 1 else (V ** (1 - g) - 1) / (1 - g) if g != 0 else V


def u_inv(x, g):
    return np.exp(x) if g == 1 else (x * (1 - g) + 1) ** (1 / (1 - g)) if g != 0 else x


def value(R, k, r, lam):
    return (R * (1 - lam) ** k - C_STORE * k) / (1 + r) ** (k / 12)


def load():
    D = pd.read_csv(B + 'decision_quantiles.csv.gz')
    D['o'] = pd.PeriodIndex(D.o, freq='M')
    panel = pd.read_csv(DERIVED + 'colab_panel.csv'); panel['ym'] = pd.PeriodIndex(pd.to_datetime(panel.ds), freq='M')
    price = {(u_, p): y for u_, p, y in zip(panel.unique_id, panel.ym, panel.y)}
    MP = pd.read_csv(DERIVED + 'monthly_panel.csv'); MP['ym'] = pd.PeriodIndex(MP['month'], freq='M')
    pol = MP.set_index('ym')['policy_rate_pct'] / 100
    return D, price, pol


def build_cases(D, price):
    """Per (model, market, season): actual price ratios R_k (k=0..8) and forecast log-ratio quantiles by origin month."""
    D['season'] = D.o.dt.year
    cases = {}
    for (m, uid, s), g in D.groupby(['model', 'unique_id', 'season']):
        may = pd.Period(f'{s}-05', 'M')
        p0 = price.get((uid, may))
        actual = [price.get((uid, may + k)) for k in range(9)]
        if p0 is None or any(a is None for a in actual):
            continue
        R = np.array(actual) / p0
        F = {}
        ok = True
        for km in range(8):                                      # origin = May + km, horizons 1..8-km
            o = may + km
            go = g[g.o == o].sort_values('h')
            if len(go) < 8 - km:
                ok = False; break
            F[km] = go[QC].to_numpy()[: 8 - km]
        if ok:
            cases[(m, uid, s)] = (R, F)
    return cases


def decide(R, DIST, g, r, lam, form, rule, kbar=8):
    """Returns realised V and the sale month for one case. DIST[form][km] = price-ratio draws, rows = horizons."""
    if rule == 'A':
        dist = DIST[form][0]
        best_k, best = 0, u(np.array([1.0]), g).mean()
        for k in range(1, kbar + 1):
            eu = u(value(dist[k - 1], k, r, lam), g).mean()
            if eu > best + 1e-12:
                best, best_k = eu, k
        return value(R[best_k], best_k, r, lam), best_k
    for km in range(0, kbar):                                    # rule B / C: decide each month km = 0..kbar-1
        now = u(np.array([value(R[km], km, r, lam)]), g).mean()
        dist = DIST[form][km]
        fut = max(u(value(R[km] * dist[h - 1], km + h, r, lam), g).mean() for h in range(1, kbar - km + 1))
        if now >= fut:
            return value(R[km], km, r, lam), km
    return value(R[kbar], kbar, r, lam), kbar


if __name__ == '__main__':
    D, price, pol = load()
    cases = build_cases(D, price)
    print('cases (model x market x season):', len(cases))
    settings = [(g, 'base', rule, kb) for g in GAMMAS for rule, kb in [('A', 8), ('B', 8), ('C', 2), ('C', 4), ('C', 6)]]
    settings += [(g, c, rule, 8) for g in [0, 2] for c in COSTS if c != 'base' for rule in ['A', 'B']]
    import os, sys
    rows = []
    for (m, uid, s), (R, F) in ([] if '--reuse' in sys.argv else cases.items()):
        rmay = pol.get(pd.Period(f'{s}-05', 'M'))
        DIST = {'point': {km: np.exp(F[km][:, 9:10]) for km in F}, 'prob': {km: np.exp(to100(F[km])) for km in F}}
        for g, cname, rule, kb in settings:
            spread, lam = COSTS[cname]; r = rmay + spread
            for form in ['point', 'prob']:
                V, k = decide(R, DIST, g, r, lam, form, rule, kb)
                orc = max(value(R[kk], kk, r, lam) for kk in range(kb + 1))
                rows.append((m, form, g, cname, rule, kb, uid, s, V, k, orc))
    X = pd.read_csv(B + 'decisions_v3_rows.csv.gz') if '--reuse' in sys.argv else pd.DataFrame(rows, columns=['model', 'form', 'gamma', 'costs', 'rule', 'kbar', 'unique_id', 'season', 'V', 'k', 'V_oracle'])
    X['tier'] = X.unique_id.map(tier); X['regime'] = X.season.map(regime_of_season)
    X.to_csv(B + 'decisions_v3_rows.csv.gz', index=False, compression='gzip', float_format='%.8f')

    def summarise(d):
        g = d.name[3]                                         # gamma is the 4th grouping key
        V = d.V.to_numpy(); orc = d.V_oracle.to_numpy()
        ce = u_inv(u(V, g).mean(), g) - 1
        return pd.Series({'ce_gain_pct': 100 * ce, 'mean_gain_pct': 100 * (V.mean() - 1), 'share_loss': (V < 1 - 1e-9).mean(),
                          'cvar20_pct': 100 * (np.sort(V)[: max(1, int(np.ceil(0.2 * len(V))))].mean() - 1),
                          'vcr': (V.mean() - 1) / (orc.mean() - 1) if orc.mean() > 1 else np.nan, 'share_hold': (d.k > 0).mean(), 'n': len(V)})

    comp = X[X.season.between(2013, 2025)]
    S = comp.groupby(['tier', 'model', 'form', 'gamma', 'costs', 'rule', 'kbar']).apply(summarise).reset_index()
    S.to_csv(B + 'decisions_v3_summary.csv', index=False, float_format='%.10g')

    # ---- primary test (pre-analysis plan H1)
    def season_ce(d, g):
        return d.groupby('season').V.apply(lambda v: u_inv(u(v.to_numpy(), g).mean(), g) - 1)
    def signflip_one_sided(x):
        x = np.asarray(x); sg = np.array(list(itertools.product([-1.0, 1.0], repeat=len(x))))
        return float(((sg * x).mean(1) >= x.mean() - 1e-12).mean())
    def boot_ci(x, B_=5000):
        rng = np.random.default_rng(1); x = np.asarray(x)
        m = x[rng.integers(0, len(x), (B_, len(x)))].mean(1); return np.percentile(m, [5, 95])
    tests = []
    def gap_test(label, model, g, rule, kbar=8, costs='base', tr='primary', a='prob', b='point', model_b=None):
        d = comp[(comp.tier == tr) & (comp.gamma == g) & (comp.rule == rule) & (comp.kbar == kbar) & (comp.costs == costs)]
        xa = season_ce(d[(d.model == model) & (d.form == a)], g)
        xb = season_ce(d[(d.model == (model_b or model)) & (d.form == b)], g)
        diff = (xa - xb).dropna()
        lo, hi = boot_ci(diff)
        slo, shi = signflip_ci(diff)
        tests.append(dict(test=label, seasons=len(diff), mean_diff_pp=100 * diff.mean(), ci90_lo=100 * lo, ci90_hi=100 * hi,
                          ci90_signflip_lo=100 * slo, ci90_signflip_hi=100 * shi, n_nonzero=int((diff.abs() > 1e-9).sum()),
                          p_one_sided=signflip_one_sided(diff), p_two_sided=min(1, 2 * min(signflip_one_sided(diff), signflip_one_sided(-diff))),
                          wins=int((diff > 1e-9).sum()), losses=int((diff < -1e-9).sum()),
                          diff_without_2024_pp=100 * diff.drop(2024, errors='ignore').mean()))
    gap_test('H1 primary: ens+R prob vs point, gamma 2, rule B', 'ens+R', 2, 'B')
    for g in GAMMAS:
        gap_test(f'H2/H3: ens+R prob vs point, gamma {g}, rule A', 'ens+R', g, 'A')
        gap_test(f'H3: ens+R prob vs point, gamma {g}, rule B', 'ens+R', g, 'B')
    gap_test('H4: ens prob vs point (no recalibration), gamma 2, rule B', 'ens', 2, 'B')
    gap_test('H4: ens+R prob vs ens prob, gamma 2, rule B', 'ens+R', 2, 'B', a='prob', b='prob', model_b='ens')
    gap_test('H6: M5 vs SARIMA (prob), gamma 2, rule B', 'm5', 2, 'B', a='prob', b='prob', model_b='sarima')
    gap_test('H6b: M5+R vs SARIMA+R (prob), gamma 2, rule B', 'm5+R', 2, 'B', a='prob', b='prob', model_b='sarima+R')
    TT = pd.DataFrame(tests); TT.to_csv(B + 'primary_test.csv', index=False, float_format='%.10g')

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(TT.round(3).to_string(index=False))
    base = S[(S.tier == 'primary') & (S.gamma == 2) & (S.costs == 'base') & (S.rule == 'B')]
    t = base.pivot(index='model', columns='form', values='ce_gain_pct'); t['gap'] = t['prob'] - t['point']
    t = t.join(base[base.form == 'prob'].set_index('model')[['mean_gain_pct', 'share_loss', 'cvar20_pct', 'vcr', 'share_hold']])
    print('\nPrimary panel, gamma 2, rule B, base costs: CE gain (% of May price) point vs probabilistic'); print(t.sort_values('prob', ascending=False).round(2).to_string())
    json.dump({'primary_test': TT.iloc[0].to_dict()}, open(B + 'key_numbers_decisions.json', 'w'), indent=1, default=float)
