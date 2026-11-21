"""
Decision type D (pre-analysis plan, Section 5, robustness): dynamic programming on coherent price paths.

At each month km = 0..7 (May..December) the farmer simulates N price paths for the months km+1..January from the forecast
issued that month: marginal quantile functions per horizon (19 quantiles -> 100 levels, as in rules A-C) are joined by a
Gaussian copula. The copula correlation between horizons is estimated from the model's own past forecast errors: normal scores
of PIT values, same origin and market, pairs of horizons, using only forecasts whose outcomes were known before May of the
season (primary or validation panel, all markets of that panel pooled). If fewer than 40 pairs are available (seasons 2013-2014),
the random-walk structure corr(h, h') = sqrt(min(h, h') / max(h, h')) is used. A second variant uses that structure throughout.
The optimal stopping rule on the paths is found by Longstaff-Schwartz least squares (cubic in the log price ratio), with CRRA
utility of the present value V as the payoff. The farmer sells now if the utility of selling now is at least the estimated
continuation value; otherwise she waits and repeats the exercise next month with the new forecast. Sale by January at the latest.
With a point forecast every path is identical and rule D reduces exactly to rule B.

Output: benchmark/rule_d_rows.csv.gz, rule_d_tests.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm

from decisions_v3 import load, build_cases, to100, u, u_inv, value, decide, tier, GAMMAS, COSTS, TAU

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
MODELS = ['ens+R', 'ens', 'm5+R', 'm5', 'sarima+R', 'sarima', 'm0+R']
NPATH = 2000
HMAX = 8


def rw_corr(H=HMAX):
    h = np.arange(1, H + 1)
    return np.sqrt(np.minimum.outer(h, h) / np.maximum.outer(h, h))


def nearest_corr(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    C2 = V @ np.diag(np.maximum(w, 1e-6)) @ V.T
    d = np.sqrt(np.diag(C2)); return C2 / np.outer(d, d)


def corr_tables():
    SR = pd.read_csv(B + 'scored_rows.csv.gz', usecols=['model', 'unique_id', 'o', 'h', 'pit', 'tier'])
    SR = SR[SR.model.isin(MODELS) & SR.tier.isin(['primary', 'validation']) & (SR.h <= HMAX)].dropna(subset=['pit'])
    SR['o'] = pd.PeriodIndex(SR.o, freq='M')
    SR['z'] = norm.ppf(SR.pit.clip(0.005, 0.995))
    SR['known'] = SR.o.dt.year * 12 + SR.o.dt.month - 1 + SR.h       # month index when the outcome becomes known
    tab = {}
    for (m, tr), g in SR.groupby(['model', 'tier']):
        for s in range(2013, 2026):
            w = g[g.known < s * 12 + 4]                              # outcomes known before May of season s
            W = w.pivot_table(index=['unique_id', 'o'], columns='h', values='z')
            W = W.reindex(columns=range(1, HMAX + 1))
            npair = W.notna().astype(int).T @ W.notna().astype(int)
            if W.empty or npair.to_numpy().min() < 40:
                tab[(m, tr, s)] = (rw_corr(), 'random-walk structure')
            else:
                C = W.corr(min_periods=40).to_numpy()
                C = np.where(np.isnan(C), rw_corr(), C)
                tab[(m, tr, s)] = (nearest_corr(C), f'estimated ({int(npair.to_numpy().min())}+ pairs)')
    return tab


def qinv(q100, U):
    """q100: (H, 100) log-ratio quantiles at TAU; U: (N, H) uniforms -> (N, H) log ratios."""
    Uc = np.clip(U, TAU[0], TAU[-1])
    return np.column_stack([np.interp(Uc[:, j], TAU, q100[j]) for j in range(q100.shape[0])])


def lsm_continue_value(R_km, km, X, g, r, lam):
    """Longstaff-Schwartz continuation value at month km for log-ratio paths X (N, H)."""
    N, H = X.shape
    pay = np.column_stack([u(value(R_km * np.exp(X[:, j]), km + j + 1, r, lam), g) for j in range(H)])
    cash = pay[:, -1].copy()
    for j in range(H - 2, -1, -1):
        x = X[:, j]
        A = np.column_stack([np.ones(N), x, x ** 2, x ** 3])
        beta, *_ = np.linalg.lstsq(A, cash, rcond=None)
        cont = A @ beta
        stop = pay[:, j] >= cont
        cash = np.where(stop, pay[:, j], cash)
    return cash.mean()


def decide_d(R, F, g, r, lam, Cfull, seed):
    for km in range(8):
        H = 8 - km
        now = u(np.array([value(R[km], km, r, lam)]), g)[0]
        q100 = to100(F[km])                                           # (H, 100)
        rng = np.random.default_rng(seed * 10 + km)
        L = np.linalg.cholesky(nearest_corr(Cfull[:H, :H]))
        Z = rng.standard_normal((NPATH, H)) @ L.T
        X = qinv(q100, norm.cdf(Z))
        if now >= lsm_continue_value(R[km], km, X, g, r, lam) - 1e-12:
            return value(R[km], km, r, lam), km
    return value(R[8], 8, r, lam), 8


if __name__ == '__main__':
    D, price, pol = load()
    D = D[D.model.isin(MODELS)].copy()
    cases = build_cases(D, price)
    tab = corr_tables()
    src = pd.Series({k: v[1] for k, v in tab.items()})
    print(src.groupby(level=[0, 1]).value_counts().head(30))
    spread, lam = COSTS['base']
    rows = []
    for i, ((m, uid, s), (R, F)) in enumerate(sorted(cases.items())):
        tr = tier(uid)
        if tr not in ('primary', 'validation') or not 2013 <= s <= 2025:
            continue
        r = pol.get(pd.Period(f'{s}-05', 'M')) + spread
        DIST = {'point': {km: np.exp(F[km][:, 9:10]) for km in F}, 'prob': {km: np.exp(to100(F[km])) for km in F}}
        seed = int(sum(ord(c) for c in uid) * 10000 + s)                   # deterministic across runs
        for g in GAMMAS:
            Vd, kd = decide_d(R, F, g, r, lam, tab[(m, tr, s)][0], seed)
            Vr, kr = decide_d(R, F, g, r, lam, rw_corr(), seed)
            Vb, kb = decide(R, DIST, g, r, lam, 'prob', 'B', 8)
            Vp, kp = decide(R, DIST, g, r, lam, 'point', 'B', 8)
            rows.append((m, g, uid, tr, s, Vd, kd, Vr, kr, Vb, kb, Vp, kp))
    X = pd.DataFrame(rows, columns=['model', 'gamma', 'unique_id', 'tier', 'season', 'V_D', 'k_D', 'V_Drw', 'k_Drw', 'V_Bprob', 'k_Bprob', 'V_point', 'k_point'])
    X.to_csv(B + 'rule_d_rows.csv.gz', index=False, compression='gzip', float_format='%.8f')

    def signflip(x):
        x = np.asarray(x); sg = np.array(list(itertools.product([-1.0, 1.0], repeat=len(x))))
        return float(((sg * x).mean(1) >= x.mean() - 1e-12).mean())

    def ce(v, g):
        return 100 * (u_inv(u(np.asarray(v), g).mean(), g) - 1)

    out = []
    for (tr, m, g), d in X.groupby(['tier', 'model', 'gamma']):
        sc = {c: d.groupby('season')[c].apply(lambda v: ce(v, g)) for c in ['V_D', 'V_Drw', 'V_Bprob', 'V_point']}
        row = dict(tier=tr, model=m, gamma=g, ce_D=ce(d.V_D, g), ce_Drw=ce(d.V_Drw, g), ce_Bprob=ce(d.V_Bprob, g), ce_point=ce(d.V_point, g),
                   same_month_D_vs_Bprob=(d.k_D == d.k_Bprob).mean())
        for lab, a, b in [('D_minus_point', 'V_D', 'V_point'), ('D_minus_Bprob', 'V_D', 'V_Bprob'), ('Drw_minus_point', 'V_Drw', 'V_point')]:
            x = (sc[a] - sc[b]).dropna()
            row[lab + '_pp'] = x.mean(); row[lab + '_p'] = signflip(x); row[lab + '_wl'] = f'{int((x > 1e-9).sum())}/{int((x < -1e-9).sum())}'
        out.append(row)
    T = pd.DataFrame(out); T.to_csv(B + 'rule_d_tests.csv', index=False, float_format='%.10g')
    pd.set_option('display.width', 260); pd.set_option('display.max_columns', 30)
    print(T.round(3).to_string(index=False))
