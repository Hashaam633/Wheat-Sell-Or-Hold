"""
Local benchmark models for the wheat sell-or-hold paper (v3). Same output format as the Colab notebook.

Every model is re-estimated each April using only data up to that April (the "cutoff"); forecasts are then made at
every month from that April to the following March, each using price data up to its own origin month. Horizons 1-12,
19 quantile levels (0.05 ... 0.95), stored as price levels (rupees per 40 kg).

Models (pre-specified; nothing tuned on test results)
  rw        random walk: no change, spread = empirical quantiles of past h-month log changes (centred)
  snaive    seasonal naive: last year's change over the same months; spread = empirical quantiles of the rule's past errors
  m0        seasonal climatology: empirical quantiles of past changes from the same calendar month over h months
  sarima    SARIMA(1,1,0)(0,1,1)12 on log price, Gaussian
  ets       damped-trend seasonal exponential smoothing on log price (state space), Gaussian
  ecm       direct error-correction regressions: for each h, OLS of the log change on origin-month dummies, the gap between
            the log price and last month's log world price (US HRW in rupees), and 1- and 12-month log changes;
            spread = empirical residual quantiles
  msar      two-regime Markov-switching AR(1) on seasonally adjusted monthly log changes, simulated forward
  lgbm      LightGBM quantile regression, one global model per quantile across all series
  qrf       quantile regression forest, one global model across all series
  m5        regime-aware: SARIMA + pull toward the world price (+ support-price floor with reach q for wheat grain);
            the floor and its reach use only policy information public at the origin (floor_for; corrected in v1.1 and
            v1.6); the floor enters as an exact mixture (_mix_floor; corrected in v1.6)
Inputs: colab_panel.csv, monthly_panel.csv (world price, policy rate), data_extra/support_prices.csv
Output: benchmark/local_forecasts.csv.gz
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')
from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
H = 12
QL = np.round(np.arange(0.05, 0.951, 0.05), 2)
MIN_CONTEXT = 36
ORIGIN_START = pd.Period('2010-01', 'M')
SEED = 2026

panel = pd.read_csv(DERIVED + 'colab_panel.csv')
panel['ym'] = pd.PeriodIndex(pd.to_datetime(panel.ds), freq='M')
MP = pd.read_csv(DERIVED + 'monthly_panel.csv'); MP['ym'] = pd.PeriodIndex(MP['month'], freq='M'); MP = MP.set_index('ym')
LW1 = np.log(MP['world_hrw_rs40']).shift(1)           # last month's world price, known at the origin
SP = pd.read_csv(COMPILED + 'support_prices.csv').set_index('harvest_year')
SERIES = sorted(panel.unique_id.unique())
LOGP = {u: np.log(g.set_index('ym')['y']) for u, g in panel.groupby('unique_id')}
GROUP = {u: u.split('_')[0] + '_' + u.split('_')[1] for u in SERIES}   # wfp_wheat / fao_flour / pbs_whsl


def cutoffs():
    return [pd.Period(f'{y}-04', 'M') for y in range(2009, 2027)]


def origins_for(u, cut):
    s = LOGP[u]; end = min(cut + 11, s.index.max())
    out = []
    for o in pd.period_range(cut, end, freq='M'):
        if o < ORIGIN_START or o not in s.index:
            continue
        if (s.index <= o).sum() >= MIN_CONTEXT:
            out.append(o)
    return out


def changes(s, h):
    """Series of log changes over h months, indexed by the start month."""
    return (s.shift(-h) - s).dropna()


def emp_q(x, center=0.0):
    x = np.asarray(x, float)
    if len(x) < 3:
        return None
    return center + np.quantile(x, QL)


# ---------------------------------------------------------------- univariate models (fit once per cutoff and series)
def m_rw(s, cut, origins):
    tr = s.loc[:cut]; out = {}
    qh = []
    for h in range(1, H + 1):
        c = changes(tr, h); qh.append(np.quantile(c - c.mean(), QL))
    for o in origins:
        out[o] = np.array(qh)
    return out


def m_snaive(s, cut, origins):
    tr = s.loc[:cut]
    err = {}
    for h in range(1, H + 1):
        rule = (tr.shift(12 - h) - tr.shift(12))           # at origin t: log p[t+h-12] - log p[t-12]
        actual = tr.shift(-h) - tr
        e = (actual - rule).dropna(); err[h] = np.quantile(e, QL)
    out = {}
    for o in origins:
        so = s.loc[:o]; v = []
        for h in range(1, H + 1):
            a, b = o + h - 12, o - 12
            pt = so.get(a, np.nan) - so.get(b, np.nan) if (a in so.index and b in so.index) else 0.0
            v.append(pt + err[h])
        out[o] = np.array(v)
    return out


def m_m0(s, cut, origins):
    tr = s.loc[:cut]; out = {}
    for o in origins:
        v = []
        for h in range(1, H + 1):
            c = changes(tr, h); c = c[(c.index.month == o.month) & (c.index + h <= cut)]
            q = emp_q(c.values)
            v.append(q if q is not None else np.quantile(changes(tr, h), QL))
        out[o] = np.array(v)
    return out


def _gauss(model_res, so):
    f = model_res.get_forecast(H)
    m, sd = np.asarray(f.predicted_mean), np.sqrt(np.asarray(f.var_pred_mean))
    return (m - so.iloc[-1])[:, None] + sd[:, None] * norm.ppf(QL)[None, :]


def m_sarima(s, cut, origins):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    tr = s.loc[:cut]
    mod = SARIMAX(tr.values, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=True)
    res = mod.fit(disp=False, maxiter=1000)
    if not res.mle_retvals.get('converged', True):
        r2 = mod.fit(start_params=res.params, method='nm', maxiter=3000, disp=False)
        res = r2 if r2.llf >= res.llf else res
    out = {}
    for o in origins:
        so = s.loc[:o]
        out[o] = _gauss(res.apply(so.values, refit=False), so)
    return out


def m_ets(s, cut, origins):
    from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
    tr = s.loc[:cut]
    res = ExponentialSmoothing(tr.values, trend=True, damped_trend=True, seasonal=12).fit(disp=False, maxiter=1000)
    out = {}
    for o in origins:
        so = s.loc[:o]
        out[o] = _gauss(res.apply(so.values), so)
    return out


def _ecm_X(s, idx):
    gap = s.reindex(idx) - LW1.reindex(idx)
    d1 = (s - s.shift(1)).reindex(idx); d12 = (s - s.shift(12)).reindex(idx)
    M = np.eye(12)[np.asarray(idx.month) - 1]
    return np.column_stack([M, gap.values, d1.values, d12.values])


def m_ecm(s, cut, origins):
    tr = s.loc[:cut]; coefs, resq = {}, {}
    for h in range(1, H + 1):
        y = changes(tr, h); y = y[y.index + h <= cut]
        X = _ecm_X(s.loc[:cut], y.index); ok = ~np.isnan(X).any(1) & ~np.isnan(y.values)
        b, *_ = np.linalg.lstsq(X[ok], y.values[ok], rcond=None)
        coefs[h] = b; resq[h] = np.quantile(y.values[ok] - X[ok] @ b, QL)
    out = {}
    for o in origins:
        x = _ecm_X(s.loc[:o], pd.PeriodIndex([o]))[0]
        if np.isnan(x).any():
            x = np.nan_to_num(x)
        out[o] = np.array([x @ coefs[h] + resq[h] for h in range(1, H + 1)])
    return out


def m_msar(s, cut, origins, nsim=2000):
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    rng = np.random.default_rng(SEED)
    d = (s - s.shift(1)).dropna()
    tr = d.loc[:cut]
    seas = tr.groupby(tr.index.month).mean()
    z = lambda x: x - seas.reindex(x.index.month).values
    mod = MarkovAutoregression(z(tr).values, k_regimes=2, order=1, switching_ar=False, switching_variance=True)
    res = None
    for attempt in range(5):                                 # the fit can fail (no valid steady-state probabilities, SVD): retry
        try:                                                 # statsmodels draws the random start values from rng, not np.random
            res = mod.fit(search_reps=10 + 10 * attempt, disp=False, rng=SEED + attempt); break
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    if res is None:
        raise RuntimeError('MS-AR failed after 5 attempts')
    p = res.params; names = mod.param_names
    P = mod.regime_transition_matrix(p)[:, :, 0]            # P[i, j] = Pr(S_t = i | S_{t-1} = j)
    mu = np.array([p[names.index('const[0]')], p[names.index('const[1]')]])
    sig = np.sqrt([p[names.index('sigma2[0]')], p[names.index('sigma2[1]')]])
    phi = p[names.index('ar.L1')]
    out = {}
    for o in origins:
        do = d.loc[:o]; zz = z(do).values
        f = MarkovAutoregression(zz, k_regimes=2, order=1, switching_ar=False, switching_variance=True).filter(p)
        pr = np.asarray(f.filtered_marginal_probabilities)[-1]
        S = (rng.random(nsim) > pr[0]).astype(int)
        prev = np.full(nsim, zz[-1]); cum = np.zeros(nsim); qs = []
        for h in range(1, H + 1):
            S_new = (rng.random(nsim) > P[0, S]).astype(int)
            e = mu[S_new] + phi * (prev - mu[S]) + sig[S_new] * rng.standard_normal(nsim)
            mon = (o + h).month
            cum = cum + e + seas.get(mon, 0.0)
            qs.append(np.quantile(cum, QL)); prev, S = e, S_new
        out[o] = np.array(qs)
    return out


def season_of(p):
    return p.year if p.month >= 4 else p.year - 1


def floor_state(o):
    """Support price and reach in seasons with routine provincial procurement (up to 2023): used to estimate f."""
    s = season_of(o)
    if s <= 2023 and s in SP.index and not pd.isna(SP.loc[s, 'support_price_rs40']):
        return float(SP.loc[s, 'support_price_rs40']), 1.0
    return np.nan, 0.0


UNCERTAIN_REACH = float(os.environ.get('M5_UNCERTAIN_REACH', 0.5))


def floor_for(o, t):
    """Floor price F and reach q that a forecaster at origin o applies to target month t, using only policy information
    public by the end of month o (corrected in v1.1; v1.0 used the 2026 reach of 0.16, known only in July 2026).
      seasons up to 2023: routine procurement at the announced support price -> q = 1 at all horizons;
      April 2024: Rs3,900 announced, procurement delayed and Punjab's target cut from over 4 to about 2 Mt
        (Dawn, 1 May 2024) -> q = UNCERTAIN_REACH for targets in the 2024 season;
      from May 2024: Punjab had said it would not buy (The News, 9 May 2024) -> no floor;
      2025 season: no support price -> no floor;
      October-November 2025: Rs3,500 announced 19-20 Oct 2025 for the 2026 crop, reach unknown in advance
        -> q = UNCERTAIN_REACH for targets in the 2026 season (April 2026 - March 2027) only;
      December 2025 - January 2026: the federal government said it would leave wheat procurement and set no support
        price, as an IMF condition (ProPakistani, 24 Dec 2025; The Express Tribune, 26 Dec 2025) -> no floor
        (corrected in v1.6; v1.1-v1.5 kept q = UNCERTAIN_REACH at these two origins);
      from February 2026: an 'illustrative' procurement price of Rs3,500 for the 2026 harvest, with current procurement
        arrangements kept for one year (Profit, 22 Feb 2026) -> q = UNCERTAIN_REACH for targets in the 2026 season."""
    so, st = season_of(o), season_of(t)
    if so <= 2023:
        return floor_state(o)
    if o == pd.Period('2024-04', 'M') and st == 2024:
        return 3900.0, UNCERTAIN_REACH
    if o in (pd.Period('2025-12', 'M'), pd.Period('2026-01', 'M')):
        return np.nan, 0.0
    if o >= pd.Period('2025-10', 'M') and st == 2026:
        return 3500.0, UNCERTAIN_REACH
    return np.nan, 0.0


def m_m5(s, cut, origins, uid, sar, use_gap=True, use_floor=True):
    tr = s.loc[:cut]
    gap = (s - LW1.reindex(s.index)).dropna()
    rows = []
    for h in range(1, H + 1):
        c = changes(tr, h); c = c[c.index + h <= cut]
        mm = c.groupby(c.index.month).transform('mean')
        ex = ((c - mm) / h)
        g = gap.reindex(c.index)
        rows.append(pd.DataFrame({'ex': ex.values, 'g': g.values}))
    R = pd.concat(rows).dropna()
    gbar = gap.loc[:cut].mean()
    beta = float(((R.g - gbar) * R.ex).sum() / ((R.g - gbar) ** 2).sum())
    grain = not uid.startswith('fao_flour')
    if grain:
        rat = []
        for o in tr.index:
            F, q = floor_state(o)
            if q == 1.0 and not np.isnan(F):
                rat.append(np.exp(tr[o]) / F)
        f = np.percentile(rat, 5) if len(rat) > 12 else 0.95
    out = {}
    for o in origins:
        Q = sar[o].copy()
        if use_gap:
            Q = Q + (np.arange(1, H + 1) * beta * (gap.get(o, gbar) - gbar))[:, None]
        if grain and use_floor:
            new = []
            for i in range(H):
                F, q = floor_for(o, o + i + 1)
                new.append(_mix_floor(Q[i], np.log(f * F) - s[o], q) if (q > 0 and not np.isnan(F)) else Q[i])
            Q = np.vstack(new)
        out[o] = Q
    return out


def _qfun(Qrow, tau):
    """Quantile function through the 19 quantiles: linear between them, extended linearly beyond 0.05 and 0.95."""
    tau = np.atleast_1d(np.asarray(tau, float)); out = np.interp(tau, QL, Qrow)
    lo, hi = tau < QL[0], tau > QL[-1]
    out[lo] = Qrow[0] + (tau[lo] - QL[0]) * (Qrow[1] - Qrow[0]) / (QL[1] - QL[0])
    out[hi] = Qrow[-1] + (tau[hi] - QL[-1]) * (Qrow[-1] - Qrow[-2]) / (QL[-1] - QL[-2])
    return out


def _cdf_at(Qrow, x):
    """Inverse of _qfun: the probability that the forecast is at or below x (between 0 and 1)."""
    if x <= Qrow[0]:
        sl = (Qrow[1] - Qrow[0]) / (QL[1] - QL[0])
        return float(np.clip(QL[0] + (x - Qrow[0]) / sl, 0.0, 1.0)) if sl > 0 else 0.0
    if x >= Qrow[-1]:
        sl = (Qrow[-1] - Qrow[-2]) / (QL[-1] - QL[-2])
        return float(np.clip(QL[-1] + (x - Qrow[-1]) / sl, 0.0, 1.0)) if sl > 0 else 1.0
    return float(np.interp(x, Qrow, QL))


def _mix_floor(Qrow, L, q):
    """Quantiles of the mixture that puts weight q on the floored forecast max(Y, L) and 1 - q on the forecast Y (Equation 1).
    With p = P(Y <= L): levels at or above p keep the quantile of Y; levels from (1 - q)p to p sit at the floor L;
    lower levels take the quantile of Y at level tau / (1 - q). q = 1 gives max(Q, L); q = 0 or a floor below the
    forecast leaves the quantiles unchanged.
    Corrected in v1.6: up to v1.5 the mixture was approximated by interpolating between point masses, which pulled the
    quantiles above the median toward the median even when the floor did not bind."""
    Qrow = np.sort(np.asarray(Qrow, float))
    if q >= 1.0:
        return np.maximum(Qrow, L)
    p = _cdf_at(Qrow, L)
    out = Qrow.copy()
    out[(QL < p) & (QL >= (1 - q) * p)] = L
    low = QL < (1 - q) * p
    if low.any():
        out[low] = _qfun(Qrow, QL[low] / (1 - q))
    return out


UNI = {'rw': m_rw, 'snaive': m_snaive, 'm0': m_m0, 'sarima': m_sarima, 'ets': m_ets, 'ecm': m_ecm, 'msar': m_msar}


def run_series(u):
    s = LOGP[u]; rows = []; fails = []
    for cut in cutoffs():
        orgs = origins_for(u, cut)
        if not orgs or (s.index <= cut).sum() < MIN_CONTEXT:
            continue
        res = {}
        for name, fn in UNI.items():
            try:
                res[name] = fn(s, cut, orgs)
            except Exception as e:
                fails.append((u, str(cut), name, repr(e)[:120]))
        if 'sarima' in res:
            try:
                res['m5'] = m_m5(s, cut, orgs, u, res['sarima'])
            except Exception as e:
                fails.append((u, str(cut), 'm5', repr(e)[:120]))
        for name, d in res.items():
            for o, Q in d.items():
                Q = np.sort(Q, axis=1)
                lvl = np.exp(s[o] + Q)
                for h in range(H):
                    rows.append((name, u, str(o), h + 1, *lvl[h]))
    return rows, fails


# ---------------------------------------------------------------- global ML models (one fit per cutoff, all series)
def features(u, o):
    s = LOGP[u]
    g = lambda k: s.get(o, np.nan) - s.get(o - k, np.nan) if (o - k) in s.index else np.nan
    gap = s.get(o, np.nan) - LW1.get(o, np.nan)
    pol = MP['policy_rate_pct'].get(o, np.nan)
    return [o.month, g(1), g(3), g(6), g(12), gap, pol, ['fao_flour', 'pbs_whsl', 'wfp_wheat'].index(GROUP[u])]


FEAT = ['month', 'd1', 'd3', 'd6', 'd12', 'gap', 'policy', 'group']


def train_rows(cut):
    X, y = [], []
    for u in SERIES:
        s = LOGP[u]
        for o in s.index[(s.index <= cut)]:
            if (s.index <= o).sum() < 13:
                continue
            f = features(u, o)
            for h in range(1, H + 1):
                if o + h <= cut and (o + h) in s.index:
                    X.append(f + [h]); y.append(s[o + h] - s[o])
    return np.array(X, float), np.array(y, float)


def run_global():
    import lightgbm as lgb
    from quantile_forest import RandomForestQuantileRegressor
    rows = []
    for cut in cutoffs():
        t0 = time.time()
        X, y = train_rows(cut)
        tgt = [(u, o) for u in SERIES for o in origins_for(u, cut)]
        if not tgt:
            continue
        Xt = np.array([features(u, o) + [h] for (u, o) in tgt for h in range(1, H + 1)], float)
        preds = {}
        q_l = np.zeros((len(Xt), len(QL)))
        for j, q in enumerate(QL):
            m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=200, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=SEED, verbose=-1)
            m.fit(np.nan_to_num(X, nan=-999), y)
            q_l[:, j] = m.predict(np.nan_to_num(Xt, nan=-999))
        preds['lgbm'] = q_l
        qrf = RandomForestQuantileRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5, random_state=SEED, n_jobs=2)
        qrf.fit(np.nan_to_num(X, nan=-999), y)
        preds['qrf'] = np.asarray(qrf.predict(np.nan_to_num(Xt, nan=-999), quantiles=list(QL)))
        for name, P in preds.items():
            P = np.sort(P, axis=1); k = 0
            for (u, o) in tgt:
                base = LOGP[u][o]
                for h in range(1, H + 1):
                    rows.append((name, u, str(o), h, *np.exp(base + P[k]))); k += 1
        print(f'  global models, cutoff {cut}: {len(y):,} training rows, {len(tgt)} origins, {time.time() - t0:.0f}s', flush=True)
    return rows


if __name__ == '__main__':
    t0 = time.time()
    cols = ['model', 'unique_id', 'origin', 'h'] + [f'q{q:.2f}' for q in QL]
    with Pool(2) as pool:
        res = pool.map(run_series, SERIES)
    rows = [r for rr, _ in res for r in rr]; fails = [f for _, ff in res for f in ff]
    print(f'univariate models done in {time.time() - t0:.0f}s; failures: {len(fails)}')
    for f in fails[:20]:
        print('  ', f)
    rows += run_global()
    D = pd.DataFrame(rows, columns=cols)
    D.to_csv(RESULTS + 'local_forecasts.csv.gz', index=False, compression='gzip', float_format='%.3f')
    pd.DataFrame(fails, columns=['unique_id', 'cutoff', 'model', 'error']).to_csv(RESULTS + 'local_failures.csv', index=False)
    print(D.groupby('model').agg(rows=('h', 'size'), series=('unique_id', 'nunique'), first=('origin', 'min'), last=('origin', 'max')).to_string())
    print(f'total {time.time() - t0:.0f}s')
