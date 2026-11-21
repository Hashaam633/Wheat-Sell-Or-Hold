"""
Wheat sell-or-hold study, step 2: probabilistic price forecasts and a cost-aware decision backtest (v1).

Input : monthly_panel.csv        (made by build_monthly_panel.py)
Output: forecasts_v1.csv         quantile forecasts of log(P[May+k] / P[May]), model x season x k x tau
        decisions_v1.csv         chosen holding month and realised gain, model x rule x cost scenario x season
        scores_forecast_v1.csv   CRPS and interval coverage, model x period
        summary_decisions_v1.csv decision value, model x rule x cost scenario x period
        live_2026_v1.csv         what each model/rule would have advised in May 2026 (season still running)

DESIGN (fixed before looking at test results; nothing below was tuned on the test seasons)
  Decision problem   Harvest 1 unit in May of season s. Sell in May (k = 0) or hold and sell k = 1..8 months later (Jun..Jan).
                     Net gain from holding, as a share of the May price:
                       G_k = R_k (1 - lam)^k - c k - (1 + rm)^k,   R_k = P[May+k] / P[May],   rm = (1 + r)^(1/12) - 1
  Prices             Punjab = mean of WFP retail wheat in Lahore and Multan, Rs per 40 kg (a retail proxy for the farm-gate price).
  Cost of capital    r = SBP policy rate in the harvest month + spread; spread 0 / 5 / 10 percentage points for low / base / high.
                     lam (monthly physical loss) and c (monthly storage cost) are ASSUMPTIONS: 0.2 / 0.3 / 0.5 per cent.
                     The rate is known at the decision date, so there is no look-ahead.
  Forecast target    y_k = log R_k, k = 1..8. Each model returns 100 quantiles (tau = 0.005..0.995) per horizon.
  Models             M0  seasonal empirical distribution of past y_k (climatology)
                     M1  SARIMA(1,1,0)(0,1,1)12 on log price, Gaussian predictive distribution
                     M2  linear quantile regression, pooled over k: y = a0 + a1 k + a2 k^2 + k (b1 z_yoy + b2 z_mom3)
                     M3  gradient-boosted quantile regression on (k, yoy, mom3), shallow trees, small learning rate
                     Features at the May origin: yoy = log P[May] - log P[May-12];  mom3 = log P[May] - log P[Feb].
  Rules              D0 sell in May | D1 always sell in December (k = 7)
                     D2 central forecast (median) | D3 expected gain | D4 risk-averse: (1-phi) E[G] + phi CVaR20[G], phi = .25/.5/.75
                     D5 hold only if P(G > 0) >= 0.70, then pick the best expected gain | O hindsight oracle
                     A hold is chosen only if the rule's score is positive; otherwise sell in May.
  Evaluation         Expanding window; first test season 2013 (7 prior seasons); no information after May of the test season is used.
                     Test seasons 2013-2025 (13 complete). 2026 is still running and is reported separately (live_2026_v1.csv).
"""
import itertools
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
SERIES = os.environ.get('WHEAT_SERIES', 'wfp_punjab_rs40')          # price series; Karachi / Peshawar are robustness checks
TAG = os.environ.get('WHEAT_TAG', 'v1')                              # suffix for the output files
KMAX = int(os.environ.get('WHEAT_KMAX', 8))                          # longest holding period in months (8 = January; 10 = March as a robustness check)
FIRST_TRAIN, FIRST_TEST, LAST_COMPLETE, LIVE = 2006, 2013, 2025, 2026
J = 100
TAUS = (np.arange(J) + 0.5) / J                         # fine grid used by decisions and scores
TAUS_FIT = np.arange(0.025, 0.9751, 0.025)              # levels at which M2 and M3 are fitted (39)
K = np.arange(1, KMAX + 1)
SCEN = {'low': dict(spread=0.00, lam=0.002, c=0.002),
        'base': dict(spread=0.05, lam=0.003, c=0.003),
        'high': dict(spread=0.10, lam=0.005, c=0.005)}
RULES = ['D2', 'D3', 'D4_25', 'D4_50', 'D4_75', 'D5']
MODELS = os.environ.get('WHEAT_MODELS', 'M0,M1,M2,M3').split(',')
PERIODS = {'R1 (2013-2023)': list(range(2013, 2024)), 'Transition (2024)': [2024], 'R2 (2025)': [2025],
           'All (2013-2025)': list(range(2013, 2026))}


def per(y, m):
    return pd.Period(f'{y}-{m:02d}', freq='M')


def regime(s):
    return 'R1 support price' if s <= 2023 else 'Transition 2024' if s == 2024 else 'R2 deregulated' if s == 2025 else 'R3 partial reversal'


# ----------------------------------------------------------------------------- data
P = pd.read_csv(DERIVED + 'monthly_panel.csv')
P['month'] = pd.PeriodIndex(P['month'], freq='M')
P = P.set_index('month')
LP = np.log(P[SERIES].dropna())                           # log retail price, Rs per 40 kg (Punjab = mean of Lahore and Multan)
DROPPED_LAST = None
if len(LP) > 1 and abs(LP.iloc[-1] - LP.iloc[-2]) < 1e-12:  # the newest WFP month is often a carry-forward of the month before: drop it
    DROPPED_LAST = str(LP.index[-1]); LP = LP.iloc[:-1]
POL = P['policy_rate_pct'] / 100


def target(lp, s, k):
    d = lp.to_dict(); a, b = per(s, 5), per(s, 5) + k
    return d[b] - d[a] if (a in d and b in d) else np.nan


def features(lp, s):
    d = lp.to_dict(); p0 = per(s, 5)
    yoy = d[p0] - d[p0 - 12] if (p0 - 12) in d else np.nan
    mom3 = d[p0] - d[p0 - 3] if (p0 - 3) in d else np.nan
    return yoy, mom3


def train_table(lp, s, need_feats):
    """One row per (training season, k). Training seasons are strictly earlier than s."""
    rows = []
    for sp in range(FIRST_TRAIN, s):
        yoy, mom3 = features(lp, sp)
        if need_feats and (np.isnan(yoy) or np.isnan(mom3)):
            continue
        for k in K:
            y = target(lp, sp, k)
            if not np.isnan(y):
                rows.append((sp, k, y, yoy, mom3))
    return pd.DataFrame(rows, columns=['season', 'k', 'y', 'yoy', 'mom3'])


def to_fine(qfit):
    """(KMAX, 39) quantiles at TAUS_FIT -> sorted (no crossing) and interpolated to the 100-point grid."""
    q = np.sort(qfit, axis=1)
    return np.vstack([np.interp(TAUS, TAUS_FIT, q[i]) for i in range(q.shape[0])])


# ----------------------------------------------------------------------------- models: each returns (KMAX, J) quantiles of log R_k
def fc_M0(lp, s):
    tr = train_table(lp, s, False)
    return np.vstack([np.quantile(tr.loc[tr.k == k, 'y'].values, TAUS) for k in K])


FALLBACKS = {'M1': 0, 'M1_polished': 0}


def _fit_sarima(y):
    """L-BFGS first; if it stops short of its tolerance, polish with Nelder-Mead from that point and keep the better likelihood."""
    mod = SARIMAX(y, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12), trend='n', enforce_stationarity=False, enforce_invertibility=True)
    res = mod.fit(disp=False, maxiter=1000)
    if not res.mle_retvals.get('converged', True):
        FALLBACKS['M1_polished'] += 1
        res2 = mod.fit(start_params=res.params, method='nm', maxiter=3000, disp=False)
        if np.isfinite(res2.llf) and res2.llf >= res.llf:
            res = res2
    return res


def fc_M1(lp, s):
    y = lp.loc[:per(s, 5)].values
    try:
        res = _fit_sarima(y)
        f = res.get_forecast(KMAX)
        m, se = np.asarray(f.predicted_mean), np.sqrt(np.asarray(f.var_pred_mean))
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(se)) and np.all(se > 0)):
            raise RuntimeError('non-finite forecast')
    except Exception:
        FALLBACKS['M1'] += 1
        tr = train_table(lp, s, False)                      # fall back to a Gaussian climatology
        mu = np.array([tr.loc[tr.k == k, 'y'].mean() for k in K]); sd = np.array([tr.loc[tr.k == k, 'y'].std(ddof=1) for k in K])
        return mu[:, None] + sd[:, None] * norm.ppf(TAUS)[None, :]
    return (m - y[-1])[:, None] + se[:, None] * norm.ppf(TAUS)[None, :]


def _std(tr, yoy, mom3):
    sea = tr.drop_duplicates('season')
    mu, sd = sea[['yoy', 'mom3']].mean(), sea[['yoy', 'mom3']].std(ddof=1)
    return (tr.yoy - mu.yoy) / sd.yoy, (tr.mom3 - mu.mom3) / sd.mom3, (yoy - mu.yoy) / sd.yoy, (mom3 - mu.mom3) / sd.mom3


def fc_M2(lp, s):
    tr = train_table(lp, s, True)
    yoy, mom3 = features(lp, s)
    z1, z2, t1, t2 = _std(tr, yoy, mom3)
    X = np.column_stack([np.ones(len(tr)), tr.k, tr.k ** 2, tr.k * z1, tr.k * z2])
    Xt = np.column_stack([np.ones(KMAX), K, K ** 2, K * t1, K * t2])
    q = np.empty((KMAX, len(TAUS_FIT)))
    for j, tau in enumerate(TAUS_FIT):
        m = QuantileRegressor(quantile=tau, alpha=0.0, fit_intercept=False, solver='highs').fit(X, tr.y.values)
        q[:, j] = Xt @ m.coef_
    return to_fine(q)


def fc_M3(lp, s):
    tr = train_table(lp, s, True)
    yoy, mom3 = features(lp, s)
    X = tr[['k', 'yoy', 'mom3']].values
    Xt = np.column_stack([K, np.full(KMAX, yoy), np.full(KMAX, mom3)])
    q = np.empty((KMAX, len(TAUS_FIT)))
    for j, tau in enumerate(TAUS_FIT):
        m = GradientBoostingRegressor(loss='quantile', alpha=tau, n_estimators=80, learning_rate=0.05, max_depth=2,
                                      min_samples_leaf=6, subsample=0.8, random_state=0).fit(X, tr.y.values)
        q[:, j] = m.predict(Xt)
    return to_fine(q)


FC = {'M0': fc_M0, 'M1': fc_M1, 'M2': fc_M2, 'M3': fc_M3}

# ----------------------------------------------------------------------------- decision layer
def gain(R, k, r, lam, c):
    rm = (1 + r) ** (1 / 12) - 1
    return R * (1 - lam) ** k - c * k - (1 + rm) ** k


def cost_params(s, scen):
    return POL[per(s, 5)] + SCEN[scen]['spread'], SCEN[scen]['lam'], SCEN[scen]['c']


def choose(Q, rule, r, lam, c):
    """Q: (KMAX, J) quantiles of log R_k. Returns k* in 0..KMAX (0 = sell in May)."""
    G = gain(np.exp(Q), K[:, None], r, lam, c)
    E, CV = G.mean(1), G[:, :int(0.2 * J)].mean(1)
    if rule == 'D2':
        score = gain(np.exp(0.5 * (Q[:, J // 2 - 1] + Q[:, J // 2])), K, r, lam, c)
    elif rule == 'D3':
        score = E
    elif rule.startswith('D4'):
        phi = int(rule.split('_')[1]) / 100
        score = (1 - phi) * E + phi * CV
    elif rule == 'D5':
        score = np.where((G > 0).mean(1) >= 0.70, E, -np.inf)
    else:
        raise ValueError(rule)
    j = int(np.argmax(score))
    return j + 1 if score[j] > 0 else 0


def realised(lp, s, k, scen):
    if k == 0:
        return 0.0
    r, lam, c = cost_params(s, scen)
    y = target(lp, s, k)
    return np.nan if np.isnan(y) else float(gain(np.exp(y), k, r, lam, c))


def oracle(lp, s, scen):
    g = [realised(lp, s, k, scen) for k in K]
    if np.all(np.isnan(g)):
        return 0, np.nan
    j = int(np.nanargmax(g))
    return (j + 1, g[j]) if g[j] > 0 else (0, 0.0)


# ----------------------------------------------------------------------------- scoring
def crps_cov(Q, y):
    """Q: (J,) quantiles for one horizon, y: realised log ratio. CRPS ~ 2 * mean pinball over the tau grid."""
    d = y - Q
    pin = np.maximum(TAUS * d, (TAUS - 1) * d)
    q = lambda t: np.interp(t, TAUS, Q)
    return 2 * pin.mean(), float(q(0.10) <= y <= q(0.90)), float(q(0.05) <= y <= q(0.95)), float(np.interp(y, Q, TAUS, left=0.0, right=1.0))


def cvar20(x):
    x = np.sort(np.asarray(x, dtype=float)); n = max(1, math.ceil(0.2 * len(x)))
    return x[:n].mean()


def boot_ci(x, B=4000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, 5), np.percentile(m, 95)


if __name__ == '__main__':
    # --------------------------------------------------------------- 1. pre-test look at the seasonal profile (training seasons only)
    tr0 = train_table(LP, FIRST_TEST, False)
    prof = tr0.groupby('k').y.agg(['mean', 'std', 'count'])
    print('Seasonal profile of log(P[May+k]/P[May]) in the first training window (2006-2012):')
    print(prof.round(3).T.to_string())

    # --------------------------------------------------------------- 2. forecasts for every test season
    test_seasons = list(range(FIRST_TEST, LAST_COMPLETE + 1))
    FCS = {m: {} for m in MODELS}
    for s in test_seasons + [LIVE]:
        for m in MODELS:
            FCS[m][s] = FC[m](LP, s)
    print(f'\nSeries: {SERIES} | newest month dropped as a carry-forward: {DROPPED_LAST} | last month used: {LP.index[-1]}')
    print('M1 SARIMA: fits polished after L-BFGS stopped short:', FALLBACKS['M1_polished'], '| fallbacks to Gaussian climatology:', FALLBACKS['M1'], '| fits made:', len(test_seasons) + 1)

    rows = []
    for m in MODELS:
        for s, Q in FCS[m].items():
            for i, k in enumerate(K):
                for j in range(J):
                    rows.append((m, s, int(k), round(TAUS[j], 3), Q[i, j]))
    pd.DataFrame(rows, columns=['model', 'season', 'k', 'tau', 'log_ratio_q']).to_csv(EXPLORATORY + f'forecasts_{TAG}.csv', index=False, float_format='%.5f')

    # --------------------------------------------------------------- 3. forecast scores
    sc = []
    for m in MODELS:
        for s in test_seasons:
            for i, k in enumerate(K):
                y = target(LP, s, k)
                crps, c80, c90, pit = crps_cov(FCS[m][s][i], y)
                sc.append((m, s, int(k), crps, c80, c90, pit))
    SC = pd.DataFrame(sc, columns=['model', 'season', 'k', 'crps', 'cov80', 'cov90', 'pit'])
    out = []
    for pname, ss in PERIODS.items():
        d = SC[SC.season.isin(ss)]
        for m in MODELS:
            x = d[d.model == m]
            out.append((pname, m, len(ss), x.crps.mean(), x.cov80.mean(), x.cov90.mean(), x.pit.mean()))
    FS = pd.DataFrame(out, columns=['period', 'model', 'seasons', 'crps_logpts', 'coverage80', 'coverage90', 'mean_pit'])
    FS.to_csv(EXPLORATORY + f'scores_forecast_{TAG}.csv', index=False, float_format='%.4f')
    hz = SC.assign(horizon=pd.cut(SC.k, [0, 3, 6, 8], labels=['k=1-3', 'k=4-6', 'k=7-8']))
    hz_tab = hz.groupby(['model', 'horizon'], observed=True).agg(crps_logpts=('crps', 'mean'), coverage80=('cov80', 'mean')).reset_index()
    hz_tab.to_csv(EXPLORATORY + f'scores_by_horizon_{TAG}.csv', index=False, float_format='%.4f')
    print('\nCRPS by horizon group, all test seasons (log points):')
    print(hz_tab.pivot(index='model', columns='horizon', values='crps_logpts').round(3).to_string())
    print('\nForecast scores (CRPS in log points, lower is better; nominal coverage 0.80 / 0.90; mean PIT should be near 0.5):')
    print(FS.pivot(index='model', columns='period', values='crps_logpts').round(3).reindex(columns=list(PERIODS)).to_string())
    print(FS.pivot(index='model', columns='period', values='coverage80').round(2).reindex(columns=list(PERIODS)).to_string())
    print(FS.pivot(index='model', columns='period', values='coverage90').round(2).reindex(columns=list(PERIODS)).to_string())
    print(FS.pivot(index='model', columns='period', values='mean_pit').round(2).reindex(columns=list(PERIODS)).to_string())

    # --------------------------------------------------------------- 4. decisions and realised gains
    dec = []
    for scen in SCEN:
        for s in test_seasons:
            r, lam, c = cost_params(s, scen)
            ko, go = oracle(LP, s, scen)
            base = dict(scenario=scen, season=s, regime=regime(s), r_annual=round(r, 4))
            dec.append({**base, 'model': '-', 'rule': 'D0', 'k': 0, 'gain': 0.0})
            dec.append({**base, 'model': '-', 'rule': 'D1', 'k': 7, 'gain': realised(LP, s, 7, scen)})
            dec.append({**base, 'model': '-', 'rule': 'O', 'k': ko, 'gain': go})
            for m in MODELS:
                for rule in RULES:
                    k = choose(FCS[m][s], rule, r, lam, c)
                    dec.append({**base, 'model': m, 'rule': rule, 'k': k, 'gain': realised(LP, s, k, scen)})
    D = pd.DataFrame(dec)
    D.to_csv(EXPLORATORY + f'decisions_{TAG}.csv', index=False, float_format='%.5f')

    summ = []
    for (scen, model, rule), g in D.groupby(['scenario', 'model', 'rule']):
        for pname, ss in PERIODS.items():
            x = g[g.season.isin(ss)].sort_values('season')
            o = D[(D.scenario == scen) & (D.rule == 'O') & D.season.isin(ss)].gain.mean()
            gv = x.gain.values
            lo, hi = boot_ci(gv)
            summ.append(dict(scenario=scen, model=model, rule=rule, period=pname, seasons=len(x),
                             mean_gain_pct=100 * gv.mean(), median_gain_pct=100 * np.median(gv),
                             share_loss=(gv < -1e-12).mean(), share_hold=(x.k.values > 0).mean(),
                             worst_pct=100 * gv.min(), cvar20_pct=100 * cvar20(gv),
                             vcr=(gv.mean() / o) if o > 1e-9 else np.nan,
                             ci90_lo_pct=100 * lo, ci90_hi_pct=100 * hi, mean_k=x.k.mean()))
    S = pd.DataFrame(summ)
    S.to_csv(EXPLORATORY + f'summary_decisions_{TAG}.csv', index=False, float_format='%.4f')

    # --------------------------------------------------------------- 4b. paired comparisons (exact sign-flip test on season-by-season differences)
    def signflip_p(d):
        d = np.asarray(d, dtype=float)
        if len(d) < 5 or np.allclose(d, 0):
            return np.nan
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d))))
        return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())

    pr = []
    for scen in SCEN:
        d1 = D[(D.scenario == scen) & (D.rule == 'D1')].set_index('season').gain
        for (model, rule), g in D[(D.scenario == scen) & (D.model != '-')].groupby(['model', 'rule']):
            g = g.set_index('season').gain
            for pname in ['R1 (2013-2023)', 'All (2013-2025)']:
                ss = PERIODS[pname]
                diff, hold = (g.loc[ss] - d1.loc[ss]).values, g.loc[ss].values
                lo, hi = boot_ci(diff)
                pr.append(dict(scenario=scen, model=model, rule=rule, period=pname, seasons=len(ss),
                               mean_diff_vs_D1_pct=100 * diff.mean(), ci90_lo=100 * lo, ci90_hi=100 * hi, p_vs_D1=signflip_p(diff),
                               mean_gain_pct=100 * hold.mean(), p_vs_D0=signflip_p(hold)))
    PR = pd.DataFrame(pr); PR.to_csv(EXPLORATORY + f'paired_tests_{TAG}.csv', index=False, float_format='%.4f')

    # --------------------------------------------------------------- 4c. mean-risk frontier over phi (0 = expected gain, 1 = pure CVaR20)
    fr = []
    for scen in SCEN:
        for m in MODELS:
            for phi in np.round(np.arange(0, 1.01, 0.1), 1):
                gs = []
                for s in test_seasons:
                    r, lam, c = cost_params(s, scen)
                    Q = FCS[m][s]; G = gain(np.exp(Q), K[:, None], r, lam, c)
                    score = (1 - phi) * G.mean(1) + phi * G[:, :int(0.2 * J)].mean(1)
                    j = int(np.argmax(score)); k = j + 1 if score[j] > 0 else 0
                    gs.append(realised(LP, s, k, scen))
                gs = np.array(gs)
                fr.append(dict(scenario=scen, model=m, phi=phi, mean_gain_pct=100 * gs.mean(), cvar20_pct=100 * cvar20(gs),
                               share_loss=(gs < -1e-12).mean(), share_hold=(gs != 0).mean()))
    FR = pd.DataFrame(fr); FR.to_csv(EXPLORATORY + f'frontier_{TAG}.csv', index=False, float_format='%.4f')
    print('\nMean-risk frontier, base scenario, all test seasons (mean gain % / CVaR20 %) by phi:')
    t = FR[FR.scenario == 'base'].assign(v=lambda d: d.mean_gain_pct.round(1).astype(str) + ' / ' + d.cvar20_pct.round(1).astype(str))
    print(t.pivot(index='model', columns='phi', values='v').to_string())

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    for scen in ['base', 'low', 'high']:
        print(f'\n=== Decision value, cost scenario: {scen}  (mean gain, % of the May price; VCR = share of hindsight gain captured)')
        t = S[(S.scenario == scen)].copy()
        t['label'] = t.model + ' ' + t.rule
        t.loc[t.model == '-', 'label'] = t.rule
        a = t.pivot(index='label', columns='period', values='mean_gain_pct').reindex(columns=list(PERIODS))
        b = t.pivot(index='label', columns='period', values='vcr').reindex(columns=list(PERIODS))
        c_ = t[t.period == 'All (2013-2025)'].set_index('label')[['share_loss', 'cvar20_pct', 'worst_pct', 'mean_k', 'ci90_lo_pct', 'ci90_hi_pct']]
        res = a.round(1).join(b[['All (2013-2025)']].round(2).rename(columns={'All (2013-2025)': 'VCR_all'})).join(c_.round(2))
        order = ['D0', 'D1', 'O'] + [f'{m} {r}' for m in MODELS for r in RULES]
        print(res.reindex(order).to_string())

    print('\nBase scenario, season by season (k chosen / net gain %), M0 and M1 with D3 and D4_50, plus D1 and oracle:')
    b = D[D.scenario == 'base']
    wide = {}
    for lab, (mod, rl) in {'D1': ('-', 'D1'), 'M0 D3': ('M0', 'D3'), 'M1 D3': ('M1', 'D3'), 'M2 D3': ('M2', 'D3'), 'M0 D4_50': ('M0', 'D4_50'), 'M1 D4_50': ('M1', 'D4_50'), 'O': ('-', 'O')}.items():
        x = b[(b.model == mod) & (b.rule == rl)].set_index('season')
        wide[lab] = x.k.astype(int).astype(str) + ' / ' + (100 * x.gain).round(1).astype(str)
    W = pd.DataFrame(wide); W.insert(0, 'r%', (100 * b[(b.rule == 'D0')].set_index('season').r_annual).round(1))
    print(W.to_string())

    # --------------------------------------------------------------- 5. live 2026 season (not complete)
    live = []
    for scen in SCEN:
        r, lam, c = cost_params(LIVE, scen)
        for m in MODELS:
            for rule in RULES:
                k = choose(FCS[m][LIVE], rule, r, lam, c)
                live.append(dict(scenario=scen, model=m, rule=rule, r_annual=round(r, 4), k_chosen=k,
                                 month_chosen=str(per(LIVE, 5) + k), gain_so_far=realised(LP, LIVE, k, scen)))
    LV = pd.DataFrame(live); LV.to_csv(EXPLORATORY + f'live_2026_{TAG}.csv', index=False, float_format='%.5f')
    print('\n2026 (live, base scenario): realised ratios P[May+k]/P[May] so far:',
          {int(k): round(float(np.exp(target(LP, LIVE, k))), 3) for k in K if not np.isnan(target(LP, LIVE, k))})
    print(LV[LV.scenario == 'base'].pivot(index='model', columns='rule', values='k_chosen').to_string())

    # --------------------------------------------------------------- 6. leakage test: delete everything after May of the test season
    print('\nLeakage test (forecast from truncated data must equal forecast from the full data):')
    worst = 0.0
    for s in [2013, 2019, 2024, 2025]:
        cut = LP.loc[:per(s, 5)]
        dmax = max(np.abs(FC[m](cut, s) - FCS[m][s]).max() for m in MODELS)
        worst = max(worst, dmax)
        print(f'  season {s}: max |difference| over models = {dmax:.2e}')
    print('  PASS' if worst < 1e-9 else '  FAIL: forecasts depend on data after the decision date')
