"""
Wheat sell-or-hold study, step 3 (v2): regime-aware model M5, adaptive recalibration, and the farmer-type grid.

Builds on wheat_forecast_backtest.py (v1), which must sit in the same folder; its models, rules and cost
scenarios are reused unchanged.

New inputs : data_extra/support_prices.csv   support price F per harvest year and whether Punjab procured
             monthly_panel.csv               world_hrw_rs40 (US HRW wheat in Rs per 40 kg, from the spliced exchange rate)
Outputs    : scores_forecast_v2.csv, decisions_v2.csv, summary_decisions_v2.csv, paired_tests_v2.csv,
             recalibration_scale_v2.csv, farmer_grid_v2.csv, live_2026_v2.csv

PRE-SPECIFIED DESIGN (fixed before running; nothing tuned on test seasons)
  M5 (regime-aware) = M1 (seasonal ARIMA) predictive distribution, plus two policy/market mechanisms:
    (a) import-parity pull: expected log change shifts by k * beta * (g_s - mean g), where g_s = log(P_Apr / world_Apr)
        is the gap between Punjab and US HRW wheat in rupees in April (known before the May decision).
        beta is the OLS slope, over earlier seasons only, of the average monthly excess change on the gap.
    (b) floor: with probability q_s the market price after harvest cannot fall below f * F_s, where F_s is the support
        price and f the 5th percentile of P[May+k]/F over earlier seasons with procurement. q_s = 1 if Punjab procured,
        0 if it did not (2024, 2025), 0.16 in 2026 (share of the procurement target reached by July 2026).
  Recalibration (all models, suffix '+R'): forecast spread around the median is multiplied by s_t, with
        log s_{t+1} = log s_t + gamma * (e_t - 0.2), where e_t is the share of the 8 horizons of season t that fell outside
        the 80% interval. gamma = 1 (sensitivity 0.5 and 2). Seasons 2010-2012 are a warm-up (scored, not reported).
        All 8 outcomes of a season are known by January, before the next May decision, so the update uses past data only.
  Farmer grid: longest affordable holding month kbar in {2, 4, 6, 8} x cost scenario; rules choose only k <= kbar.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
spec = importlib.util.spec_from_file_location('v1', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wheat_forecast_backtest.py'))
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
per, target, gain, LP, K, KMAX, J, TAUS, SCEN = v1.per, v1.target, v1.gain, v1.LP, v1.K, v1.KMAX, v1.J, v1.TAUS, v1.SCEN

WARM = [2010, 2011, 2012]
TEST = list(range(v1.FIRST_TEST, v1.LAST_COMPLETE + 1))
LIVE = v1.LIVE
GAMMA, ALPHA = 1.0, 0.2

FL = pd.read_csv(COMPILED + 'support_prices.csv').set_index('harvest_year')
Q_REACH = {'yes': 1.0, 'no': 0.0, 'limited': 0.16}
LW = np.log(v1.P['world_hrw_rs40'].dropna())


def floor_info(s):
    F = FL.loc[s, 'support_price_rs40']
    return (np.nan if pd.isna(F) else float(F)), Q_REACH[FL.loc[s, 'punjab_procurement']]


def gap(lp, s):
    d, w = lp.to_dict(), LW.to_dict(); p = per(s, 4)
    return d[p] - w[p] if (p in d and p in w) else np.nan


def mix_floor(Q, L, q):
    """Mixture of the original quantiles (weight 1-q) and the same quantiles censored below at L (weight q)."""
    if q <= 0 or np.isnan(L):
        return Q
    vals = np.concatenate([np.maximum(Q, L), Q]); w = np.concatenate([np.full(J, q), np.full(J, 1 - q)]) / J
    o = np.argsort(vals, kind='stable'); vals, w = vals[o], w[o]
    cum = np.cumsum(w) - w / 2
    return np.interp(TAUS, cum, vals)


M5_INFO = {}


def fc_M5(lp, s, use_gap=True, use_floor=True):
    Q = v1.fc_M1(lp, s).copy()
    d = lp.to_dict()
    tr = [sp for sp in range(v1.FIRST_TRAIN, s)]
    # (a) import-parity pull
    Y = np.array([[target(lp, sp, k) for k in K] for sp in tr])
    G = np.array([gap(lp, sp) for sp in tr])
    ok = ~np.isnan(G) & ~np.isnan(Y).any(1)
    dev = ((Y[ok] - Y[ok].mean(0)) / K).mean(1)                     # average monthly excess change per season
    g = G[ok] - G[ok].mean()
    beta = float((g * dev).sum() / (g * g).sum())
    gs = gap(lp, s) - G[ok].mean()
    if use_gap:
        Q = Q + (K * beta * gs)[:, None]
    # (b) floor with reach q
    F, q = floor_info(s)
    rat = [np.exp(d[per(sp, 5) + k]) / floor_info(sp)[0] for sp in tr if floor_info(sp)[1] == 1.0 for k in range(0, KMAX + 1)
           if (per(sp, 5) + k) in d]
    f = float(np.percentile(rat, 5))
    if use_floor and q > 0 and not np.isnan(F):
        L = np.log(f * F) - d[per(s, 5)]
        Q = np.vstack([mix_floor(Q[i], L, q) for i in range(KMAX)])
    M5_INFO[s] = dict(beta=beta, gap_centered=gs, floor_level_f=f, F=F, q=q)
    return Q


FC = {**v1.FC, 'M5': fc_M5}
MODELS = ['M0', 'M1', 'M2', 'M3', 'M5']


def recalibrate(fcs, gamma):
    """fcs: {season: (KMAX, J)} for WARM + TEST + LIVE, in time order. Returns recalibrated forecasts and scales used."""
    logs, out, scale = 0.0, {}, {}
    for s in sorted(fcs):
        Q = fcs[s]; med = 0.5 * (Q[:, J // 2 - 1] + Q[:, J // 2])
        Qs = med[:, None] + np.exp(logs) * (Q - med[:, None])
        out[s], scale[s] = Qs, np.exp(logs)
        y = np.array([target(LP, s, k) for k in K])
        if not np.isnan(y).any():                                        # season complete: update for the next one
            lo = np.array([np.interp(0.10, TAUS, Qs[i]) for i in range(KMAX)])
            hi = np.array([np.interp(0.90, TAUS, Qs[i]) for i in range(KMAX)])
            e = float(((y < lo) | (y > hi)).mean())
            logs += gamma * (e - ALPHA)
    return out, scale


def choose_capped(Q, rule, r, lam, c, kbar):
    Qc = Q.copy(); Qc[kbar:] = -10.0                                  # months after kbar are infeasible
    return v1.choose(Qc, rule, r, lam, c)


def oracle_capped(s, scen, kbar):
    g = [v1.realised(LP, s, k, scen) for k in range(1, kbar + 1)]
    j = int(np.nanargmax(g)); return (j + 1, g[j]) if g[j] > 0 else (0, 0.0)


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    seasons = WARM + TEST + [LIVE]
    RAW = {m: {s: FC[m](LP, s) for s in seasons} for m in MODELS}
    print('M5 parameters by season (beta = monthly pull per unit of gap; f = floor level as share of support price):')
    print(pd.DataFrame(M5_INFO).T.loc[TEST + [LIVE]].round(3).to_string())

    ALL = {m: RAW[m] for m in MODELS}
    SCALES = {}
    for m in MODELS:
        for gname, gam in [('+R', GAMMA), ('+R05', 0.5), ('+R2', 2.0)]:
            ALL[m + gname], SCALES[m + gname] = recalibrate(RAW[m], gam)
    pd.DataFrame({k: v for k, v in SCALES.items() if k.endswith('+R')}).round(3).to_csv(EXPLORATORY + 'recalibration_scale_v2.csv')
    print('\nSpread multiplier s_t used in each season (gamma = 1):')
    print(pd.DataFrame({k: v for k, v in SCALES.items() if k.endswith('+R')}).round(2).loc[TEST + [LIVE]].T.to_string())

    # ---------------------------------------------------------------- forecast scores
    sc = []
    for m, fcs in ALL.items():
        for s in TEST:
            for i, k in enumerate(K):
                crps, c80, c90, pit = v1.crps_cov(fcs[s][i], target(LP, s, k))
                sc.append((m, s, int(k), crps, c80, c90, pit))
    SC = pd.DataFrame(sc, columns=['model', 'season', 'k', 'crps', 'cov80', 'cov90', 'pit'])
    rows = []
    for pname, ss in v1.PERIODS.items():
        for m in ALL:
            x = SC[(SC.model == m) & SC.season.isin(ss)]
            rows.append((pname, m, x.crps.mean(), x.cov80.mean(), x.cov90.mean(), x.pit.mean()))
    FS = pd.DataFrame(rows, columns=['period', 'model', 'crps_logpts', 'coverage80', 'coverage90', 'mean_pit'])
    FS.to_csv(EXPLORATORY + 'scores_forecast_v2.csv', index=False, float_format='%.4f')
    show = [m for m in ALL if not (m.endswith('+R05') or m.endswith('+R2'))]
    for col in ['crps_logpts', 'coverage80']:
        print(f'\n{col} by model and period:')
        print(FS[FS.model.isin(show)].pivot(index='model', columns='period', values=col).reindex(index=show, columns=list(v1.PERIODS)).round(3).to_string())
    print('\nCoverage80, all test seasons, recalibration step-size sensitivity:')
    t = FS[FS.period == 'All (2013-2025)'].set_index('model').coverage80
    print(pd.DataFrame({g: [t[m + g] for m in MODELS] for g in ['+R05', '+R', '+R2']}, index=MODELS).round(2).to_string())
    # first season after each break
    print('\n80% coverage in the first season after each break (2024, 2025):')
    print(SC[SC.season.isin([2024, 2025]) & SC.model.isin(show)].groupby(['model', 'season']).cov80.mean().unstack().reindex(show).round(2).to_string())

    # ---------------------------------------------------------------- decisions
    dec = []
    for scen in SCEN:
        for s in TEST:
            r, lam, c = v1.cost_params(s, scen)
            base = dict(scenario=scen, season=s, r_annual=round(r, 4))
            ko, go = v1.oracle(LP, s, scen)
            dec += [{**base, 'model': '-', 'rule': 'D0', 'k': 0, 'gain': 0.0},
                    {**base, 'model': '-', 'rule': 'D1', 'k': 7, 'gain': v1.realised(LP, s, 7, scen)},
                    {**base, 'model': '-', 'rule': 'O', 'k': ko, 'gain': go}]
            for m in [x for x in ALL if not (x.endswith('+R05') or x.endswith('+R2'))]:
                for rule in v1.RULES:
                    k = v1.choose(ALL[m][s], rule, r, lam, c)
                    dec.append({**base, 'model': m, 'rule': rule, 'k': k, 'gain': v1.realised(LP, s, k, scen)})
    D = pd.DataFrame(dec); D.to_csv(EXPLORATORY + 'decisions_v2.csv', index=False, float_format='%.5f')

    summ = []
    for (scen, model, rule), g in D.groupby(['scenario', 'model', 'rule']):
        for pname, ss in v1.PERIODS.items():
            x = g[g.season.isin(ss)]; gv = x.gain.values
            o = D[(D.scenario == scen) & (D.rule == 'O') & D.season.isin(ss)].gain.mean()
            lo, hi = v1.boot_ci(gv)
            summ.append(dict(scenario=scen, model=model, rule=rule, period=pname, seasons=len(x), mean_gain_pct=100 * gv.mean(),
                             share_loss=(gv < -1e-12).mean(), worst_pct=100 * gv.min(), cvar20_pct=100 * v1.cvar20(gv),
                             vcr=gv.mean() / o if o > 1e-9 else np.nan, ci90_lo_pct=100 * lo, ci90_hi_pct=100 * hi))
    S = pd.DataFrame(summ); S.to_csv(EXPLORATORY + 'summary_decisions_v2.csv', index=False, float_format='%.4f')

    import itertools
    def signflip_p(d):
        d = np.asarray(d, float)
        if np.allclose(d, 0): return np.nan
        sg = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d))))
        return float((np.abs((sg * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())
    pr = []
    for scen in SCEN:
        d1 = D[(D.scenario == scen) & (D.rule == 'D1')].set_index('season').gain
        for (model, rule), g in D[(D.scenario == scen) & (D.model != '-')].groupby(['model', 'rule']):
            g = g.set_index('season').gain
            for pname in ['R1 (2013-2023)', 'All (2013-2025)']:
                ss = v1.PERIODS[pname]; diff = (g.loc[ss] - d1.loc[ss]).values
                pr.append(dict(scenario=scen, model=model, rule=rule, period=pname, mean_gain_pct=100 * g.loc[ss].mean(),
                               mean_diff_vs_D1_pct=100 * diff.mean(), p_vs_D1=signflip_p(diff), p_vs_D0=signflip_p(g.loc[ss].values)))
    PR = pd.DataFrame(pr); PR.to_csv(EXPLORATORY + 'paired_tests_v2.csv', index=False, float_format='%.4f')

    for scen in ['base']:
        t = S[(S.scenario == scen)].copy()
        t['label'] = np.where(t.model == '-', t.rule, t.model + ' ' + t.rule)
        a = t.pivot(index='label', columns='period', values='mean_gain_pct').reindex(columns=list(v1.PERIODS))
        allp = t[t.period == 'All (2013-2025)'].set_index('label')[['vcr', 'share_loss', 'cvar20_pct', 'worst_pct']]
        p = PR[(PR.scenario == scen) & (PR.period == 'All (2013-2025)')].assign(label=lambda d: d.model + ' ' + d.rule).set_index('label')[['p_vs_D1']]
        res = a.round(1).join(allp.round(2)).join(p.round(3))
        keep = ['D0', 'D1', 'O'] + [f'{m} {r}' for m in ['M0', 'M0+R', 'M1', 'M1+R', 'M5', 'M5+R', 'M2+R', 'M3+R'] for r in ['D2', 'D3', 'D4_25', 'D4_50', 'D5']]
        print(f'\n=== Decision value, {scen} scenario (mean net gain % of May price; VCR, losses, CVaR20, worst and p vs D1 over 2013-2025)')
        print(res.reindex(keep).to_string())
        print('\nRules with p_vs_D1 < 0.10 and a positive difference (any scenario/period):')
        print(PR[(PR.p_vs_D1 < 0.10) & (PR.mean_diff_vs_D1_pct > 0)].round(3).to_string(index=False))

    b = D[D.scenario == 'base']
    wide = {}
    for lab, (mod, rl) in {'D1': ('-', 'D1'), 'M1 D3': ('M1', 'D3'), 'M5 D3': ('M5', 'D3'), 'M5+R D3': ('M5+R', 'D3'),
                           'M5+R D4_50': ('M5+R', 'D4_50'), 'M0+R D4_50': ('M0+R', 'D4_50'), 'O': ('-', 'O')}.items():
        x = b[(b.model == mod) & (b.rule == rl)].set_index('season')
        wide[lab] = x.k.astype(int).astype(str) + ' / ' + (100 * x.gain).round(1).astype(str)
    print('\nBase scenario by season (k / net gain %):'); print(pd.DataFrame(wide).to_string())

    # ---------------------------------------------------------------- farmer-type grid (H4)
    grid = []
    for scen in SCEN:
        for kbar in [2, 4, 6, 8]:
            for m, rule in [('-', 'D1'), ('M0', 'D3'), ('M1', 'D3'), ('M5', 'D3'), ('M5+R', 'D3'), ('M5+R', 'D4_25'), ('M5+R', 'D4_50'), ('M0+R', 'D4_50')]:
                gs, go = [], []
                for s in TEST:
                    r, lam, c = v1.cost_params(s, scen)
                    if m == '-':
                        k = min(7, kbar)
                    else:
                        k = choose_capped(ALL[m][s], rule, r, lam, c, kbar)
                    gs.append(v1.realised(LP, s, k, scen)); go.append(oracle_capped(s, scen, kbar)[1])
                gs, go = np.array(gs), np.array(go)
                grid.append(dict(scenario=scen, kbar=kbar, model=m, rule=rule, mean_gain_pct=100 * gs.mean(),
                                 oracle_pct=100 * go.mean(), vcr=gs.mean() / go.mean() if go.mean() > 0 else np.nan,
                                 cvar20_pct=100 * v1.cvar20(gs)))
    GR = pd.DataFrame(grid); GR.to_csv(EXPLORATORY + 'farmer_grid_v2.csv', index=False, float_format='%.4f')
    print('\nFarmer grid: VCR by cost scenario (rows) and longest affordable holding month kbar (columns):')
    for lab in [('M5+R', 'D3'), ('M0', 'D3'), ('-', 'D1')]:
        t = GR[(GR.model == lab[0]) & (GR.rule == lab[1])]
        print(' ', lab, '\n', t.pivot(index='scenario', columns='kbar', values='vcr').reindex(['low', 'base', 'high']).round(2).to_string())
    t = GR[(GR.model == '-')]
    print('  oracle mean gain %:\n', t.pivot(index='scenario', columns='kbar', values='oracle_pct').reindex(['low', 'base', 'high']).round(1).to_string())

    # ---------------------------------------------------------------- ablation of M5 (which mechanism matters?)
    abl = []
    variants = {'M5 full': dict(), 'M5 no floor (gap only)': dict(use_floor=False), 'M5 no gap (floor only)': dict(use_gap=False)}
    for name, kw in variants.items():
        fcs = {s: fc_M5(LP, s, **kw) for s in TEST}
        crps = np.mean([v1.crps_cov(fcs[s][i], target(LP, s, k))[0] for s in TEST for i, k in enumerate(K)])
        cov = np.mean([v1.crps_cov(fcs[s][i], target(LP, s, k))[1] for s in TEST for i, k in enumerate(K)])
        for rule in ['D3', 'D4_25', 'D4_50']:
            gs, ks = [], []
            for s in TEST:
                r, lam, c = v1.cost_params(s, 'base'); k = v1.choose(fcs[s], rule, r, lam, c)
                gs.append(v1.realised(LP, s, k, 'base')); ks.append(k)
            abl.append(dict(variant=name, rule=rule, crps=crps, cov80=cov, mean_gain_pct=100 * np.mean(gs),
                            k_2017=ks[TEST.index(2017)], k_2024=ks[TEST.index(2024)], k_2025=ks[TEST.index(2025)]))
    AB = pd.DataFrame(abl); AB.to_csv(EXPLORATORY + 'ablation_m5_v2.csv', index=False, float_format='%.4f')
    print('\nAblation of M5, base scenario, all test seasons:'); print(AB.round(3).to_string(index=False))

    # ---------------------------------------------------------------- live 2026
    live = []
    for scen in SCEN:
        r, lam, c = v1.cost_params(LIVE, scen)
        for m in ['M0', 'M1', 'M5', 'M0+R', 'M1+R', 'M5+R']:
            for rule in v1.RULES:
                k = v1.choose(ALL[m][LIVE], rule, r, lam, c)
                live.append(dict(scenario=scen, model=m, rule=rule, k_chosen=k, month=str(per(LIVE, 5) + k),
                                 gain_so_far=v1.realised(LP, LIVE, k, scen)))
    LV = pd.DataFrame(live); LV.to_csv(EXPLORATORY + 'live_2026_v2.csv', index=False, float_format='%.5f')
    print('\n2026 live, base scenario, chosen month:')
    print(LV[LV.scenario == 'base'].pivot(index='model', columns='rule', values='month').to_string())
    Qm = ALL['M5+R'][LIVE]
    print('M5+R 2026 forecast of P[May+k]/P[May], 10/50/90%:', {int(k): tuple(np.round(np.exp(np.interp([.1, .5, .9], TAUS, Qm[i])), 3)) for i, k in enumerate(K)})

    # ---------------------------------------------------------------- leakage test for M5
    print('\nLeakage test M5:')
    worst = 0.0
    for s in [2013, 2019, 2024, 2025]:
        dmax = np.abs(fc_M5(LP.loc[:per(s, 5)], s) - RAW['M5'][s]).max(); worst = max(worst, dmax)
        print(f'  season {s}: {dmax:.2e}')
    print('  PASS' if worst < 1e-9 else '  FAIL')
