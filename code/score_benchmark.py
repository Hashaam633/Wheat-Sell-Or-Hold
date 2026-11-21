"""
Score every forecast model on the same rows (wheat sell-or-hold paper, v3).

Inputs : benchmark/local_forecasts.csv.gz, benchmark/colab_forecasts.csv.gz, colab_panel.csv, data_extra/panel_flags.csv
Outputs: benchmark/forecast_logratio.parquet-like CSV (all models incl. ensemble and recalibrated versions, log ratios)
         benchmark/scores_by_tier_regime.csv, scores_may_origins.csv, mcs_primary.csv, stepm_primary.csv,
         foundation_clean_window.csv, key_numbers_forecast.json

Rules
 - Forecast target: log(P[origin + h] / P[origin]); quantile score = 2 x mean pinball loss over the 19 levels 0.05..0.95
   (a proper score; equals the CRPS restricted to that quantile range). Coverage of the 80% (0.10-0.90) and 90% (0.05-0.95) intervals.
 - Imputed months (forward-filled) are never used as an origin or as a target.
 - Evaluation: origins from April 2013 (season 2013) onward. Seasons 2010-2012 are a warm-up for recalibration only.
 - Regime of an origin month: R1 support price (seasons <= 2023), T2024 (season 2024), R2 deregulated (Apr-Sep 2025),
   R3 partial reversal (Oct 2025 onward: the Rs3,500 price was announced on 19-20 October 2025).
 - Panel tiers: primary = WFP wheat grain in Lahore, Multan, Karachi, Peshawar, Quetta; validation = PBS wholesale wheat (6 markets,
   to June 2025); robustness = FAO retail flour (5 cities); descriptive = Punjab average (not used in tests).
 - Ensemble 'ens' = equal-weight average of quantiles of the trained models (not the foundation models).
 - '+R' = adaptive recalibration: the spread around the median is multiplied by s, updated every month from the forecasts whose
   outcomes became known that month: log s <- log s + 0.1 * (miss share of the issued 80% intervals - 0.2). Per model and series.
 - Foundation models (Chronos-Bolt, Chronos-2, TimesFM 2.5) may have seen these years in pre-training: they are reported
   separately, on all origins (flagged) and on the window after each model's release (Bolt from Dec 2024, TimesFM 2.5 from
   Oct 2025, Chronos-2 from Nov 2025).
 - Inference: season-level average losses on the primary tier, seasons 2013-2025; Model Confidence Set (Hansen, Lunde and Nason
   2011) and Romano-Wolf StepM against the seasonal climatology M0, both with season bootstrap.
"""
import json

import numpy as np
import pandas as pd

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
QL = np.round(np.arange(0.05, 0.951, 0.05), 2); QC = [f'q{q:.2f}' for q in QL]
TRAINED = ['snaive', 'm0', 'sarima', 'ets', 'ecm', 'msar', 'lgbm', 'qrf', 'deepar', 'tft', 'm5']
FOUNDATION = {'chronos_bolt_base': '2024-12', 'chronos2': '2025-11', 'chronos2_worldcov': '2025-11', 'timesfm25': '2025-10'}
PRIMARY = [f'wfp_wheat_{c}' for c in ['Lahore', 'Multan', 'Karachi', 'Peshawar', 'Quetta']]
GAMMA = 0.1


def tier(u):
    if u in PRIMARY: return 'primary'
    if u.startswith('pbs_whsl'): return 'validation'
    if u.startswith('fao_flour'): return 'robustness'
    return 'descriptive'


def season_of(p):
    return p.year if p.month >= 4 else p.year - 1


def regime_of(p):
    s = season_of(p)
    if s <= 2023: return 'R1'
    if s == 2024: return 'T2024'
    return 'R2' if p < pd.Period('2025-10', 'M') else 'R3'


def load():
    panel = pd.read_csv(DERIVED + 'colab_panel.csv'); panel['ym'] = pd.PeriodIndex(pd.to_datetime(panel.ds), freq='M')
    flags = pd.read_csv(DERIVED + 'panel_flags.csv'); flags['ym'] = pd.PeriodIndex(pd.to_datetime(flags.ds), freq='M')
    price = {(u, p): y for u, p, y in zip(panel.unique_id, panel.ym, panel.y)}
    imputed = {(u, p) for u, p, f in zip(flags.unique_id, flags.ym, flags.imputed) if f}
    F = pd.concat([pd.read_csv(B + 'local_forecasts.csv.gz'), pd.read_csv(B + 'colab_forecasts.csv.gz')], ignore_index=True)
    F['o'] = pd.PeriodIndex(F.origin, freq='M')
    p0 = np.array([price.get((u, o), np.nan) for u, o in zip(F.unique_id, F.o)])
    tgt = [o + h for o, h in zip(F.o, F.h)]
    p1 = np.array([price.get((u, t), np.nan) for u, t in zip(F.unique_id, tgt)])
    bad = np.array([((u, o) in imputed) or ((u, t) in imputed) for u, o, t in zip(F.unique_id, F.o, tgt)])
    Q = np.sort(np.log(F[QC].to_numpy(float) / p0[:, None]), axis=1)
    F = F[['model', 'unique_id', 'o', 'h']].copy()
    F[QC] = Q; F['y'] = np.log(p1 / p0); F['masked'] = bad
    return F


def add_ensemble(F):
    E = F[F.model.isin(TRAINED)]
    n = E.groupby(['unique_id', 'o', 'h']).model.transform('nunique')
    E = E[n == len(TRAINED)]
    G = E.groupby(['unique_id', 'o', 'h'], as_index=False).agg({**{c: 'mean' for c in QC}, 'y': 'first', 'masked': 'first'})
    G['model'] = 'ens'
    return pd.concat([F, G[F.columns]], ignore_index=True)


def recalibrate(F):
    """Adaptive spread scaling per (model, series); uses only outcomes observed by each origin month."""
    out = []
    for (m, u), g in F.groupby(['model', 'unique_id']):
        g = g.sort_values(['o', 'h'])
        Q = g[QC].to_numpy(); med = Q[:, 9:10]
        orgs = g.o.to_numpy(); hs = g.h.to_numpy(); ys = g.y.to_numpy(); mk = g.masked.to_numpy()
        tgt = np.array([o + int(h) for o, h in zip(orgs, hs)])
        by_origin = {}
        for i, o in enumerate(orgs):
            by_origin.setdefault(o, []).append(i)
        by_target = {}
        for i, t in enumerate(tgt):
            by_target.setdefault(t, []).append(i)
        logs, scale = 0.0, np.ones(len(g))
        lo = np.empty(len(g)); hi = np.empty(len(g))
        for o in sorted(by_origin):
            idx = [i for i in by_target.get(o, []) if not mk[i] and not np.isnan(ys[i]) and orgs[i] < o]
            if idx:                                   # outcomes that became known at o, judged against the intervals issued then
                miss = np.mean([(ys[i] < lo[i]) or (ys[i] > hi[i]) for i in idx])
                logs += GAMMA * (miss - 0.2)
            for i in by_origin[o]:
                scale[i] = np.exp(logs)
                lo[i] = med[i, 0] + scale[i] * (Q[i, 1] - med[i, 0]); hi[i] = med[i, 0] + scale[i] * (Q[i, 17] - med[i, 0])
        g2 = g.copy(); g2[QC] = med + scale[:, None] * (Q - med); g2['model'] = m + '+R'
        out.append(g2)
    return pd.concat([F] + out, ignore_index=True)


def score(F):
    Q = F[QC].to_numpy(); y = F.y.to_numpy()[:, None]
    d = y - Q
    F['qs'] = 2 * np.maximum(QL * d, (QL - 1) * d).mean(1)
    F['cov80'] = ((y[:, 0] >= Q[:, 1]) & (y[:, 0] <= Q[:, 17])).astype(float)
    F['cov90'] = ((y[:, 0] >= Q[:, 0]) & (y[:, 0] <= Q[:, 18])).astype(float)
    F['pit'] = [np.interp(v, q, QL, left=0.025, right=0.975) for v, q in zip(F.y.to_numpy(), Q)]
    F['point_ae'] = np.abs(F.y - Q[:, 9])
    return F


if __name__ == '__main__':
    F = load()
    F = add_ensemble(F)
    F = F[F.o >= pd.Period('2010-01', 'M')]
    F = recalibrate(F)
    F = F[~F.masked & F.y.notna()].copy()
    F = F[F.o >= pd.Period('2013-04', 'M')].copy()
    F = score(F)
    F['season'] = [season_of(o) for o in F.o]; F['regime'] = [regime_of(o) for o in F.o]; F['tier'] = [tier(u) for u in F.unique_id]
    F['may'] = [o.month == 5 for o in F.o]
    # comparable rows: every model (trained + foundation, raw) present
    allm = TRAINED + ['ens'] + list(FOUNDATION) + ['rw']
    key = ['unique_id', 'o', 'h']
    cnt = F[F.model.isin(allm)].groupby(key).model.nunique().rename('n').reset_index()
    F = F.merge(cnt, on=key, how='left'); F = F[F.n == len(allm)].drop(columns='n')
    F.drop(columns=QC).to_csv(B + 'scored_rows.csv.gz', index=False, compression='gzip', float_format='%.5f')
    dq = F[[o.month in (5, 6, 7, 8, 9, 10, 11, 12) and h <= 13 - o.month for o, h in zip(F.o, F.h)]]
    dq[['model', 'unique_id', 'o', 'h', 'y'] + QC].assign(o=lambda d: d.o.astype(str)).to_csv(
        B + 'decision_quantiles.csv.gz', index=False, compression='gzip', float_format='%.5f')

    agg = dict(qs=('qs', 'mean'), cov80=('cov80', 'mean'), cov90=('cov90', 'mean'), pit=('pit', 'mean'), mae=('point_ae', 'mean'), n=('qs', 'size'))
    T = F.groupby(['tier', 'regime', 'model']).agg(**agg).reset_index()
    T_all = F.groupby(['tier', 'model']).agg(**agg).reset_index().assign(regime='All')
    T = pd.concat([T, T_all]); T.to_csv(B + 'scores_by_tier_regime.csv', index=False, float_format='%.10g')
    M = F[F.may].groupby(['tier', 'model']).agg(**agg).reset_index(); M.to_csv(B + 'scores_may_origins.csv', index=False, float_format='%.10g')

    # season-level losses, primary tier, complete seasons 2013-2025
    P = F[(F.tier == 'primary') & F.season.between(2013, 2025)]
    L = P.groupby(['season', 'model']).qs.mean().unstack()
    main = TRAINED + ['ens'] + [m + '+R' for m in TRAINED + ['ens']]
    from arch.bootstrap import MCS, StepM
    Lm = L[main]
    mcs = MCS(Lm, size=0.10, reps=5000, block_size=1, method='R', seed=1); mcs.compute()
    pv = mcs.pvalues.rename(columns={'Pvalue': 'mcs_pvalue'})
    pv['in_90pct_mcs'] = pv.index.isin(mcs.included)
    pv['mean_qs'] = Lm.mean().reindex(pv.index)
    pv.sort_values('mean_qs').to_csv(B + 'mcs_primary.csv', float_format='%.10g')
    bench = L['m0']; others = L[[m for m in main if m != 'm0']]
    st = StepM(bench, others, size=0.10, reps=5000, block_size=1, seed=1); st.compute()
    sm = pd.DataFrame({'mean_qs': others.mean(), 'gain_vs_m0_pct': 100 * (1 - others.mean() / bench.mean()),
                       'beats_m0_stepm_10pct': [m in st.superior_models for m in others.columns]})
    sm.sort_values('mean_qs').to_csv(B + 'stepm_primary.csv', float_format='%.10g')

    # foundation models: all origins (flagged) and clean post-release window, same rows as sarima / ens / m5
    rows = []
    for fm, start in FOUNDATION.items():
        for window, cond in [('all origins (possible contamination)', F.o >= pd.Period('2013-04', 'M')),
                             (f'clean: origins from {start}', F.o >= pd.Period(start, 'M'))]:
            sub = F[cond & (F.tier == 'primary')]
            r = sub.groupby('model').qs.mean()
            c = sub.groupby('model').cov80.mean()
            rows.append(dict(model=fm, window=window, n=int((sub.model == fm).sum()), qs=r.get(fm), qs_sarima=r.get('sarima'),
                             qs_ens=r.get('ens'), qs_m5=r.get('m5'), qs_m0=r.get('m0'), cov80=c.get(fm), cov80_ens=c.get('ens')))
    FD = pd.DataFrame(rows); FD.to_csv(B + 'foundation_clean_window.csv', index=False, float_format='%.10g')

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    order = ['rw', 'snaive', 'm0', 'sarima', 'ets', 'ecm', 'msar', 'lgbm', 'qrf', 'deepar', 'tft', 'm5', 'ens'] + list(FOUNDATION)
    for tr in ['primary', 'validation', 'robustness']:
        t = T[(T.tier == tr)]
        a = t.pivot(index='model', columns='regime', values='qs').reindex(index=order)
        c = t[t.regime == 'All'].set_index('model')[['cov80', 'cov90', 'pit', 'mae', 'n']].reindex(order)
        r = t[t.regime == 'All'].set_index('model')
        a['cov80_R'] = [r.cov80.get(m + '+R', np.nan) for m in order]; a['qs_All_R'] = [r.qs.get(m + '+R', np.nan) for m in order]
        print(f'\n=== {tr} tier: quantile score by regime of origin (log points; lower is better), coverage, recalibrated (+R)')
        print(a.join(c).round(3).to_string())
    print('\nMCS (primary tier, season-level QS, 2013-2025):'); print(pv.sort_values('mean_qs').round(3).to_string())
    print('\nStepM vs M0:'); print(sm.sort_values('mean_qs').round(3).to_string())
    print('\nFoundation models:'); print(FD.round(3).to_string(index=False))
    t = T[(T.tier == 'primary') & (T.regime == 'All')].set_index('model')
    json.dump({'forecast': {m: {'qs': round(float(t.qs[m]), 4), 'cov80': round(float(t.cov80[m]), 3)} for m in t.index}},
              open(B + 'key_numbers_forecast.json', 'w'), indent=1)
