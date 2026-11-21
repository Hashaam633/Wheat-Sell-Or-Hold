"""
Diagnostics added in v1.9 after an outside review (descriptive; nothing here changes a forecast or a decision).

1. M5 world-price coefficient: beta (Equation 1) for every series and annual refit, estimated exactly as in
   benchmark_local.m_m5. The adjustment at horizon h is h * beta * (gap - mean gap), so 12 * |beta| < 1 means the
   12-month adjustment never more than closes the gap's deviation from its mean.
2. Adaptive recalibration of M5 on the primary panel: the spread factor s (Equation 2) at each origin, tracked per series and
   pooled over horizons, exactly as in score_benchmark.recalibrate, and the share of outcomes arriving each month that fell
   outside the issued 80 percent intervals.
3. Decisions changed by the support-price floor: M5 with and without the floor (ablation), gamma 2, rule B, primary panel.
4. Repeated WFP prices (a month equal to the previous month) by regime, primary panel.

Outputs: results/m5_beta.csv, results/m5_recal_scale.csv, results/diagnostics.json
"""
import json

import numpy as np
import pandas as pd

from paths import RESULTS, DERIVED
import benchmark_local as BL
import score_benchmark as SB

B = RESULTS
PRIMARY = [f'wfp_wheat_{c}' for c in ['Lahore', 'Multan', 'Karachi', 'Peshawar', 'Quetta']]


def regime(s):
    return 'R1' if s <= 2023 else ('2024' if s == 2024 else ('2025' if s == 2025 else '2026'))


def season_of(p):
    return p.year if p.month >= 4 else p.year - 1


out = {}

# ---------------------------------------------------------------- 1. beta by series and refit
rows = []
for u in BL.SERIES:
    s = BL.LOGP[u]
    gap = (s - BL.LW1.reindex(s.index)).dropna()
    for cut in BL.cutoffs():
        if not BL.origins_for(u, cut) or (s.index <= cut).sum() < BL.MIN_CONTEXT:   # the refits benchmark_local actually runs
            continue
        tr = s.loc[:cut]
        R = []
        for h in range(1, BL.H + 1):
            c = BL.changes(tr, h); c = c[c.index + h <= cut]
            mm = c.groupby(c.index.month).transform('mean')
            R.append(pd.DataFrame({'ex': ((c - mm) / h).values, 'g': gap.reindex(c.index).values}))
        R = pd.concat(R).dropna()
        if len(R) == 0:
            continue
        gbar = gap.loc[:cut].mean()
        rows.append((u, str(cut), int(cut.year), float(((R.g - gbar) * R.ex).sum() / ((R.g - gbar) ** 2).sum()), float(gbar)))
Bt = pd.DataFrame(rows, columns=['unique_id', 'cutoff', 'year', 'beta', 'gap_mean'])
Bt.to_csv(B + 'm5_beta.csv', index=False, float_format='%.10g')
EVAL = Bt.unique_id != 'wfp_wheat_Punjab'                    # the 16 evaluated series (the Punjab average is descriptive)
ev = Bt[EVAL & (Bt.year >= 2013)]                            # refits that issue scored forecasts (origins from April 2013)
evp = ev[ev.unique_id.isin(PRIMARY)]
warm = Bt[EVAL & (Bt.year < 2013) & (12 * Bt.beta.abs() > 1)]
out['m5_beta'] = dict(n_refits_scored=len(ev), min=ev.beta.min(), median=ev.beta.median(), max=ev.beta.max(),
                      max_12abs=float(12 * ev.beta.abs().max()), n_positive=int((ev.beta > 0).sum()),
                      primary_min=evp.beta.min(), primary_median=evp.beta.median(), primary_max=evp.beta.max(),
                      primary_max_12abs=float(12 * evp.beta.abs().max()),
                      warmup_over1=len(warm), warmup_over1_years=sorted(set(int(y) for y in warm.year)),
                      warmup_max_12abs=float(12 * Bt[EVAL & (Bt.year < 2013)].beta.abs().max()))

# ---------------------------------------------------------------- 2. recalibration factor of M5, primary panel
F = SB.load()
F = F[(F.model == 'm5') & F.unique_id.isin(PRIMARY) & (F.o >= pd.Period('2010-01', 'M'))]
path, miss_rows = [], []
for u, g in F.groupby('unique_id'):
    g = g.sort_values(['o', 'h']); Q = g[SB.QC].to_numpy(); med = Q[:, 9:10]
    orgs = g.o.to_numpy(); hs = g.h.to_numpy(); ys = g.y.to_numpy(); mk = g.masked.to_numpy()
    tgt = np.array([o + int(h) for o, h in zip(orgs, hs)])
    by_origin, by_target = {}, {}
    for i, o in enumerate(orgs):
        by_origin.setdefault(o, []).append(i)
    for i, t in enumerate(tgt):
        by_target.setdefault(t, []).append(i)
    logs = 0.0; lo = np.empty(len(g)); hi = np.empty(len(g))
    for o in sorted(by_origin):
        idx = [i for i in by_target.get(o, []) if not mk[i] and not np.isnan(ys[i]) and orgs[i] < o]
        if idx:
            miss = np.mean([(ys[i] < lo[i]) or (ys[i] > hi[i]) for i in idx]); logs += SB.GAMMA * (miss - 0.2)
            miss_rows.append((u, str(o), miss, len(idx)))
        sc = np.exp(logs); path.append((u, str(o), sc))
        for i in by_origin[o]:
            lo[i] = med[i, 0] + sc * (Q[i, 1] - med[i, 0]); hi[i] = med[i, 0] + sc * (Q[i, 17] - med[i, 0])
P = pd.DataFrame(path, columns=['unique_id', 'origin', 'scale'])
M = pd.DataFrame(miss_rows, columns=['unique_id', 'month', 'miss_share', 'n_outcomes'])
P.merge(M.rename(columns={'month': 'origin'}), on=['unique_id', 'origin'], how='left').to_csv(B + 'm5_recal_scale.csv', index=False, float_format='%.10g')
pm = P.groupby('origin').scale.mean()
mm = M.groupby('month').miss_share.mean()
pk = pm.loc['2024-04':'2025-12']
out['m5_recal'] = dict(scale_mean={k: float(pm.get(k)) for k in ['2023-05', '2024-03', '2024-05', '2024-09', '2025-01', '2025-05', '2025-09']},
                       peak_month=str(pk.idxmax()), peak=float(pk.max()),
                       miss_mean={k: float(mm.get(k)) for k in ['2024-04', '2024-05', '2024-09', '2024-12', '2025-01', '2025-03', '2025-04', '2025-05']},
                       first_month_below_target=str(mm.loc['2024-05':][mm.loc['2024-05':] < 0.2].index[0]),
                       max_monthly_fall_pct=float(100 * (1 - np.exp(-SB.GAMMA * 0.2))), max_monthly_rise_pct=float(100 * (np.exp(SB.GAMMA * 0.8) - 1)))

# ---------------------------------------------------------------- 3. decisions changed by the floor
A = pd.read_csv(B + 'ablation_decisions.csv')
a = A[(A.gamma == 2) & (A.form == 'prob') & (A.tier == 'primary') & A.season.between(2013, 2025)]
w = a.pivot_table(index=['unique_id', 'season'], columns='model', values='k')
d = w[w['m5_full'] != w['m5_nofloor']]
out['floor_decisions'] = dict(market_seasons=len(w), changed=len(d), changed_by_regime={r_: int(sum(regime(s) == r_ for s in d.index.get_level_values(1))) for r_ in ['R1', '2024', '2025']},
                              seasons=sorted(set(int(s) for s in d.index.get_level_values(1))), floor_later=int((d['m5_full'] > d['m5_nofloor']).sum()),
                              mean_months_full=float(d['m5_full'].mean()), mean_months_nofloor=float(d['m5_nofloor'].mean()))

# ---------------------------------------------------------------- 4. repeated WFP prices by regime
panel = pd.read_csv(DERIVED + 'colab_panel.csv'); panel['ym'] = pd.PeriodIndex(pd.to_datetime(panel.ds), freq='M')
rep = []
for u in PRIMARY:
    s = panel[panel.unique_id == u].set_index('ym').y.sort_index()
    rep += [(u, p) for p, v in (s.diff().abs() < 1e-9).items() if v]
RP = pd.DataFrame(rep, columns=['unique_id', 'ym']); RP['season'] = [season_of(p) for p in RP.ym]
RP['period'] = ['before 2013' if s < 2013 else regime(s) for s in RP.season]
repset = set(zip(RP.unique_id, RP.ym))
X = pd.read_csv(B + 'decisions_v3_rows.csv.gz', usecols=['model', 'form', 'gamma', 'rule', 'costs', 'kbar', 'unique_id', 'season', 'tier'])
ms = X[(X.model == 'ens+R') & (X.form == 'prob') & (X.gamma == 2) & (X.rule == 'B') & (X.costs == 'base') & (X.kbar == 8)
       & (X.tier == 'primary') & X.season.between(2013, 2025)][['unique_id', 'season']]
ms['any'] = [any((u, pd.Period(f'{s}-05', 'M') + k) in repset for k in range(9)) for u, s in zip(ms.unique_id, ms.season)]
ms['regime'] = ms.season.map(regime)
out['repeats'] = dict(total=len(RP), by_period=RP.groupby('period').size().astype(int).to_dict(),
                      market_seasons_with_repeat={r_: [int(g['any'].sum()), int(len(g))] for r_, g in ms.groupby('regime')})

txt = json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
open(B + 'diagnostics.json', 'w').write(txt)
print(txt)
