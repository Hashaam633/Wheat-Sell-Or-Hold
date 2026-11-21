"""
Sensitivity analyses named in the pre-analysis plan (Section 6, robustness list), v3.

1. Excluding repeated WFP values (a monthly price equal to the previous month's):
   - forecast scores: drop rows whose target month or origin month is a repeat;
   - decisions: drop market-seasons with any repeat between May and January.
2. Real-terms equivalence: prices deflated by CPI and discounted at the real rate r_real = (1 + r) / (1 + pi_e) - 1, where pi_e is
   CPI inflation over the 12 months to April (known at the May decision; v1.6, was to May). With expected inflation the real-terms decision problem is
   identical to the nominal one, so sale months do not change; realised values differ only by the inflation surprise:
   V_real = V_nominal * (1 + pi_e)^(k/12) / (CPI[May + k] / CPI[May]).
   CPI: PBS urban CPI (2007-08 = 100) Jul 2015 - Jul 2017, then chained with PBS national CPI (2015-16 = 100) from Aug 2017.
   Seasons 2016-2025 (no monthly CPI before Jul 2015 in our sources; in 2016 pi_e is annualised from 9 months).
3. Without Quetta (v1.6): the market with the most repeated values, which enters only in 2018.
The H1 contrast is also reported under the literal reading of the plan (mean realised V over markets; v1.6).
Output: benchmark/sensitivity_tests.csv, sensitivity_scores.csv, key_numbers_sensitivity.json
"""
import itertools
import json
import re
import subprocess

import numpy as np
import pandas as pd

from decisions_v3 import u, u_inv, value, COSTS

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
PRIMARY = [f'wfp_wheat_{c}' for c in ['Lahore', 'Multan', 'Karachi', 'Peshawar', 'Quetta']]
MAIN = ['snaive', 'm0', 'sarima', 'ets', 'ecm', 'msar', 'lgbm', 'qrf', 'deepar', 'tft', 'm5', 'ens']


def signflip(x):
    x = np.asarray(x); sg = np.array(list(itertools.product([-1.0, 1.0], repeat=len(x))))
    return float(((sg * x).mean(1) >= x.mean() - 1e-12).mean())


def ce(v, g):
    return 100 * (u_inv(u(np.asarray(v), g).mean(), g) - 1)


panel = pd.read_csv(DERIVED + 'colab_panel.csv'); panel['ym'] = pd.PeriodIndex(pd.to_datetime(panel.ds), freq='M')
rep = {}
for uid, g in panel[panel.unique_id.isin(PRIMARY)].groupby('unique_id'):
    s = g.set_index('ym').y.sort_index()
    full = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq='M'))
    r = (full.diff().abs() < 1e-9)
    rep.update({(uid, p): bool(v) for p, v in r.items()})

X = pd.read_csv(B + 'decisions_v3_rows.csv.gz')
X = X[(X.tier == 'primary') & X.season.between(2013, 2025)].copy()


def tests(D, label, seasons_note=''):
    out = []
    for lab, a, b, g, rule in [('H1: ens+R prob - point', ('ens+R', 'prob'), ('ens+R', 'point'), 2, 'B'),
                               ('ens prob - point (no recalibration)', ('ens', 'prob'), ('ens', 'point'), 2, 'B'),
                               ('H6: M5 - SARIMA (prob)', ('m5', 'prob'), ('sarima', 'prob'), 2, 'B'),
                               ('M5+R - SARIMA+R (prob)', ('m5+R', 'prob'), ('sarima+R', 'prob'), 2, 'B')]:
        d = D[(D.gamma == g) & (D.rule == rule) & (D.costs == 'base') & (D.kbar == 8)]
        xa = d[(d.model == a[0]) & (d.form == a[1])].groupby('season').V.apply(lambda v: ce(v, g))
        xb = d[(d.model == b[0]) & (d.form == b[1])].groupby('season').V.apply(lambda v: ce(v, g))
        x = (xa - xb).dropna()
        out.append(dict(variant=label, contrast=lab, seasons=len(x), market_seasons=int(len(d[(d.model == a[0]) & (d.form == a[1])])),
                        mean_diff_pp=x.mean(), p_one_sided=signflip(x), wins=int((x > 1e-9).sum()), losses=int((x < -1e-9).sum()), note=seasons_note))
        if lab.startswith('H1'):                      # v1.6: the literal reading of the plan (mean realised V over markets)
            ma = d[(d.model == a[0]) & (d.form == a[1])].groupby('season').V.mean()
            mb = d[(d.model == b[0]) & (d.form == b[1])].groupby('season').V.mean()
            x = (100 * (ma - mb)).dropna()
            out.append(dict(variant=label, contrast='H1 literal: ens+R prob - point (mean realised V)', seasons=len(x),
                            market_seasons=out[-1]['market_seasons'], mean_diff_pp=x.mean(), p_one_sided=signflip(x),
                            wins=int((x > 1e-9).sum()), losses=int((x < -1e-9).sum()), note=seasons_note))
    for m, form in [('ens+R', 'prob'), ('ens+R', 'point'), ('m5', 'prob'), ('sarima', 'prob'), ('m0+R', 'prob')]:
        d = D[(D.gamma == 2) & (D.rule == 'B') & (D.costs == 'base') & (D.kbar == 8) & (D.model == m) & (D.form == form)]
        out.append(dict(variant=label, contrast=f'CE gain {m} {form}', seasons=d.season.nunique(), market_seasons=len(d),
                        mean_diff_pp=ce(d.V, 2), p_one_sided=np.nan, wins=np.nan, losses=np.nan, note=seasons_note))
    return out


rows = []
rows += tests(X, 'all data (as reported)')

# ---- 1. repeated values
X['any_repeat'] = [any(rep.get((u_, pd.Period(f'{s}-05', 'M') + k), False) for k in range(9)) for u_, s in zip(X.unique_id, X.season)]
rows += tests(X[~X.any_repeat], 'excluding market-seasons with a repeated WFP value May-Jan')
X13 = X[X.season.between(2016, 2025)]
rows += tests(X13, 'all data, seasons 2016-2025 (sample of the real-terms check)')
rows += tests(X[X.unique_id != 'wfp_wheat_Quetta'], 'without Quetta (most repeated values; enters in 2018)')   # v1.6

SR = pd.read_csv(B + 'scored_rows.csv.gz', usecols=['model', 'unique_id', 'o', 'h', 'qs', 'cov80', 'tier', 'season'])
SR = SR[(SR.tier == 'primary') & SR.model.isin(MAIN + [m + '+R' for m in MAIN]) & SR.season.between(2013, 2025)]
op = pd.PeriodIndex(SR.o, freq='M')
SR['rep_target'] = [rep.get((u_, p + int(h)), False) for u_, p, h in zip(SR.unique_id, op, SR.h)]
SR['rep_origin'] = [rep.get((u_, p), False) for u_, p in zip(SR.unique_id, op)]
sc = SR.groupby('model').agg(qs_all=('qs', 'mean'), cov80_all=('cov80', 'mean'), n_all=('qs', 'size'))
keep = SR[~SR.rep_target & ~SR.rep_origin]
sc = sc.join(keep.groupby('model').agg(qs_norepeat=('qs', 'mean'), cov80_norepeat=('cov80', 'mean'), n_norepeat=('qs', 'size')))
sc['rank_all'] = sc.qs_all.rank().astype(int); sc['rank_norepeat'] = sc.qs_norepeat.rank().astype(int)
sc.sort_values('qs_norepeat').to_csv(B + 'sensitivity_scores.csv', float_format='%.10g')
rank_corr = float(sc[['qs_all', 'qs_norepeat']].corr(method='spearman').iloc[0, 1])


# ---- 2. real terms
def pdf_text(path):
    return subprocess.run(['pdftotext', '-layout', path, '-'], capture_output=True, text=True).stdout


def urban_general(text):
    out, fy = {}, None
    for line in text.splitlines():
        if line.strip().startswith('SrNo'):
            m = re.search(r'Jul\s+(\d\d)', line)
            if m:
                fy = 2000 + int(m.group(1))
            continue
        toks = line.split()
        if fy is None or len(toks) < 17:
            continue
        try:
            nums = [float(t) for t in toks[-13:]]
        except ValueError:
            continue
        head = toks[:-13]
        if len(head) >= 4 and all(h.isdigit() for h in head[:3]) and ' '.join(head[3:]).strip().lower() == 'general':
            for i, v in enumerate(nums[:12]):
                yy, mm = (fy, 7 + i) if i < 6 else (fy + 1, i - 5)
                out[pd.Period(f'{yy}-{mm:02d}', 'M')] = v
    return pd.Series(out).sort_index()


_cu = pd.read_csv(DERIVED + 'cpi_urban_general.csv')   # written by build_monthly_panel.py from the PBS PDF
urb = pd.Series(_cu.cpi_urban_general.to_numpy(float), index=pd.PeriodIndex(_cu.month, freq='M')).sort_index()
MP = pd.read_csv(DERIVED + 'monthly_panel.csv'); MP['ym'] = pd.PeriodIndex(MP['month'], freq='M')
nat = MP.set_index('ym').pbs_national_cpi_idx.dropna()
cpi = urb.loc[:'2017-07'].copy()
for p, g in nat.pct_change().loc['2017-08':].items():
    cpi.loc[p] = cpi.iloc[-1] * (1 + g)
cpi = cpi.sort_index()
pol = MP.set_index('ym')['policy_rate_pct'] / 100
price = {(u_, p): y for u_, p, y in zip(panel.unique_id, panel.ym, panel.y)}
spread, lam = COSTS['base']


def infl_terms(s):
    may = pd.Period(f'{s}-05', 'M'); apr = may - 1          # CPI enters with a one-month lag: the May decision knows April's CPI
    back = apr - 12 if (apr - 12) in cpi.index else cpi.index.min()
    months = (apr - back).n
    pi_e = (cpi[apr] / cpi[back]) ** (12 / months) - 1
    I = np.array([cpi[may + k] / cpi[may] for k in range(9)])
    return pi_e, I


XR = X[X.season.between(2016, 2025)].copy()
fac = {}
for s in range(2016, 2026):
    pi_e, I = infl_terms(s)
    fac[s] = np.array([(1 + pi_e) ** (k / 12) / I[k] for k in range(9)])
XR['V'] = [v * fac[s][int(k)] for v, s, k in zip(XR.V, XR.season, XR.k)]
rows += tests(XR, 'real terms (CPI-deflated, real rate), seasons 2016-2025')
infl = pd.DataFrame({s: dict(pi_expected=infl_terms(s)[0], pi_realised_may_jan_annualised=infl_terms(s)[1][8] ** (12 / 8) - 1) for s in range(2016, 2026)}).T

T = pd.DataFrame(rows); T.to_csv(B + 'sensitivity_tests.csv', index=False, float_format='%.10g')
KN = {'repeats': {'market_seasons_total': int(X[(X.model == 'ens+R') & (X.form == 'prob') & (X.gamma == 2) & (X.rule == 'B') & (X.costs == 'base')].shape[0]),
                  'market_seasons_with_repeat': int(X[(X.model == 'ens+R') & (X.form == 'prob') & (X.gamma == 2) & (X.rule == 'B') & (X.costs == 'base')].any_repeat.sum()),
                  'scored_rows_kept_share': float(len(keep) / len(SR)), 'spearman_rank_corr_qs': rank_corr},
      'inflation': infl.round(4).to_dict('index')}
json.dump(KN, open(B + 'key_numbers_sensitivity.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
print(T.round(3).to_string(index=False)); print(json.dumps(KN['repeats'], indent=1)); print(infl.round(3))
print(sc.sort_values('qs_norepeat').round(4).to_string())
