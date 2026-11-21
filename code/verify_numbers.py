"""Independent check of key paper numbers: recomputes them from the row-level result files with plain Python and the
csv module (no project code) and compares them with results/paper_numbers.json."""
import csv
import gzip
import json
import math
from collections import defaultdict

import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
B = R + 'results/'
N = json.load(open(B + 'paper_numbers.json'))
ok_all = []


def chk(name, mine, paper, tol=1e-3):
    ok = mine is not None and paper is not None and abs(float(mine) - float(paper)) < tol
    ok_all.append(ok)
    print(f'{name:46s} independent {mine!s:>12}  paper file {paper!s:>12}  {"OK" if ok else "MISMATCH"}')


def ce(v):                       # gamma = 2: CE = harmonic mean of V
    return 100 * (len(v) / sum(1 / max(x, 1e-6) for x in v) - 1)


# ---- decisions (rule B, gamma 2, base costs, kbar 8, primary panel, seasons 2013-2025)
acc = defaultdict(list); kk = defaultdict(list)
with gzip.open(B + 'decisions_v3_rows.csv.gz', 'rt') as f:
    for r in csv.DictReader(f):
        if r['tier'] != 'primary' or r['gamma'] != '2' or r['rule'] != 'B' or r['costs'] != 'base' or r['kbar'] != '8':
            continue
        s = int(r['season'])
        if 2013 <= s <= 2025:
            acc[(r['model'], r['form'], s)].append(float(r['V'])); kk[(r['model'], r['form'], s)].append(float(r['k']))
S = range(2013, 2026)
sce = lambda m, f, s: ce(acc[(m, f, s)])
gap = lambda a, b: sum(sce(*a, s) - sce(*b, s) for s in S) / 13
pooled = lambda m, f, excl=(): ce([x for s in S if s not in excl for x in acc[(m, f, s)]])
dec = {d['model']: d for d in N['dec']}
pt = {r['test']: r for r in N['primary_test_rows']}
chk('H1 ens+R prob - point', gap(('ens+R', 'prob'), ('ens+R', 'point')), pt['H1 primary: ens+R prob vs point, gamma 2, rule B']['mean_diff_pp'])
chk('H6 M5 - SARIMA (prob)', gap(('m5', 'prob'), ('sarima', 'prob')), pt['H6: M5 vs SARIMA (prob), gamma 2, rule B']['mean_diff_pp'])
chk('H6b M5+R - SARIMA+R (prob)', gap(('m5+R', 'prob'), ('sarima+R', 'prob')), pt['H6b: M5+R vs SARIMA+R (prob), gamma 2, rule B']['mean_diff_pp'])
chk('pooled CE ens+R prob', pooled('ens+R', 'prob'), dec['ens']['ce_probR'])
chk('pooled CE ens point', pooled('ens', 'point'), dec['ens']['ce_point'])
chk('pooled CE M5 prob (Table 9)', pooled('m5', 'prob'), dec['m5']['ce_prob'])
chk('pooled CE M5+R prob (Table 9)', pooled('m5+R', 'prob'), dec['m5']['ce_probR'])
chk('market-seasons', sum(len(acc[('ens+R', 'prob', s)]) for s in S), N['dec_n_market_seasons'])
E = N['m5_recal_effect']
chk('v1.2 7.5: M5+R - M5 (prob), 2025', sce('m5+R', 'prob', 2025) - sce('m5', 'prob', 2025), E['by_season']['2025'])
chk('v1.2 7.4: M5+R prob - point, 2025', sce('m5+R', 'prob', 2025) - sce('m5+R', 'point', 2025), E['by_season']['2025'], tol=0.05)
chk('v1.2 7.5: mean recal effect, other 12', sum(sce('m5+R', 'prob', s) - sce('m5', 'prob', s) for s in S if s != 2025) / 12, E['mean_without_2025'])
chk('v1.2 7.5: pooled M5 prob without 2025', pooled('m5', 'prob', (2025,)), E['pooled_m5_wo2025'])
chk('v1.2 7.5: pooled M5+R prob without 2025', pooled('m5+R', 'prob', (2025,)), E['pooled_m5R_wo2025'])
chk('v1.2 7.5: mean sale month 2025, M5', sum(kk[('m5', 'prob', 2025)]) / len(kk[('m5', 'prob', 2025)]), E['k_2025_m5'])
chk('v1.2 7.5: mean sale month 2025, M5+R', sum(kk[('m5+R', 'prob', 2025)]) / len(kk[('m5+R', 'prob', 2025)]), E['k_2025_m5R'])

# ---- rule D (copula paths)
accD = defaultdict(list); kD = defaultdict(list)
with gzip.open(B + 'rule_d_rows.csv.gz', 'rt') as f:
    for r in csv.DictReader(f):
        if r['tier'] == 'primary' and r['gamma'] == '2' and r['model'] in ('m5', 'm5+R') and 2013 <= int(r['season']) <= 2025:
            accD[r['model']].append(float(r['V_D'])); kD[(r['model'], int(r['season']))].append(float(r['k_D']))
rd = {(x['model']): x for x in N['rule_d'] if x['tier'] == 'primary' and x['gamma'] == 2}
chk('v1.2 7.5: rule D pooled CE, M5', ce(accD['m5']), rd['m5']['ce_D'], tol=0.006)
chk('v1.2 7.5: rule D pooled CE, M5+R', ce(accD['m5+R']), rd['m5+R']['ce_D'], tol=0.006)
for m in ('m5', 'm5+R'):
    for s in (2022, 2025):
        chk(f'v1.2 7.5: rule D sale month {s}, {m}', sum(kD[(m, s)]) / len(kD[(m, s)]), N['m5_recal_effect_D']['k_D'][m][str(s)])

# ---- calibration diagnostics (Table C2) from the scored rows
cal = defaultdict(lambda: [0, 0, 0, 0, 0.0, 0.0, 0, 0])   # n, in80, in90, -, sum pit, sum pit^2, lo, hi
MAIN = ['snaive', 'm0', 'sarima', 'ets', 'ecm', 'msar', 'lgbm', 'qrf', 'deepar', 'tft', 'm5', 'ens']
with gzip.open(B + 'scored_rows.csv.gz', 'rt') as f:
    for r in csv.DictReader(f):
        m = r['model']
        if r['tier'] != 'primary' or m.replace('+R', '') not in MAIN or r['pit'] in ('', 'nan'):
            continue
        p = float(r['pit']); c = cal[m]
        c[0] += 1; c[1] += float(r['cov80']); c[2] += float(r['cov90']); c[4] += p; c[5] += p * p; c[6] += p < 0.05; c[7] += p > 0.95
calN = {x['model']: x for x in N['calibration']}
for m in ['deepar', 'm5', 'ens+R', 'tft+R']:
    n, a80, a90, _, sp, sp2, lo, hi = cal[m]
    mean = sp / n; sd = math.sqrt((sp2 - n * mean * mean) / (n - 1))
    chk(f'C2 {m}: cov90', a90 / n, calN[m]['cov90']); chk(f'C2 {m}: PIT sd', sd, calN[m]['pit_sd'])
    chk(f'C2 {m}: below 5%', lo / n, calN[m]['pit_lo']); chk(f'C2 {m}: above 95%', hi / n, calN[m]['pit_hi'])
up = lambda sfx: sum(cal[m + sfx][7] > cal[m + sfx][6] for m in MAIN)
print('7.1 counts (upper > lower):', up(''), 'as issued,', up('+R'), 'after recalibration;',
      'PIT mean > 0.5:', sum(cal[m][4] / cal[m][0] > 0.5 for m in MAIN))
miss = lambda sfx: [1 - cal[m + sfx][2] / cal[m + sfx][0] for m in MAIN]
print('7.1 90%% misses as issued %.0f-%.0f%%, recalibrated %.0f-%.0f%%' % (100 * min(miss('')), 100 * max(miss('')), 100 * min(miss('+R')), 100 * max(miss('+R'))))

# ---- support price as a soft lower bound (Section 5.3)
sp = {}
with open(R + 'data/compiled/support_prices.csv') as f:
    for r in csv.DictReader(f):
        if r['support_price_rs40'] and r['punjab_procurement'] == 'yes':
            sp[int(r['harvest_year'])] = float(r['support_price_rs40'])
late, harv = [], []
with open(R + 'data/derived/monthly_panel.csv') as f:
    for r in csv.DictReader(f):
        if not r['wfp_punjab_rs40']:
            continue
        y, mth = int(r['month'][:4]), int(r['month'][5:7]); s = y if mth >= 4 else y - 1
        if 2006 <= s <= 2023 and s in sp:
            (harv if mth in (4, 5, 6) else late).append(float(r['wfp_punjab_rs40']) / sp[s])
FB = N['floor_beyond_harvest']
chk('5.3 share below support, Jul-Mar', sum(x < 1 for x in late) / len(late), FB['share_below_late'])
chk('5.3 share below support, Apr-Jun', sum(x < 1 for x in harv) / len(harv), FB['share_below_harvest'])
print('\nALL MATCH' if all(ok_all) else '\nSOME MISMATCH', f'({sum(ok_all)}/{len(ok_all)})')
