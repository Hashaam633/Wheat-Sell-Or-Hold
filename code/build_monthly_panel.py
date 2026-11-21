"""
Build the monthly analysis panel for the wheat sell-or-hold study.

Inputs (all in data/raw; see data/README.md for where to download them):
  wfp_food_prices_pak.csv                       WFP retail wheat prices (Lahore, Multan, Karachi, Peshawar)
  Policy_Rate.csv                               State Bank of Pakistan policy rate (steps)
  Exchange_Rates_-_PKR_US_.csv                  PKR per USD, daily
  CMO-Historical-Data-Monthly.xlsx              World Bank Pink Sheet, monthly commodity prices (USD)
  RPI-Gropwise-Cumulative-Indices.pdf           PBS rural price index by group (2015-16 = 100), incl. Wheat
  CpI-Urban-Groupwise-Cumulative-Indices.pdf    PBS urban CPI by group, incl. Wheat
  indices_and_growth_rates_historical-1.pdf     PBS national CPI, UCPI, RPI, WPI (2015-16 = 100)

Outputs: data/derived/monthly_panel.csv (one row per month, Jan 2006 - latest)
         data/derived/cpi_urban_general.csv (PBS urban CPI, general index, both bases; used by sensitivity_v3.py)
"""
import re
import subprocess
import numpy as np
import pandas as pd

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
UP = RAW

# ----------------------------------------------------------------------------- WFP retail wheat (Rs per 40 kg)
w = pd.read_csv(UP + 'wfp_food_prices_pak.csv', parse_dates=['date'])
w = w[(w.commodity == 'Wheat') & (w.pricetype == 'Retail') & w.market.isin(['Lahore', 'Multan', 'Karachi', 'Peshawar'])].copy()
w['ym'] = w.date.dt.to_period('M')
w['rs40'] = w.price * 40
wide = w.pivot_table(index='ym', columns='market', values='rs40')
wide.columns = [f'wfp_{c.lower()}_rs40' for c in wide.columns]
wide['wfp_punjab_rs40'] = wide[['wfp_lahore_rs40', 'wfp_multan_rs40']].mean(axis=1)

# ----------------------------------------------------------------------------- policy rate (step function, forward-filled)
pr = pd.read_csv(UP + 'Policy_Rate.csv', skiprows=[1])
pr['ym'] = pd.to_datetime(pr['Period'], format='%b %Y').dt.to_period('M')
pr = pr.sort_values('ym').drop_duplicates('ym', keep='last').set_index('ym')['Policy Rate'].astype(float)

# ----------------------------------------------------------------------------- exchange rate (monthly average and month end)
fx = pd.read_csv(UP + 'Exchange_Rates_-_PKR_US_.csv', skiprows=[1])
fx['date'] = pd.to_datetime(fx['Date'], format='%d-%b-%Y')
fx['ym'] = fx.date.dt.to_period('M')
fx = fx.sort_values('date')
fx_m = pd.DataFrame({'fx_avg': fx.groupby('ym')['Exchange Rate'].mean(), 'fx_end': fx.groupby('ym')['Exchange Rate'].last()})
fx_m['fx_source'] = 'SBP'
# Before Jul 2015 the SBP file is empty: splice in the Kaggle daily USD/PKR series (1992-2023).
# Over the overlap Jul 2015 - Nov 2023 it matches SBP monthly means to -0.05% on average (max 1.2%).
kg = pd.read_csv(RAW + 'USD_To_PKR_1992_kaggle.csv', encoding='utf-8-sig')
kg.columns = [c.strip() for c in kg.columns]
kg['date'] = pd.to_datetime(kg['Date'].str.strip(), format='%d/%m/%Y')
kg['Rate'] = pd.to_numeric(kg['Rate'], errors='coerce')
kg['ym'] = kg.date.dt.to_period('M'); kg = kg.sort_values('date')
kg_m = pd.DataFrame({'fx_avg': kg.groupby('ym').Rate.mean(), 'fx_end': kg.groupby('ym').Rate.last()})
kg_m = kg_m[kg_m.index < fx_m.index.min()]; kg_m['fx_source'] = 'Kaggle USD_To_PKR_1992'
fx_m = pd.concat([kg_m, fx_m]).sort_index()

# ----------------------------------------------------------------------------- World Bank Pink Sheet wheat (USD per tonne)
raw = pd.read_excel(UP + 'CMO-Historical-Data-Monthly.xlsx', sheet_name='Monthly Prices', header=None)
names = [str(x) for x in raw.iloc[4].tolist()]
body = raw.iloc[6:].copy()
body.columns = ['period'] + names[1:]
body['ym'] = pd.PeriodIndex(body['period'].astype(str).str.replace('M', '-'), freq='M')
body = body.set_index('ym')
wheat_cols = [c for c in body.columns if 'wheat' in c.lower()]
wb = body[wheat_cols].apply(pd.to_numeric, errors='coerce')
wb.columns = ['wb_' + re.sub(r'[^a-z0-9]+', '_', c.lower()).strip('_') + '_usd_mt' for c in wb.columns]

# ----------------------------------------------------------------------------- PBS price indices from PDFs
def pdf_text(path):
    return subprocess.run(['pdftotext', '-layout', path, '-'], capture_output=True, text=True).stdout


def parse_groupwise(text, wanted):
    """Return {label: {Period: value}} for rows like '3 01 01 Wheat 98.01 ... 100.74' under 'SrNo ... Jul YY ...' headers."""
    out = {v: {} for v in wanted.values()}
    base_by_fy = {}
    fy = None
    for line in text.splitlines():
        if line.strip().startswith('SrNo'):
            m = re.search(r'Jul\s+(\d\d)', line)
            if m:
                fy = 2000 + int(m.group(1))
            continue
        if fy is None:
            continue
        toks = line.split()
        if len(toks) < 17:
            continue
        try:
            nums = [float(t) for t in toks[-13:]]
        except ValueError:
            continue
        head = toks[:-13]
        if len(head) < 4 or not all(h.isdigit() for h in head[:3]):
            continue
        desc = ' '.join(head[3:]).strip().lower()
        if desc in wanted:
            lab = wanted[desc]
            for i, v in enumerate(nums[:12]):
                yy, mm = (fy, 7 + i) if i < 6 else (fy + 1, i - 5)
                out[lab][pd.Period(f'{yy}-{mm:02d}', 'M')] = v
    return out


wanted = {'general': 'general', 'wheat': 'wheat', 'wheat flour': 'wheat_flour'}
rpi = parse_groupwise(pdf_text(UP + 'RPI-Gropwise-Cumulative-Indices.pdf'), wanted)
cpiu = parse_groupwise(pdf_text(UP + 'CpI-Urban-Groupwise-Cumulative-Indices.pdf'), wanted)
rpi = pd.DataFrame({f'pbs_rpi_{k}_idx': pd.Series(v) for k, v in rpi.items()})
cpiu = pd.DataFrame({f'pbs_cpiurban_{k}_idx_raw': pd.Series(v) for k, v in cpiu.items()})
# The urban file switches base at Jul 2019 (2007-08=100 before, 2015-16=100 after): keep the new base only
for c in list(cpiu.columns):
    new = cpiu[c].copy(); new[new.index < pd.Period('2019-07', 'M')] = np.nan
    cpiu[c.replace('_raw', '_new_base')] = new
cpiu = cpiu[[c for c in cpiu.columns if c.endswith('_new_base')]]

# national CPI (2015-16 = 100), monthly from Jul 2017
txt = pdf_text(UP + 'indices_and_growth_rates_historical-1.pdf')
txt = txt.split('Inflation Rate')[0]
natcpi, cur_year = {}, None
for line in txt.splitlines():
    t = line.split()
    if not t:
        continue
    try:
        if len(t) == 1 and re.fullmatch(r'20\d\d', t[0]):
            cur_year = int(t[0]); continue
        if len(t) >= 6 and re.fullmatch(r'20\d\d', t[0]) and t[1].isdigit() and 1 <= int(t[1]) <= 12:
            cur_year, mon, vals = int(t[0]), int(t[1]), t[2:]
        elif cur_year and len(t) >= 4 and t[0].isdigit() and 1 <= int(t[0]) <= 12:
            mon, vals = int(t[0]), t[1:]
        else:
            continue
        natcpi[pd.Period(f'{cur_year}-{mon:02d}', 'M')] = float(vals[0])
    except ValueError:
        continue
natcpi = pd.Series(natcpi, name='pbs_national_cpi_idx')

# ----------------------------------------------------------------------------- merge
idx = pd.period_range('2006-01', '2026-11', freq='M')
P = pd.DataFrame(index=idx)
P = P.join(wide).join(fx_m).join(wb).join(rpi).join(cpiu).join(natcpi)
P['policy_rate_pct'] = 100 * pr.reindex(idx).ffill()          # per cent per annum (file stores fractions)
first_pr = pr.index.min()
P.loc[P.index < first_pr, 'policy_rate_pct'] = np.nan
# world wheat in PKR per 40 kg: USD per tonne x PKR per USD / 25 (1 tonne = 25 x 40 kg)
hrw = [c for c in P.columns if 'hrw' in c]
if hrw:
    P['world_hrw_rs40'] = P[hrw[0]] * P['fx_avg'] / 25
    P['gap_punjab_vs_world'] = np.log(P['wfp_punjab_rs40'] / P['world_hrw_rs40'])
P.index.name = 'month'
P.reset_index().assign(month=lambda d: d['month'].astype(str)).to_csv(DERIVED + 'monthly_panel.csv', index=False)

# urban CPI, general index, as printed (2007-08 = 100 up to Jun 2019, 2015-16 = 100 after); the real-terms check in
# sensitivity_v3.py chains it to the national CPI from Aug 2017. Same parser as used there originally.
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


urb = urban_general(pdf_text(UP + 'CpI-Urban-Groupwise-Cumulative-Indices.pdf'))
pd.DataFrame({'month': urb.index.astype(str), 'cpi_urban_general': urb.values}).to_csv(DERIVED + 'cpi_urban_general.csv', index=False)

# ----------------------------------------------------------------------------- diagnostics
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 100)
print('World Bank wheat columns:', wheat_cols)
print('Policy rate steps:'); print((pr * 100).round(2).to_string() if len(pr) < 60 else (pr * 100).round(2).tail(40).to_string())
print('\nFX annual mean:'); print(P.groupby(P.index.year).fx_avg.mean().round(1).to_string())
print('\nCoverage (first, last non-null month):')
for c in P.columns:
    s = P[c].dropna()
    print(f'  {c:38s} {str(s.index.min()) if len(s) else "-":>8s} {str(s.index.max()) if len(s) else "-":>8s}  n={len(s)}')
