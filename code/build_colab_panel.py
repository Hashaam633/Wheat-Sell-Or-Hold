"""
Build the multi-market monthly panel used by the benchmark (local models and the Colab notebook).

Series (all in rupees per 40 kg):
  wfp_wheat_<city>   WFP retail wheat grain: Lahore, Multan, Karachi, Peshawar, Quetta; plus wfp_wheat_Punjab = mean(Lahore, Multan)
  fao_flour_<city>   FAO GIEWS FPMA retail wheat flour (PKR/kg x 40): Lahore, Multan, Karachi, Peshawar, Quetta
  pbs_whsl_<city>    PBS / Agricultural Policy Institute wholesale wheat: Lahore, Rawalpindi, Multan, Hyderabad, Sukkur, Peshawar
Gap rules (no look-ahead): gaps of up to 3 months are forward-filled; if a longer gap exists, the series starts after the last
long gap. A final observation equal to the one before it is treated as a carry-forward and dropped.
Output: data/derived/colab_panel.csv (unique_id, ds, y), panel_flags.csv and panel_series_info.csv.
"""
import numpy as np
import pandas as pd

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
UP = RAW

series = {}
w = pd.read_csv(UP + 'wfp_food_prices_pak.csv', parse_dates=['date'])
w = w[(w.commodity == 'Wheat') & (w.pricetype == 'Retail')].copy()
w['ym'] = w.date.dt.to_period('M')
for city in ['Lahore', 'Multan', 'Karachi', 'Peshawar', 'Quetta']:
    series[f'wfp_wheat_{city}'] = w[w.market == city].groupby('ym').price.mean() * 40
series['wfp_wheat_Punjab'] = pd.concat([series['wfp_wheat_Lahore'], series['wfp_wheat_Multan']], axis=1).dropna().mean(axis=1)

f = pd.read_excel(RAW + 'fao_fpma_pakistan_wheat_flour_retail.xlsx', header=0)   # FAO GIEWS FPMA export, see data/README.md
f['ym'] = pd.to_datetime(f['Date'], format='%m/%d/%Y').dt.to_period('M')
for col in f.columns[2:-1]:
    city = col.split(',')[2].strip()
    series[f'fao_flour_{city}'] = f.set_index('ym')[col].astype(float).sort_index() * 40

p = pd.read_csv(DERIVED + 'pbs_wholesale_wheat.csv')
p['ym'] = pd.PeriodIndex(p['month'], freq='M')
for city in ['Lahore', 'Rawalpindi', 'Multan', 'Hyderabad', 'Sukkur', 'Peshawar']:
    series[f'pbs_whsl_{city}'] = p[p.market == city].set_index('ym').rs40.sort_index()

rows, info, flags = [], [], []
for sid, s in series.items():
    s = s.dropna().sort_index()
    full = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq='M'))
    miss = full.isna()
    run = miss.groupby((~miss).cumsum()).transform('sum').where(miss, 0)          # length of each missing run
    long_gap_end = full.index[(run > 3) & miss]
    start = (long_gap_end.max() + 1) if len(long_gap_end) else full.index.min()
    imputed = miss.loc[start:].copy()
    full = full.loc[start:].ffill()
    dropped_last = False
    if len(full) > 1 and abs(full.iloc[-1] - full.iloc[-2]) < 1e-9:
        full = full.iloc[:-1]; dropped_last = True
    n_filled = int(miss.loc[start:].sum())
    info.append(dict(unique_id=sid, start=str(full.index.min()), end=str(full.index.max()), n=len(full),
                     months_forward_filled=n_filled, last_month_dropped=dropped_last))
    rows += [(sid, str(ix) + '-01', round(float(v), 3)) for ix, v in full.items()]
    flags += [(sid, str(ix) + '-01', bool(imputed.get(ix, False))) for ix in full.index]

panel = pd.DataFrame(rows, columns=['unique_id', 'ds', 'y'])
pd.DataFrame(flags, columns=['unique_id', 'ds', 'imputed']).to_csv(DERIVED + 'panel_flags.csv', index=False)
panel.to_csv(DERIVED + 'colab_panel.csv', index=False)
I = pd.DataFrame(info); I.to_csv(DERIVED + 'panel_series_info.csv', index=False)
print(I.to_string(index=False)); print('rows', len(panel))
