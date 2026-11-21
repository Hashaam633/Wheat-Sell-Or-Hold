"""
Parse Table 145 "Monthly Wholesale Price of Wheat" (Rs per 40 kg) from the Agricultural Statistics of Pakistan
(Ministry of National Food Security and Research; data from provincial marketing departments and PBS).

Inputs : ASP_for_Web_2024.pdf (fiscal years 2006-07 to 2023-24), ASP_2024-25_FINAL.pdf (adds 2024-25)
Output : data/derived/pbs_wholesale_wheat.csv  (market, month, rs40, source_edition, note)

Rules
 - Rows are fiscal years (July-June); '-' = missing.
 - 2006-07 to 2023-24 are taken from the 2024 edition (it keeps decimals); 2024-25 from the 2024-25 edition.
 - The 2024-25 row is about 2.5 times the 2023-24 level in every market (e.g. Lahore June 2024 3,050 -> July 2024 8,000),
   while retail prices did not jump. It matches prices per 100 kg, so it is divided by 2.5 and flagged.
"""
import re
import subprocess
import sys

import pandas as pd

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
ATT = sys.argv[1] if len(sys.argv) > 1 else RAW
FILES = {'2024': ATT + 'ASP_for_Web_2024.pdf', '2024-25': ATT + 'ASP_2024-25_FINAL.pdf'}
MARKETS = ['Lahore', 'Rawalpindi', 'Multan', 'Hyderabad', 'Sukkur', 'Peshawar']
MONTHS = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]


def parse(path, edition):
    txt = subprocess.run(['pdftotext', '-layout', path, '-'], capture_output=True, text=True).stdout.splitlines()
    starts = [i for i, l in enumerate(txt) if 'Monthly Wholesale Price of Wheat' in l][1:]      # skip table of contents
    end = [i for i, l in enumerate(txt) if 'Monthly Wholesale Prices of Rice Basmati' in l and i > starts[0]][0]
    rows, market = [], None
    for line in txt[starts[0]:end]:
        s = line.strip()
        if s in MARKETS:
            market = s; continue
        m = re.match(r'^(\d{4})-(\d{2})\s+(.*)$', s)
        if not (m and market):
            continue
        toks = m.group(3).split()
        if len(toks) != 13:
            raise ValueError(f'{edition} {market} {m.group(1)}: expected 13 values, got {len(toks)}: {s}')
        fy0 = int(m.group(1))
        for mon, t in zip(MONTHS, toks[:12]):
            yr = fy0 if mon >= 7 else fy0 + 1
            rows.append(dict(market=market, month=f'{yr}-{mon:02d}', rs40=None if t == '-' else float(t), edition=edition, fy=f'{fy0}-{str(fy0 + 1)[2:]}'))
    return pd.DataFrame(rows)


a = parse(FILES['2024'], '2024')
b = parse(FILES['2024-25'], '2024-25')
b = b[b.fy == '2024-25'].copy()
b['note'] = 'reported value / 2.5 (row appears to be Rs per 100 kg)'
b['rs40'] = b['rs40'] / 2.5
a['note'] = ''
d = pd.concat([a, b]).sort_values(['market', 'month']).reset_index(drop=True)
d.to_csv(DERIVED + 'pbs_wholesale_wheat.csv', index=False)

# consistency check against the 2024-25 edition for overlapping years
chk = parse(FILES['2024-25'], '2024-25')
j = a.merge(chk, on=['market', 'month'], suffixes=('_24', '_25')).dropna(subset=['rs40_24', 'rs40_25'])
print('rows', len(d), '| missing', int(d.rs40.isna().sum()), '| overlap rows', len(j),
      '| max |2024 edition - 2024-25 edition|', round((j.rs40_24 - j.rs40_25).abs().max(), 2))
w = d.pivot(index='month', columns='market', values='rs40')
print(w.loc['2024-03':'2024-09'].round(0).to_string())
print(w.loc['2025-03':'2025-06'].round(0).to_string())
