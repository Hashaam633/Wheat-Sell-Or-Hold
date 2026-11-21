"""Figure 1 of the paper: Punjab wheat prices, the support price, world parity and the policy regimes."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
FIG = FIGS; os.makedirs(FIG, exist_ok=True)
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'violet': '#6250d6', 'grey': '#8a8984', 'ink': '#0b0b0b', 'ink2': '#52514e'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9,
                     'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': C['ink2'], 'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'legend.frameon': False, 'savefig.dpi': 300, 'axes.axisbelow': True})
W = 6.3

P = pd.read_csv(DERIVED + 'monthly_panel.csv'); P['date'] = pd.PeriodIndex(P['month'], freq='M').to_timestamp(); P = P.set_index('date')
FL = pd.read_csv(COMPILED + 'support_prices.csv').set_index('harvest_year')

# ---------------------------------------------------------------- Figure 1: prices, floor, world parity, regimes
fig, ax = plt.subplots(figsize=(W, 2.9))
pun = P['wfp_punjab_rs40'].loc[:'2026-08']
ax.plot(pun.index, pun, color=C['blue'], lw=1.6, label='Punjab wheat grain (WFP, Lahore–Multan)')
ax.plot(P.index, P['world_hrw_rs40'], color=C['orange'], lw=1.0, ls='--', label='US hard red winter wheat in rupees (world price)')
xs, ys = [], []
for s, row in FL.iterrows():
    if pd.notna(row.support_price_rs40):
        xs += [pd.Timestamp(f'{s}-04-01'), pd.Timestamp(f'{s + 1}-03-31')]; ys += [row.support_price_rs40] * 2
    else:
        xs += [pd.Timestamp(f'{s}-04-01')]; ys += [np.nan]
ax.plot(xs, ys, color=C['ink2'], lw=1.2, drawstyle='steps-post', label='Support price (harvest year)')
for (a, b, lab, yy) in [('2024-04-01', '2025-03-31', '2024', 5300), ('2025-04-01', '2025-10-19', 'R2', 5300), ('2025-10-19', '2026-09-30', 'R3', 5300)]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='#e6e5e0' if lab != 'R2' else '#d6d5cf', alpha=0.7, lw=0)
    ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, yy, lab, ha='center', va='top', fontsize=6.5, color=C['ink2'])
ax.text(pd.Timestamp('2014-06-01'), 5300, 'R1 support price', ha='center', va='top', fontsize=7.5, color=C['ink2'])
ax.set_ylabel('Rupees per 40 kg'); ax.set_ylim(0, 5400); ax.set_xlim(pd.Timestamp('2006-01-01'), pd.Timestamp('2026-10-31'))
ax.legend(loc='upper left', bbox_to_anchor=(0, 0.92), fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG + 'fig1_prices.png'); plt.close(fig)
