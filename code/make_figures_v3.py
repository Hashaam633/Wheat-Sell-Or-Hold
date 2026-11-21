"""Figures from the v3 benchmark and decision analysis. Reads benchmark/*.csv; writes figures/fig6_*.png onwards."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from decisions_v3 import u, u_inv

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
FIG = FIGS; os.makedirs(FIG, exist_ok=True)
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'violet': '#6250d6', 'grey': '#8a8984', 'ink': '#0b0b0b', 'ink2': '#52514e'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9,
                     'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': C['ink2'], 'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'legend.frameon': False, 'savefig.dpi': 300, 'axes.axisbelow': True})
W = 6.3
MAIN = ['snaive', 'm0', 'sarima', 'ets', 'ecm', 'msar', 'lgbm', 'qrf', 'deepar', 'tft', 'm5', 'ens']

# ---------------------------------------------------------------- Figure 6: calibration and the value of the distribution
CV = pd.read_csv(B + 'calibration_value.csv', keep_default_na=False)
X = pd.read_csv(B + 'decisions_v3_rows.csv.gz')
X = X[(X.tier == 'primary') & (X.gamma == 2) & (X.rule == 'B') & (X.costs == 'base') & X.season.between(2013, 2025)
      & X.model.isin(MAIN + [m + '+R' for m in MAIN])]
ce = X.groupby(['model', 'form', 'season']).V.apply(lambda v: 100 * (u_inv(u(v.to_numpy(), 2).mean(), 2) - 1)).unstack('form')
G = (ce['prob'] - ce['point']).unstack('model')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.5), gridspec_kw={'width_ratios': [1, 1.25]})
col = {'R1': C['blue'], 'T2024': C['orange'], '2025': C['aqua']}
lab = {'R1': 'Support price, 2013–2023', 'T2024': 'Transition, 2024', '2025': '2025 season'}
for r in ['R1', 'T2024', '2025']:
    d = CV[CV.regime == r]
    raw, rec = d[~d.recal.astype(str).eq('True')], d[d.recal.astype(str).eq('True')]
    a1.scatter(raw.cov80, raw.gap_pp, s=22, facecolors='white', edgecolors=col[r], linewidths=1.1, zorder=3)
    a1.scatter(rec.cov80, rec.gap_pp, s=22, color=col[r], edgecolors='white', linewidths=0.6, zorder=3, label=lab[r])
a1.axvline(0.80, color=C['ink2'], lw=0.8, ls=':'); a1.axhline(0, color='#b5b4ae', lw=0.8)
YLO = float(np.floor(min(CV.gap_pp.min(), np.nanmin(G.to_numpy())) - 4)); YHI = float(np.ceil(max(CV.gap_pp.max(), np.nanmax(G.to_numpy())) + 3))
a1.text(0.81, YHI - 1, 'nominal 80%', fontsize=7, color=C['ink2'], va='top')
a1.set_xlabel('Coverage of 80% intervals (decision forecasts)'); a1.set_ylabel('Probabilistic minus point (CE, % points)')
a1.set_xlim(0.1, 1.03); a1.set_ylim(YLO, YHI)
h1 = a1.scatter([], [], s=22, facecolors='white', edgecolors=C['grey'], linewidths=1.1, label='as issued (hollow)')
h2 = a1.scatter([], [], s=22, color=C['grey'], label='recalibrated (filled)')
a1.legend(fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.3, columnspacing=0.8)
a1.set_title('(a) Calibration and decision value', fontsize=8.5, color=C['ink'], loc='left')

seasons = G.index.to_numpy()
rng = np.random.default_rng(3)
for i, s in enumerate(seasons):
    v = G.loc[s].to_numpy()
    jit = rng.uniform(-0.22, 0.22, len(v))
    a2.scatter(np.full(len(v), i) + jit, v, s=9, color=C['grey'], alpha=0.55, lw=0, zorder=2)
a2.plot(range(len(seasons)), G['ens+R'].to_numpy(), color=C['blue'], lw=1.4, marker='o', ms=3.5, zorder=4, label='Recalibrated ensemble (ens+R)')
a2.plot(range(len(seasons)), G['m5+R'].to_numpy(), color=C['orange'], lw=1.4, marker='o', ms=3.5, zorder=4, label='Regime-aware model (M5+R)')
a2.scatter([], [], s=9, color=C['grey'], label='Each of 24 forecasters')
a2.axhline(0, color='#b5b4ae', lw=0.8)
a2.set_xticks(range(len(seasons))); a2.set_xticklabels([str(s)[2:] for s in seasons])
a2.set_xlabel('Harvest season (20xx)'); a2.set_ylim(YLO, YHI)
a2.legend(fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, handletextpad=0.3)
a2.set_title('(b) The same gap, season by season', fontsize=8.5, color=C['ink'], loc='left')
fig.tight_layout(); fig.savefig(FIG + 'fig3_calibration_value.png'); plt.close(fig)
print('fig6 written')

# ---------------------------------------------------------------- Figure 7 (paper v1.0 Figure 2): accuracy vs calibration, before/after recalibration
T = pd.read_csv(B + 'scores_by_tier_regime.csv')
T = T[(T.tier == 'primary') & (T.regime == 'All')].set_index('model')
short = {'snaive': 'S-naive', 'm0': 'M0', 'sarima': 'SARIMA', 'ets': 'ETS', 'ecm': 'ECM', 'msar': 'MS-AR', 'lgbm': 'LightGBM',
         'qrf': 'QRF', 'deepar': 'DeepAR', 'tft': 'TFT', 'm5': 'M5', 'ens': 'Ensemble'}
# v1.9: a second panel zooms on the crowded area near nominal coverage and labels every recalibrated model there
from matplotlib.patches import Rectangle
ZX, ZY = (0.615, 0.805), (0.0945, 0.1115)                    # zoom window (coverage, quantile score)
fig, (ax, az) = plt.subplots(2, 1, figsize=(W, 6.6), gridspec_kw={'height_ratios': [1.15, 1]})
for m in MAIN:
    x0, y0 = T.loc[m, 'cov80'], T.loc[m, 'qs']; x1, y1 = T.loc[m + '+R', 'cov80'], T.loc[m + '+R', 'qs']
    c = C['blue'] if m == 'm5' else (C['orange'] if m == 'ens' else C['grey'])
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=c, lw=0.9, shrinkA=3, shrinkB=3, mutation_scale=7))
    az.plot([x0, x1], [y0, y1], color=c, lw=0.9, zorder=2)            # plain line, clipped at the panel edge
    az.annotate('', xy=(x1, y1), xytext=(x0 + 0.85 * (x1 - x0), y0 + 0.85 * (y1 - y0)),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=0.9, shrinkA=0, shrinkB=3, mutation_scale=7))
    for a_ in (ax, az):
        a_.scatter([x0], [y0], s=26, facecolors='white', edgecolors=c, linewidths=1.1, zorder=3)
        a_.scatter([x1], [y1], s=26, color=c, edgecolors='white', linewidths=0.6, zorder=4)
    lab_pos = {'snaive': (-6, -2, 'right'), 'deepar': (0, 6, 'center'), 'tft': (-6, -2, 'right'), 'lgbm': (-6, -2, 'right'),
               'qrf': (-6, -2, 'right'), 'ets': None, 'm0': (-6, -2, 'right'), 'ecm': (-6, -4, 'right'),
               'sarima': None, 'msar': None, 'm5': (-6, -2, 'right'), 'ens': None}[m]
    if lab_pos:
        ax.annotate(short[m], (x0, y0), xytext=lab_pos[:2], textcoords='offset points', fontsize=7, color=C['ink2'], ha=lab_pos[2])
    end_pos = {'qrf': (6, 5, 'left'), 'lgbm': (6, -2, 'left'), 'ets': (-6, -3, 'right'), 'm0': (6, 1, 'left'), 'sarima': (-6, -3, 'right'),
               'msar': (6, 1, 'left'), 'ecm': (6, -2, 'left'), 'ens': (6, -3, 'left')}.get(m)
    if end_pos and ZX[0] <= x1 <= ZX[1] and ZY[0] <= y1 <= ZY[1]:
        az.annotate(short[m] + ' +R', (x1, y1), xytext=end_pos[:2], textcoords='offset points', fontsize=7, color=C['ink'], ha=end_pos[2], va='center')
for m, txt, off, ha in [('ets', 'ETS', (0, 9), 'center'), ('sarima', 'SARIMA', (-2, 9), 'center'), ('msar', 'MS-AR and ensemble', (5, 8), 'left')]:
    az.annotate(txt + ' (as issued)', (T.loc[m, 'cov80'], T.loc[m, 'qs']), xytext=off, textcoords='offset points', fontsize=6.5, color=C['ink2'], ha=ha)
ax.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0], fill=False, ec=C['ink2'], lw=0.7, ls='--', zorder=1))
ax.text(ZX[0] + 0.003, ZY[1] + 0.0008, '(b)', fontsize=7, color=C['ink2'], va='bottom')
for a_ in (ax, az):
    a_.axvline(0.80, color=C['ink2'], lw=0.8, ls=':')
ax.text(0.803, 0.141, 'nominal 80%', fontsize=7, color=C['ink2'], va='top')
ax.set_ylabel('Quantile score (log points, lower is better)')
ax.scatter([], [], s=26, facecolors='white', edgecolors=C['grey'], linewidths=1.1, label='as issued')
ax.scatter([], [], s=26, color=C['grey'], label='after adaptive recalibration')
ax.scatter([], [], s=26, color=C['blue'], label='M5 (regime-aware)'); ax.scatter([], [], s=26, color=C['orange'], label='Equal-weight ensemble')
ax.legend(fontsize=7, loc='lower left', ncol=2)
ax.set_xlim(0.22, 0.86); ax.set_ylim(0.078, 0.147)
ax.set_title('(a) All models: as issued (hollow) and after recalibration (filled)', fontsize=8.5, color=C['ink'], loc='left')
az.set_xlim(*ZX); az.set_ylim(*ZY)
az.set_xlabel('Coverage of 80% intervals (primary panel, all origins)'); az.set_ylabel('Quantile score (log points)')
az.set_title('(b) Detail of the dashed box: each recalibrated model labeled (+R)', fontsize=8.5, color=C['ink'], loc='left')
fig.tight_layout(); fig.savefig(FIG + 'fig2_accuracy_calibration.png'); plt.close(fig)
print('fig7 written')

# ---------------------------------------------------------------- Figure 8 (paper v1.0 Figure 4): M5 minus SARIMA by season, with ablations
A = pd.read_csv(B + 'ablation_decisions.csv')
A = A[(A.tier == 'primary') & (A.gamma == 2) & (A.form == 'prob')]
sc = A.groupby(['model', 'season']).V.apply(lambda v: 100 * (u_inv(u(v.to_numpy(), 2).mean(), 2) - 1)).unstack('model')
var = [('m5_full', 'Full M5', C['blue']), ('m5_nofloor', 'Without the support-price floor', C['aqua']), ('m5_nogap', 'Without the world-price term', C['grey'])]
fig, ax = plt.subplots(figsize=(W, 2.9))
xs = np.arange(len(sc.index)); wd = 0.26
for i, (m, lab, c) in enumerate(var):
    ax.bar(xs + (i - 1) * wd, (sc[m] - sc['sarima']).to_numpy(), width=wd - 0.03, color=c, label=lab, zorder=3)
ax.axhline(0, color='#b5b4ae', lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in sc.index], fontsize=7.5)
ax.set_ylabel('CE gain minus SARIMA (% points)'); ax.legend(fontsize=7, loc='upper left')
fig.tight_layout(); fig.savefig(FIG + 'fig4_m5_ablation_by_season.png'); plt.close(fig)
print('fig8 written')
