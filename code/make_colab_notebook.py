"""Write wheat_benchmark_colab.ipynb: foundation and deep-learning forecasts for the benchmark, run on a free Colab GPU."""
import json

import numpy as np
import pandas as pd

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
panel_csv = open(DERIVED + 'colab_panel.csv').read()
P = pd.read_csv(DERIVED + 'monthly_panel.csv')
w = P[['month', 'world_hrw_rs40']].copy()
w['world_rs40_lag1'] = w['world_hrw_rs40'].shift(1)              # last month's world price: known at the decision date
w = w.dropna(subset=['world_rs40_lag1'])
w['ds'] = w['month'] + '-01'
world_csv = w[['ds', 'world_rs40_lag1']].round(2).to_csv(index=False)

HELPERS = r'''
import os
import numpy as np
import pandas as pd

H = 12                                              # forecast horizon in months
QL = [round(0.05 * i, 2) for i in range(1, 20)]     # quantile levels 0.05 ... 0.95
ORIGIN_START = pd.Timestamp('2010-01-01')           # first forecast origin
MIN_CONTEXT = 36                                    # at least 3 years of history before forecasting
OUTDIR = '/content/drive/MyDrive/wheat_benchmark_outputs'   # Google Drive: survives disconnects
os.makedirs(OUTDIR, exist_ok=True)


def done(name):
    """True if this result is already saved in Drive (so a re-run skips it)."""
    ok = os.path.exists(f'{OUTDIR}/{name}.csv.gz')
    if ok:
        print(f'{name}: already saved in Drive, skipping')
    return ok


def load_panel():
    df = pd.read_csv('/content/panel.csv', parse_dates=['ds'])
    return df.sort_values(['unique_id', 'ds']).reset_index(drop=True)


def load_world():
    return pd.read_csv('/content/world.csv', parse_dates=['ds']).set_index('ds')['world_rs40_lag1']


def make_origins(df):
    """Every (series, month) with enough history. The context stops at the origin month: no look-ahead."""
    out = []
    for uid, g in df.groupby('unique_id', sort=True):
        y = g['y'].to_numpy(dtype='float32'); ds = list(g['ds'])
        for i in range(MIN_CONTEXT - 1, len(g)):
            if ds[i] >= ORIGIN_START:
                out.append((uid, ds[i], y[: i + 1], ds[: i + 1]))
    return out


def to_wide(model, O, Q):
    """Q has shape (number of origins, H, len(QL)); one output row per origin and horizon."""
    n = len(O)
    assert Q.shape == (n, H, len(QL)), Q.shape
    rec = {'model': np.repeat(model, n * H),
           'unique_id': np.repeat([o[0] for o in O], H),
           'origin': np.repeat([o[1].strftime('%Y-%m') for o in O], H),
           'h': np.tile(np.arange(1, H + 1), n)}
    for j, q in enumerate(QL):
        rec[f'q{q:.2f}'] = Q[:, :, j].reshape(-1)
    return pd.DataFrame(rec)


def save(df, name):
    path = f'{OUTDIR}/{name}.csv.gz'
    df.to_csv(path, index=False, compression='gzip', float_format='%.3f')
    print(f'saved {path}: {len(df):,} rows')
'''

cells = []
md = lambda s: cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s})
code = lambda s: cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s})

md("""# Wheat sell-or-hold paper: foundation and deep-learning benchmark

This notebook makes probabilistic price forecasts for 17 Pakistani wheat price series (WFP retail wheat, FAO retail flour, PBS wholesale wheat) with models that need a GPU and internet access:

| Section | Models | Typical time on a T4 GPU |
|---|---|---|
| 2 | Chronos-Bolt (base), Chronos-2, Chronos-2 with world wheat price, TimesFM 2.5 | 5–15 min |
| 3 | DeepAR, Temporal Fusion Transformer (trained here, refitted every year) | 30–60 min |

Forecasts are made at every month from 2010 onward, 1–12 months ahead, as 19 quantiles. Each forecast uses only data up to its origin month.

**How to run**
1. *Runtime → Change runtime type → T4 GPU → Save.*
2. *Runtime → Run all.* Allow Google Drive access when asked.
3. **Every result is saved to your Google Drive** (folder `wheat_benchmark_outputs`) as soon as it is made. If the runtime disconnects, just reconnect and *Run all* again: finished models and finished years are skipped, and work continues where it stopped.
4. If a cell asks you to **restart the session** after installing packages, click restart, then *Run all* again.
5. Nothing needs to be downloaded. When the last cell prints `ALL DONE`, the forecast files are in the Drive folder `wheat_benchmark_outputs`; copy them to `results/colab_raw/`.

Tips against disconnects: keep this tab open and visible, keep the laptop plugged in and stop it from sleeping, and don't run other Colab notebooks at the same time.""")

md('## 0. Connect Google Drive (results are saved there)')
code(r'''from google.colab import drive
drive.mount('/content/drive')
import os, glob, shutil
os.makedirs('/content/drive/MyDrive/wheat_benchmark_outputs', exist_ok=True)
# keep anything an earlier run left on the Colab disk
for p in glob.glob('/content/outputs/*.csv.gz'):
    dst = '/content/drive/MyDrive/wheat_benchmark_outputs/' + os.path.basename(p)
    if not os.path.exists(dst):
        shutil.copy(p, dst); print('copied earlier result to Drive:', os.path.basename(p))
print('Drive ready')''')
md('## 1. Data (written to the Colab disk)')
code('%%writefile /content/panel.csv\n' + panel_csv)
code('%%writefile /content/world.csv\n' + world_csv)
code('%%writefile /content/fc_helpers.py\n' + HELPERS.strip() + '\n')

md('## 2. Foundation models (zero-shot)')
code('!pip -q install "chronos-forecasting>=2.3.2,<3" "timesfm[torch]>=3.0.2"')
code(r'''import sys, time, traceback
sys.path.insert(0, '/content')
import numpy as np, pandas as pd, torch
from fc_helpers import *
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
df = load_panel(); world = load_world(); O = make_origins(df)
print(f'{len(O):,} forecast origins across {df.unique_id.nunique()} series | device: {dev}')
if dev == 'cpu':
    print('WARNING: no GPU found. Use Runtime > Change runtime type > T4 GPU, otherwise this will be slow.')''')

code(r'''# 2a. Chronos-Bolt base. It was trained on quantiles 0.1-0.9; levels outside that range are set to the 0.1 / 0.9 values.
from chronos import BaseChronosPipeline
try:
    if done('chronos_bolt_base'): raise StopIteration
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-base', device_map=dev, torch_dtype=torch.float32)
    res = []
    for i in range(0, len(O), 256):
        q, _ = pipe.predict_quantiles([torch.tensor(o[2]) for o in O[i:i + 256]], prediction_length=H, quantile_levels=QL)
        res.append(q.float().cpu().numpy())
    save(to_wide('chronos_bolt_base', O, np.concatenate(res)), 'chronos_bolt_base')
    print(f'Chronos-Bolt done in {time.time() - t0:.0f}s'); del pipe
except StopIteration:
    pass
except Exception:
    traceback.print_exc()''')

code(r'''# 2b. Chronos-2, univariate (own price history only). cross_learning=False keeps every series and origin separate.
try:
    if done('chronos2'): raise StopIteration
    t0 = time.time()
    pipe2 = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map=dev)
    res = []
    for i in range(0, len(O), 128):
        qs, _ = pipe2.predict_quantiles([o[2] for o in O[i:i + 128]], prediction_length=H, quantile_levels=QL, cross_learning=False)
        res.append(np.stack([x[0].float().cpu().numpy() for x in qs]))
    save(to_wide('chronos2', O, np.concatenate(res)), 'chronos2')
    print(f'Chronos-2 done in {time.time() - t0:.0f}s')
except StopIteration:
    pass
except Exception:
    traceback.print_exc()''')

code(r'''# 2c. Chronos-2 with a past covariate: last month's US wheat price in rupees (information set B).
try:
    if done('chronos2_worldcov'): raise StopIteration
    t0 = time.time()
    if 'pipe2' not in globals():
        pipe2 = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map=dev)
    def with_cov(o):
        cov = world.reindex(pd.DatetimeIndex(o[3])).bfill().ffill().to_numpy(dtype='float32')
        return {'target': o[2], 'past_covariates': {'world_rs40': cov}}
    res = []
    for i in range(0, len(O), 128):
        qs, _ = pipe2.predict_quantiles([with_cov(o) for o in O[i:i + 128]], prediction_length=H, quantile_levels=QL, cross_learning=False)
        res.append(np.stack([x[0].float().cpu().numpy() for x in qs]))
    save(to_wide('chronos2_worldcov', O, np.concatenate(res)), 'chronos2_worldcov')
    print(f'Chronos-2 + covariate done in {time.time() - t0:.0f}s'); del pipe2
except StopIteration:
    pass
except Exception:
    traceback.print_exc()''')

code(r'''# 2d. TimesFM 2.5 (Apache-2.0 weights). Output index 0 is the mean, 1-9 are the 0.1-0.9 quantiles.
try:
    if done('timesfm25'): raise StopIteration
    import timesfm
    t0 = time.time()
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch', torch_compile=False)
    tfm.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True, per_core_batch_size=64,
                                       use_continuous_quantile_head=True, force_flip_invariance=True,
                                       infer_is_positive=True, fix_quantile_crossing=True))
    _, qf = tfm.forecast(horizon=H, inputs=[o[2].astype('float64') for o in O])
    dec = np.asarray(qf)[:, :H, 1:10]
    levels = np.round(np.arange(0.1, 0.91, 0.1), 2)
    Q = np.stack([[np.interp(QL, levels, dec[i, h]) for h in range(H)] for i in range(len(O))])
    save(to_wide('timesfm25', O, Q), 'timesfm25')
    print(f'TimesFM done in {time.time() - t0:.0f}s'); del tfm
except StopIteration:
    pass
except Exception:
    traceback.print_exc()''')

md('## 3. Deep-learning models trained on the panel (refitted every April)')
code('!pip -q install neuralforecast')
code(r'''import sys, re, time, logging, warnings, traceback
sys.path.insert(0, '/content')
import numpy as np, pandas as pd
from fc_helpers import *
from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR, TFT
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
warnings.filterwarnings('ignore')
for name in ['pytorch_lightning', 'lightning', 'lightning.pytorch', 'lightning_fabric']:
    logging.getLogger(name).setLevel(logging.ERROR)

df = load_panel()
INPUT, STEPS = 24, 300
NAMES = ['DeepAR', 'TFT']

def build_models():
    common = dict(h=H, input_size=INPUT, max_steps=STEPS, scaler_type='robust', random_seed=1,
                  enable_progress_bar=False, enable_model_summary=False, logger=False)
    return [DeepAR(loss=DistributionLoss(distribution='StudentT', quantiles=QL), **common),
            TFT(loss=MQLoss(quantiles=QL), hidden_size=32, **common)]

def quantile_columns(fc, name):
    cols = {}
    for c in fc.columns:
        if not c.startswith(name + '-'):
            continue
        if c.endswith('-median'):
            cols[0.5] = c
        m = re.search(r'-(lo|hi)-([0-9.]+)$', c)
        if m:
            lvl = float(m.group(2)); q = (100 - lvl) / 200 if m.group(1) == 'lo' else 1 - (100 - lvl) / 200
            cols[round(q, 4)] = c
    return [cols[min(cols, key=lambda k: abs(k - q))] for q in QL]

cutoffs = pd.date_range('2010-04-01', df.ds.max(), freq='12MS')
t_all = time.time()
for cut in cutoffs:
    t0 = time.time()
    tag = f'deep_cutoff_{cut:%Y}'
    if done(tag):
        continue
    results = {n: ([], []) for n in NAMES}
    try:
        train = df[df.ds <= cut]
        n_obs = train.groupby('unique_id').size()
        train = train[train.unique_id.isin(n_obs[n_obs >= INPUT + H + 12].index)]
        nf = NeuralForecast(models=build_models(), freq='MS')
        nf.fit(df=train)
        last_origin = min(cut + pd.DateOffset(months=11), df.ds.max())
        for origin in pd.date_range(cut, last_origin, freq='MS'):
            hist = df[df.ds <= origin]
            last = hist.groupby('unique_id').ds.max(); n_h = hist.groupby('unique_id').size()
            ok = [u for u in last.index if last[u] == origin and n_h[u] >= MIN_CONTEXT]
            if not ok:
                continue
            fc = nf.predict(df=hist[hist.unique_id.isin(ok)])
            if 'unique_id' not in fc.columns:
                fc = fc.reset_index()
            fc = fc.sort_values(['unique_id', 'ds'])
            for name in NAMES:
                qc = quantile_columns(fc, name)
                for uid, g in fc.groupby('unique_id', sort=True):
                    q = np.sort(g[qc].to_numpy(dtype='float64')[:H], axis=1)
                    results[name][0].append((uid, origin)); results[name][1].append(q)
        parts = [to_wide(name.lower(), results[name][0], np.stack(results[name][1])) for name in NAMES if results[name][1]]
        if parts:
            save(pd.concat(parts), tag)                      # one Drive file per year: a disconnect loses at most one year
        print(f'cutoff {cut:%Y-%m}: trained on {train.unique_id.nunique()} series, done in {time.time() - t0:.0f}s')
    except Exception:
        print(f'cutoff {cut:%Y-%m} failed:'); traceback.print_exc()
print(f'all deep models done in {(time.time() - t_all) / 60:.0f} min')''')

md('## 4. Check what is saved in Drive')
code(r'''import glob, os, pandas as pd
OUTDIR = '/content/drive/MyDrive/wheat_benchmark_outputs'
files_ = sorted(glob.glob(OUTDIR + '/*.csv.gz'))
for p in files_:
    print(f'{os.path.basename(p):32s} {os.path.getsize(p) / 1e6:6.2f} MB')
need = ['chronos_bolt_base', 'chronos2', 'chronos2_worldcov', 'timesfm25'] + [f'deep_cutoff_{y}' for y in range(2010, pd.Timestamp.today().year + 1) if pd.Timestamp(f'{y}-04-01') <= pd.Timestamp('2026-10-01')]
missing = [n for n in need if not os.path.exists(f'{OUTDIR}/{n}.csv.gz')]
print('ALL DONE: the forecast files are in the Drive folder wheat_benchmark_outputs.' if not missing else f'Still missing: {missing}. Reconnect and Run all again.')''')

for c in cells:
    src = c['source']
    lines = src.split('\n')
    c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
json.dump(nb, open(COLAB + 'wheat_benchmark_colab.ipynb', 'w'), indent=1)
print('notebook written with', len(cells), 'cells')
