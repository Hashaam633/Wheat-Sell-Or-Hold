"""
Combine the forecast files written by the Colab notebook (colab/wheat_benchmark_colab.ipynb) into one file.

Input : results/colab_raw/*.csv.gz   one file per foundation model, and one per April refit for DeepAR and TFT
Output: results/colab_forecasts.csv.gz  (model, unique_id, origin, h, q0.05 ... q0.95; prices in Rs per 40 kg)
The notebook already sets the 0.05 and 0.95 quantiles of Chronos-Bolt and TimesFM 2.5 to their 0.1 and 0.9 values
(those models issue quantiles from 0.1 to 0.9 only), so the files are only concatenated here.
"""
import glob

import pandas as pd

from paths import RESULTS

files = sorted(glob.glob(RESULTS + 'colab_raw/*.csv.gz'))
C = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
key = ['model', 'unique_id', 'origin', 'h']
assert not C.duplicated(key).any(), 'duplicate forecast rows'
C.to_csv(RESULTS + 'colab_forecasts.csv.gz', index=False, compression='gzip', float_format='%.3f')
print(len(files), 'files,', len(C), 'rows;', C.model.value_counts().to_dict())
