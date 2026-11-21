"""Repository-relative folders used by every script (run the scripts from any working directory)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw', '')                 # source files as downloaded (see data/README.md)
COMPILED = os.path.join(ROOT, 'data', 'compiled', '')       # tables compiled by hand from cited sources
DERIVED = os.path.join(ROOT, 'data', 'derived', '')         # panels built by the data scripts
RESULTS = os.path.join(ROOT, 'results', '')                 # forecasts, scores, decisions and tests
ARCHIVE = os.path.join(RESULTS, 'archive_v1.0', '')         # v1.0 outputs kept for the correction reported in the paper
ARCHIVE11 = os.path.join(RESULTS, 'archive_v1.1', '')       # v1.1-v1.3 outputs kept for the v1.4 correction (MS-AR)
ARCHIVE15 = os.path.join(RESULTS, 'archive_v1.5', '')       # v1.4-v1.5 outputs kept for the v1.6 correction (M5 floor)
FIGS = os.path.join(ROOT, 'figures', '')
EXPLORATORY = os.path.join(ROOT, 'exploratory', '')
COLAB = os.path.join(ROOT, 'colab', '')
