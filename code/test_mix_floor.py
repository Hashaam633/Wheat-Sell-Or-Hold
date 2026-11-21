"""
Unit tests of the support-price floor mixture in the regime-aware model (benchmark_local._mix_floor, Equation 1).
Added in v1.6, after a review found that the earlier approximation pulled the upper quantiles toward the median even
when the floor did not bind. Stops with an error if any test fails.
"""
import numpy as np
from scipy.stats import norm

import benchmark_local as BL

QL = BL.QL
Q = 0.10 * norm.ppf(QL)                      # a forecast of the log change: normal, standard deviation 0.10


def old_mix_floor(Qrow, L, q):
    """The v1.0-v1.5 approximation (kept here only to show what changed)."""
    vals = np.concatenate([np.maximum(Qrow, L), Qrow]); w = np.concatenate([np.full(len(QL), q), np.full(len(QL), 1 - q)])
    o = np.argsort(vals, kind='stable'); vals, w = vals[o], w[o] / w.sum()
    return np.interp(QL, np.cumsum(w) - w / 2, vals)


def mc_mix(L, q, n=2_000_000, seed=1):
    """Monte Carlo quantiles of the mixture: with probability q the price cannot fall below the floor."""
    rng = np.random.default_rng(seed)
    y = 0.10 * rng.standard_normal(n)
    y = np.where(rng.random(n) < q, np.maximum(y, L), y)
    return np.quantile(y, QL)


checks = []
for q in (0.0, 0.5, 1.0):                    # a floor far below the forecast changes nothing
    checks.append((f'non-binding floor, q = {q}', np.abs(BL._mix_floor(Q, -1.0, q) - Q).max(), 1e-12))
checks.append(('q = 1 gives max(Q, L)', np.abs(BL._mix_floor(Q, -0.05, 1.0) - np.maximum(Q, -0.05)).max(), 1e-12))
for L, q in ((-0.05, 0.5), (-0.10, 0.5), (0.0, 0.5), (-0.05, 0.25), (-0.15, 0.8)):
    # the 19-point quantile function is piecewise linear, so allow a small interpolation error against the exact mixture
    checks.append((f'matches Monte Carlo, L = {L}, q = {q}', np.abs(BL._mix_floor(Q, L, q) - mc_mix(L, q)).max(), 0.006))

ok = True
for name, err, tol in checks:
    good = err <= tol; ok &= good
    print(f'{name:40s} max abs error {err:.2e}  (tolerance {tol:.0e})  {"OK" if good else "FAIL"}')
old = old_mix_floor(Q, -1.0, 1.0)
print(f'v1.0-v1.5 approximation with a non-binding floor and q = 1: 0.90 quantile {old[17]:.4f} instead of {Q[17]:.4f}, '
      f'0.95 quantile {old[18]:.4f} instead of {Q[18]:.4f}')
if not ok:
    raise SystemExit('floor mixture test failed')
print('ALL FLOOR-MIXTURE TESTS PASSED')
