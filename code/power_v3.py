"""
Power of the season-level tests (pre-analysis plan, Section 7: "Thirteen seasons limit power").

For a season-level contrast d (13 values) the exact one-sided sign-flip test at alpha = 0.05 is applied to simulated samples
d* = (bootstrap draw of d - mean(d)) + delta, i.e. the observed season-to-season spread with a true effect delta.
Power = share of 4,000 simulated samples rejected. Also reported: the minimum detectable effect at 80% power (sign-flip, simulated)
and the normal-theory value (t(0.95) + t(0.80)) * sd / sqrt(n).
v1.6 adds a second design that respects the seasons in which the two forecasters chose the same months (difference
exactly zero): each simulated season differs with the observed share pi of seasons with a nonzero difference, and only
those seasons carry the effect (nonzero differences drawn from the observed ones, centred, plus delta / pi, so the
average effect over all seasons is delta). With k nonzero seasons the exact one-sided test cannot give p below 2^-k, so
it can reject at 5% only if k >= 5; power is then capped at P(k >= 5).
Output: benchmark/power_curves.csv, key_numbers_power.json
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from decisions_v3 import u, u_inv

from paths import RAW, COMPILED, DERIVED, RESULTS, ARCHIVE, FIGS, EXPLORATORY, COLAB  # noqa: F401
B = RESULTS
X = pd.read_csv(B + 'decisions_v3_rows.csv.gz')
X = X[(X.tier == 'primary') & X.season.between(2013, 2025) & (X.gamma == 2) & (X.rule == 'B') & (X.costs == 'base') & (X.kbar == 8)]


def sce(m, f):
    d = X[(X.model == m) & (X.form == f)]
    return d.groupby('season').V.apply(lambda v: 100 * (u_inv(u(v.to_numpy(), 2).mean(), 2) - 1))


contrasts = {'H1: ens+R prob - point': sce('ens+R', 'prob') - sce('ens+R', 'point'),
             'H6: M5 - SARIMA (prob)': sce('m5', 'prob') - sce('sarima', 'prob'),
             'M5+R prob - M0+R prob': sce('m5+R', 'prob') - sce('m0+R', 'prob')}
n = 13
SG = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))           # 8192 x 13


def reject(D):
    """D: (R, n) samples -> boolean array of one-sided sign-flip rejections at 5%."""
    obs = D.mean(1)
    flips = (D @ SG.T) / n                                              # (R, 8192)
    p = (flips >= obs[:, None] - 1e-12).mean(1)
    return p <= 0.05


rng = np.random.default_rng(7)
rows, kn = [], {}
deltas = np.round(np.arange(0, 12.01, 0.5), 2)
for lab, d in contrasts.items():
    d = d.dropna().to_numpy(); c = d - d.mean()
    idx = rng.integers(0, n, (4000, n))
    base = c[idx]
    pw = [float(reject(base + dl).mean()) for dl in deltas]
    rows += [dict(contrast=lab, delta_pp=dl, power=p) for dl, p in zip(deltas, pw)]
    fine = np.arange(0, 15.01, 0.1)
    pw_f = np.array([reject(base[:1500] + dl).mean() for dl in fine])
    mde_sim = float(fine[np.argmax(pw_f >= 0.8)]) if (pw_f >= 0.8).any() else np.nan
    sd = d.std(ddof=1)
    mde_t = float((tdist.ppf(0.95, n - 1) + tdist.ppf(0.80, n - 1)) * sd / np.sqrt(n))
    kn[lab] = dict(observed_mean_pp=float(d.mean()), sd_pp=float(sd), mde80_signflip_pp=mde_sim, mde80_normal_theory_pp=mde_t,
                   power_at_observed=float(np.interp(abs(d.mean()), fine, pw_f)), size_at_zero=pw[0])
    # v1.6: effects only in seasons whose decisions differ
    nz = d[np.abs(d) > 1e-9]; pi = len(nz) / n
    if len(nz) >= 2:
        cz = nz - nz.mean()
        Z = rng.random((2000, n)) < pi
        draws = cz[rng.integers(0, len(nz), (2000, n))]
        pw_h = []
        for dl in fine:
            Dh = np.where(Z, draws + dl / pi, 0.0)
            pw_h.append(float(reject(Dh).mean()))
        pw_h = np.array(pw_h)
        from scipy.stats import binom
        kn[lab].update(n_nonzero=int(len(nz)), share_nonzero=float(pi), hurdle_power_max=float(pw_h.max()),
                       hurdle_power_at_observed=float(np.interp(abs(d.mean()), fine, pw_h)),
                       hurdle_mde80_pp=float(fine[np.argmax(pw_h >= 0.8)]) if (pw_h >= 0.8).any() else None,
                       hurdle_power_at_mde80=float(np.interp(mde_sim, fine, pw_h)) if not np.isnan(mde_sim) else None,
                       power_cap=float(1 - binom.cdf(4, n, pi)))
        rows += [dict(contrast=lab + ' (effect only in differing seasons)', delta_pp=dl, power=p) for dl, p in zip(fine, pw_h) if abs(dl * 2 - round(dl * 2)) < 1e-9]
    else:
        kn[lab].update(n_nonzero=int(len(nz)), share_nonzero=float(pi))
P = pd.DataFrame(rows); P.to_csv(B + 'power_curves.csv', index=False, float_format='%.10g')

# scale: what is at stake for a farmer (perfect-foresight CE gain and the M5 gain), for comparison with the MDE
orc = X[(X.model == 'ens+R') & (X.form == 'prob')]
kn['stake'] = dict(oracle_mean_gain_pp=float(100 * (orc.V_oracle.mean() - 1)),
                   ens_R_prob_ce_pp=float(100 * (u_inv(u(orc.V.to_numpy(), 2).mean(), 2) - 1)))
json.dump(kn, open(B + 'key_numbers_power.json', 'w'), indent=1, default=float)
print(json.dumps(kn, indent=1, default=float))
print(P.pivot_table(index='delta_pp', columns='contrast', values='power').round(2).to_string())
