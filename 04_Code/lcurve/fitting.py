"""Learning-curve models, sample-complexity estimates and forecasts.

Models (fitted on log(e + e0), e0 = 1/(2 n_test_attack), the resolution floor):
  M1  e = a n^-b              (pure power law)
  M2  e = c + a n^-b          (saturating power law; c = asymptote / floor)
  M3  e = c + a exp(-b n)     (exponential)
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .curves import CURVE_DIR

KEYS = ["dataset", "family", "variant", "order", "alt", "learner"]


def load_curves():
    fs = [f for f in glob.glob(os.path.join(CURVE_DIR, "*.csv"))]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def _m1(p, n):
    return p[0] * n ** (-p[1])


def _m2(p, n):
    return p[2] + p[0] * n ** (-p[1])


def _m3(p, n):
    return p[2] + p[0] * np.exp(-p[1] * n)


MODELS = {
    "M1": (_m1, [0.5, 0.5], ([1e-8, 0.0], [50.0, 3.0])),
    "M2": (_m2, [0.5, 0.5, 0.001], ([1e-8, 0.0, 0.0], [50.0, 3.0, 0.5])),
    "M3": (_m3, [0.3, 0.01, 0.001], ([1e-8, 1e-7, 0.0], [1.0, 1.0, 0.5])),
}


def fit(n, e, e0, model, starts=None):
    f, p0, bounds = MODELS[model]
    n = np.asarray(n, float); e = np.asarray(e, float)
    y = np.log(e + e0)
    best = None
    starts = starts or [p0] + ([[1.0, 1.0, 0.0], [0.2, 0.3, 0.01], [2.0, 0.8, 0.0]] if model == "M2" else
                     [[1.0, 1.0], [0.2, 0.3]] if model == "M1" else [[0.5, 0.001, 0.0], [0.2, 0.1, 0.01]])
    for s in starts:
        s = np.clip(s, bounds[0], bounds[1])
        try:
            r = least_squares(lambda p: np.log(np.maximum(f(p, n), 0) + e0) - y, s, bounds=bounds, max_nfev=5000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    rss = float(2 * best.cost)
    m, k = len(n), len(p0)
    aicc = m * np.log(max(rss, 1e-300) / m) + 2 * k + (2 * k * (k + 1) / (m - k - 1) if m - k - 1 > 0 else np.inf)
    return best.x, rss, aicc


def nstar(n, e, eps):
    """Smallest n with e <= eps, log-linear interpolation; np.inf if never reached
    (right-censored); n[0] if already reached at the smallest size (left-censored)."""
    n = np.asarray(n, float); e = np.asarray(e, float)
    hit = np.flatnonzero(e <= eps)
    if len(hit) == 0:
        return np.inf
    i = hit[0]
    if i == 0:
        return n[0]
    x0, x1 = np.log(n[i - 1]), np.log(n[i])
    y0, y1 = e[i - 1], e[i]
    t = (y0 - eps) / (y0 - y1) if y0 != y1 else 1.0
    return float(np.exp(x0 + t * (x1 - x0)))


def invert(model, p, eps):
    """n at which the fitted curve reaches eps (np.inf if never)."""
    if model == "M1":
        a, b = p
        return np.inf if b <= 0 else float((a / eps) ** (1 / b))
    a, b, c = p
    if eps <= c or b <= 0:
        return np.inf
    if model == "M2":
        return float((a / (eps - c)) ** (1 / b))
    return float(np.log(a / (eps - c)) / b) if a > eps - c else 1.0


def summarise(df, eps=(0.01, 0.05), n_boot=0, rng=None, forecast_max=128):
    """One row per curve (KEYS): mean curve, fits, n*, forecast, bootstrap CIs."""
    rows = []
    for key, g in df.groupby(KEYS):
        piv = g.pivot_table(index="n", columns="rep", values="be")
        n = piv.index.values.astype(float)
        e = piv.mean(1).values
        e0 = 1.0 / (2 * g.n_test_a.iloc[0])
        rec = dict(zip(KEYS, key))
        rec.update(n_max=float(n.max()), n_points=len(n), e_min_n=float(e[0]), e_max_n=float(e[-1]), e0=e0,
                   reps=piv.shape[1])
        for m in MODELS:
            if len(n) >= len(MODELS[m][1]) + 2:
                p, rss, aicc = fit(n, e, e0, m)
                rec[f"{m}_aicc"] = aicc
                for j, nm in enumerate(["a", "b", "c"][:len(p)]):
                    rec[f"{m}_{nm}"] = float(p[j])
            else:
                rec[f"{m}_aicc"] = np.nan
        for ep in eps:
            rec[f"nstar_{ep}"] = nstar(n, e, ep)
        # detection-based outcome: miss rate at 1 % FPR <= 5 %
        miss = 1 - g.groupby("n").tpr_at_1fpr.mean().reindex(piv.index).values
        rec["nstar_det95"] = nstar(n, miss, 0.05)
        rec["miss_max_n"] = float(miss[-1])
        # forecast from the first points (n <= forecast_max)
        sel = n <= forecast_max
        if sel.sum() >= 5:
            for m in ("M1", "M2"):
                p, _, _ = fit(n[sel], e[sel], e0, m)
                rec[f"fc_{m}_{0.01}"] = invert(m, p, 0.01)
        if n_boot:
            bs_b, bs_c, bs_n = [], [], []
            p_hat = [rec.get("M2_a", 0.5), rec.get("M2_b", 0.5), rec.get("M2_c", 0.001)]
            reps = piv.columns.values
            for _ in range(n_boot):
                pick = rng.choice(reps, len(reps), replace=True)
                eb = piv[pick].mean(1).values
                p, _, _ = fit(n, eb, e0, "M2", starts=[p_hat]) if len(n) >= 5 else (np.full(3, np.nan), 0, 0)
                bs_b.append(p[1]); bs_c.append(p[2]); bs_n.append(nstar(n, eb, 0.01))
            rec["M2_b_lo"], rec["M2_b_hi"] = np.nanpercentile(bs_b, [2.5, 97.5])
            rec["M2_c_lo"], rec["M2_c_hi"] = np.nanpercentile(bs_c, [2.5, 97.5])
            bn = np.array(bs_n)
            rec["nstar_0.01_lo"], rec["nstar_0.01_hi"] = np.percentile(bn, [2.5, 97.5])
        rows.append(rec)
    return pd.DataFrame(rows)
