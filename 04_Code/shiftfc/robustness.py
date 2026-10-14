"""Robustness of the follow-up outcomes to the analysis choices (exploratory; uses the saved curves only).

Varies one choice at a time around the primary setting (target 1 %, pilot n <= 128, tolerance factor 2):
  target error eps in {1 %, 5 %}; pilot size n <= {64, 128, 256}; tolerance factor in {1.5, 2, 3}.
Output: 13_Results/Shift_Robustness.csv (kind, setting, measurable forecasts, share within tolerance, verifiable
forecasts, unsafe share, false-assurance count). Primary analysis set (CIC-IoT2023 without artefact features).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from lcurve import config
from lcurve.fitting import fit, invert, nstar

from . import analysis

OUT = os.path.join(config.PROJECT_ROOT, "13_Results", "Shift_Robustness.csv")


def _forecasts(c, eps, pilot):
    rows = []
    for (sid, L), g in c.groupby(["sid", "learner"]):
        m = g.groupby("n")[["be_val", "be_tgt"]].mean()
        n = m.index.values.astype(float)
        e0 = 1.0 / (2 * g.n_val_a.iloc[0])
        sel = n <= pilot
        fc = np.nan
        if sel.sum() >= 5:
            p, _, _ = fit(n[sel], m.be_val.values[sel], e0, "M2")
            fc = invert("M2", p, eps)
        rows.append(dict(sid=sid, learner=L, kind=g.kind.iloc[0], dataset=g.dataset.iloc[0], n_max=float(n.max()),
                         fc=fc, nstar_tgt=nstar(n, m.be_tgt.values, eps)))
    return pd.DataFrame(rows)


def _summ(f, tol):
    fc, ob, nm = f.fc.astype(float), f.nstar_tgt.astype(float), f.n_max
    fc_in, ob_in = np.isfinite(fc) & (fc <= nm), np.isfinite(ob)
    meas = fc_in & ob_in
    within = meas & (np.abs(np.log2(fc / ob)) <= np.log2(tol))
    verif = fc_in & (tol * fc <= nm)
    unsafe = verif & (~ob_in | (ob > tol * fc))
    return dict(measurable=int(meas.sum()), within_share=float(within.sum() / meas.sum()) if meas.sum() else np.nan,
                verifiable=int(verif.sum()), unsafe_share=float(unsafe.sum() / verif.sum()) if verif.sum() else np.nan,
                false_assurance=int((fc_in & ~ob_in).sum()))


def run():
    c, _ = analysis.load()
    art = list(analysis.S.get("artefact_features", {}))
    noart = c.sid.str.endswith("__noart")
    c = c[(c.dataset.isin(art) & noart) | (~c.dataset.isin(art) & ~noart)]
    settings = [("eps0.01_pilot128_tol2", 0.01, 128, 2.0), ("eps0.05_pilot128_tol2", 0.05, 128, 2.0),
                ("eps0.01_pilot64_tol2", 0.01, 64, 2.0), ("eps0.01_pilot256_tol2", 0.01, 256, 2.0),
                ("eps0.01_pilot128_tol1.5", 0.01, 128, 1.5), ("eps0.01_pilot128_tol3", 0.01, 128, 3.0)]
    cache, rows = {}, []
    for name, eps, pilot, tol in settings:
        key = (eps, pilot)
        if key not in cache:
            cache[key] = _forecasts(c, eps, pilot)
        f = cache[key]
        for k, g in f.groupby("kind"):
            rows.append(dict(setting=name, kind=k, forecasts=len(g), **_summ(g, tol)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string())
