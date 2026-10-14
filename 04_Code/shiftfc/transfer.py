"""Paired source-validation / target learning curves and label-free shift scores for one scenario.

For every repeat and size n the learner is trained ONCE on n source flows per class and evaluated on
(i) the held-out source validation set (pilot curve) and (ii) the target test set (realised target curve).
Operating threshold for the detection metrics is fixed on the SOURCE validation benign scores (1 % FPR), never on
target or test data; the realised target FPR/TPR at that threshold are recorded (this replaces the test-set
threshold of the original secondary metric).
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lcurve.curves import make_learner, phi

from . import scenarios, signals

CURVE_DIR = os.path.join(scenarios.SHIFT_DIR, "curves")
SIGNAL_DIR = os.path.join(scenarios.SHIFT_DIR, "signals")


def run_rep(sc, rep, learners):
    S = scenarios.S
    Xa, Xb = scenarios.arrays(sc)
    a_src, a_val, a_tgt, b_src, b_val, b_tgt = scenarios.pools(sc, rep)
    Pva_a, Pva_b, Pt_a, Pt_b = phi(Xa[a_val]), phi(Xb[b_val]), phi(Xa[a_tgt]), phi(Xb[b_tgt])
    nmax = min(len(a_src), len(b_src))
    rec, sig = [], None
    for n in S["sizes"]:
        if n > nmax:
            break
        Ztr = np.vstack([phi(Xb[b_src[:n]]), phi(Xa[a_src[:n]])])
        ytr = np.r_[np.zeros(n), np.ones(n)]
        mu, sd = Ztr.mean(0), Ztr.std(0)
        sd[sd == 0] = 1.0
        Ztr = (Ztr - mu) / sd
        Zva_a, Zva_b, Zt_a, Zt_b = (Pva_a - mu) / sd, (Pva_b - mu) / sd, (Pt_a - mu) / sd, (Pt_b - mu) / sd
        Zall = np.vstack([Zva_b, Zva_a, Zt_b, Zt_a])
        c = np.cumsum([0, len(Zva_b), len(Zva_a), len(Zt_b), len(Zt_a)])
        for L in learners:
            k_nn = {"kNN1": 1, "kNN5": 5}.get(L, 0)
            if k_nn > 2 * n:
                continue
            clf = make_learner(L, n, seed=rep)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Ztr, ytr)
                p = clf.predict_proba(Zall)[:, 1]
            sec = time.perf_counter() - t0
            pvb, pva, ptb, pta = p[c[0]:c[1]], p[c[1]:c[2]], p[c[2]:c[3]], p[c[3]:c[4]]
            thr = np.quantile(pvb, 0.99, method="higher")      # 1 % FPR on SOURCE validation benign
            r = dict(sid=sc["sid"], kind=sc["kind"], dataset=sc["dataset"], family=sc["family"],
                     heldout=sc["heldout"], rep=rep, learner=L, n=n,
                     n_val_a=len(a_val), n_val_b=len(b_val), n_tgt_a=len(a_tgt), n_tgt_b=len(b_tgt),
                     be_val=0.5 * (float((pvb > 0.5).mean()) + float((pva <= 0.5).mean())),
                     fpr_tgt=float((ptb > 0.5).mean()), fnr_tgt=float((pta <= 0.5).mean()),
                     tpr_tgt_cal=float((pta > thr).mean()), fpr_tgt_cal=float((ptb > thr).mean()),
                     tpr_val_cal=float((pva > thr).mean()), sec=sec)
            r["be_tgt"] = 0.5 * (r["fpr_tgt"] + r["fnr_tgt"])
            try:
                r["auc_tgt"] = float(roc_auc_score(np.r_[np.zeros(len(ptb)), np.ones(len(pta))], np.r_[ptb, pta]))
            except ValueError:
                r["auc_tgt"] = np.nan
            rec.append(r)
        if n == S["signal_n"]:
            m_a = min(S["signal_attack"], len(a_val), len(a_tgt))
            m_b = min(S["signal_benign"], len(b_val), len(b_tgt))
            prev = S["prevalence_targets"] if sc["kind"] == "id" else ()
            sig = dict(sid=sc["sid"], kind=sc["kind"], dataset=sc["dataset"], family=sc["family"],
                       heldout=sc["heldout"], rep=rep, m_a=m_a, m_b=m_b)
            sig.update(signals.compute(Ztr, ytr, Zva_a, Zva_b, Zt_a, Zt_b, m_a, m_b,
                                       seed=S["seed"] * 1000 + rep * 10 + 7, prevalences=prev))
    return rec, sig


def run_cell(sc, learners=None):
    os.makedirs(CURVE_DIR, exist_ok=True); os.makedirs(SIGNAL_DIR, exist_ok=True)
    cp = os.path.join(CURVE_DIR, sc["sid"] + ".csv")
    sp = os.path.join(SIGNAL_DIR, sc["sid"] + ".csv")
    if os.path.exists(cp) and os.path.exists(sp):
        return "exists"
    t0 = time.time()
    learners = learners or scenarios.S["learners"]
    recs, sigs = [], []
    for rep in range(scenarios.S["repeats"]):
        r, s = run_rep(sc, rep, learners)
        recs += r
        if s is not None:
            sigs.append(s)
    for path, df in ((sp, pd.DataFrame(sigs)), (cp, pd.DataFrame(recs))):    # curves written last = cell done
        df.to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
    return f"{time.time() - t0:.0f}s"
