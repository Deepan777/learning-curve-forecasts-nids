"""Intrinsic-dimension estimators and geometric descriptors of each task.

TwoNN (Facco et al. 2017): with mu = r2/r1 the ratio of second- to first-neighbour
distances, -log(1 - F(mu)) = d log(mu); d is the slope through the origin after
discarding the 10 % largest mu. Points with r1 = 0 (exact ties) are excluded and
their share reported.
MLE (Levina & Bickel 2004): d_k(x) = [ (1/(k-1)) sum_{j<k} log(T_k/T_j) ]^{-1},
averaged over points (MacKay-Ghahramani inverse-average correction), for k = 10, 20.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config, tasks
from .curves import phi

GEOM_CSV = os.path.join(config.RAW_RESULTS_DIR, "geometry.csv")


def twonn(Z, discard=0.1):
    d, _ = NearestNeighbors(n_neighbors=3).fit(Z).kneighbors(Z)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    k = int((1 - discard) * len(mu))
    F = np.arange(1, len(mu) + 1) / len(mu)
    x, y = np.log(mu[:k]), -np.log(1 - F[:k])
    return float((x * y).sum() / (x * x).sum()), float(1 - ok.mean())


def mle(Z, k):
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    T = d[:, 1:]
    ok = T[:, 0] > 0
    T = T[ok]
    s = np.log(T[:, -1:] / T[:, :-1]).sum(1) / (k - 1)
    s = s[np.isfinite(s) & (s > 0)]
    return float(1.0 / s.mean())       # inverse of the mean inverse estimate


def continuous_columns(X, max_levels=10):
    return np.array([len(np.unique(X[:5000, j])) > max_levels for j in range(X.shape[1])])


def task_geometry(P, ds, fam, rng):
    Xa, _, _ = tasks.load(tasks.task_name(ds, fam))
    Xb, _, _ = tasks.load(f"{ds}__Benign")
    m = P["id_subsample"]
    ia = rng.choice(len(Xa), min(m, len(Xa)), replace=False)
    ib = rng.choice(len(Xb), min(m, len(Xb)), replace=False)
    out = dict(dataset=ds, family=fam)
    for tag, cols in (("", None), ("_cont", continuous_columns(np.vstack([Xa[ia], Xb[ib]])))):
        Za, Zb = phi(Xa[ia]), phi(Xb[ib])
        if cols is not None:
            Za, Zb = Za[:, cols], Zb[:, cols]
        U = np.vstack([Za, Zb])
        mu, sd = U.mean(0), U.std(0)
        keep = sd > 0
        Za, Zb, U = [(M[:, keep] - mu[keep]) / sd[keep] for M in (Za, Zb, U)]
        for name, Z in (("attack", Za), ("benign", Zb), ("union", U)):
            d2, tie = twonn(Z)
            out[f"twonn_{name}{tag}"] = d2
            out[f"tie_{name}{tag}"] = tie
            if tag == "":
                for k in P["mle_k"]:
                    out[f"mle{k}_{name}"] = mle(Z, k)
        out[f"n_features{tag}"] = int(keep.sum())
        if tag == "":
            # cross-class nearest-neighbour share (overlap descriptor on the subsample)
            lab = np.r_[np.ones(len(Za)), np.zeros(len(Zb))]
            _, idx = NearestNeighbors(n_neighbors=2).fit(U).kneighbors(U)
            nn = lab[idx[:, 1]]
            out["nn_cross_attack"] = float((nn[lab == 1] == 0).mean())
            out["nn_cross_benign"] = float((nn[lab == 0] == 1).mean())
    return out


def run(P, task_list, log=print):
    rows = []
    for i, (ds, fam) in enumerate(task_list):
        rows.append(task_geometry(P, ds, fam, np.random.default_rng([P["seed"], 21, i])))
        log(f"  geometry {ds}/{fam}: TwoNN attack {rows[-1]['twonn_attack']:.2f} benign {rows[-1]['twonn_benign']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(GEOM_CSV, index=False)
    return df
