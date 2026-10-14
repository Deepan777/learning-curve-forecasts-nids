"""Exploratory controls for the residual errors above 1 % in four CIC-IoT2023 families (follow-up F6).

(a) sub-type decomposition  per-sub-type miss rate of RF and HGB at the largest size of the original protocol
(b) more data               Recon and Spoofing with up to 32,768 flows per class from the full low-floor cache
                            (05_Data/cache/ciciot2023_lowfloor_full.parquet; exact duplicates removed within class,
                            vectors carrying both labels removed, as in the original task construction)
(c) capacity                random forest and gradient boosting tuned by 3-fold cross-validation on the TRAINING data
                            only (balanced error), at the largest size; test data never used for selection
Outputs: 06_Experiments/shift/floors/*.csv
"""
from __future__ import annotations

import itertools
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from lcurve import curves, tasks
from lcurve.curves import phi

from . import scenarios, variants

OUT = os.path.join(scenarios.SHIFT_DIR, "floors")
FAMS = ("Recon", "Spoofing", "Web", "BruteForce")
P = scenarios.P_ORIG


def _std(Ztr, *Zs):
    mu, sd = Ztr.mean(0), Ztr.std(0); sd[sd == 0] = 1
    return [(Ztr - mu) / sd] + [(Z - mu) / sd for Z in Zs]


def _be(clf, Zb, Za):
    return 0.5 * (float(clf.predict(Zb).mean()) + float(1 - clf.predict(Za).mean()))


def _rf(seed, mf="sqrt", leaf=1, trees=200):
    return RandomForestClassifier(trees, max_features=mf, min_samples_leaf=leaf, n_jobs=2, random_state=seed)


def _hgb(seed, n, it=100, leaves=31, msl=None):
    return HistGradientBoostingClassifier(max_iter=it, max_leaf_nodes=leaves,
                                          min_samples_leaf=msl if msl else min(20, max(1, (2 * n) // 20)),
                                          early_stopping=False, random_state=seed)


def subtype_decomposition(reps=5):
    rows = []
    for fam in FAMS:
        Xa, oa, _ = curves._get(tasks.task_name("ciciot", fam))
        Xb, _, _ = curves._get("ciciot__Benign")
        v = variants.fine_labels("ciciot", oa).astype(str)
        for rep in range(reps):
            a_tr, a_te, b_tr, b_te = curves._pools(P, "ciciot", fam, "main", "random", rep)
            n = 2 ** int(np.floor(np.log2(min(len(a_tr), len(b_tr), 16384))))
            Ztr, Zb, Za = _std(np.vstack([phi(Xb[b_tr[:n]]), phi(Xa[a_tr[:n]])]), phi(Xb[b_te]), phi(Xa[a_te]))
            y = np.r_[np.zeros(n), np.ones(n)]
            for name, clf in (("RF", _rf(rep)), ("HGB", _hgb(rep, n))):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(Ztr, y)
                miss = clf.predict(Za) == 0
                fpr = float(clf.predict(Zb).mean())
                for sub in np.unique(v[a_te]):
                    m = v[a_te] == sub
                    rows.append(dict(family=fam, subtype=sub, rep=rep, learner=name, n=n, n_test=int(m.sum()),
                                     miss_rate=float(miss[m].mean()), misses=int(miss[m].sum()),
                                     total_misses=int(miss.sum()), fpr=fpr))
    return pd.DataFrame(rows)


def _lowfloor_task(fam):
    d = pd.read_parquet(variants.LOWFLOOR_CACHE)
    feat = [c for c in d.columns if c not in ("fine", "family", "order")]
    X = d[feat].to_numpy(np.float32)
    H = tasks.row_hash(X)
    isb, isa = (d.family == "Benign").to_numpy(), (d.family == fam).to_numpy()
    hb = set(H[isb].tolist()); ha = set(H[isa].tolist())
    both = hb & ha
    def uniq(mask):
        idx = np.flatnonzero(mask)
        _, first = np.unique(H[idx], return_index=True)
        idx = np.sort(idx[first])
        return idx[~np.isin(H[idx], np.fromiter(both, dtype=H.dtype))] if both else idx
    return X, uniq(isb), uniq(isa), len(both)


def more_data(reps=5, sizes=(1024, 4096, 16384, 32768)):
    rows = []
    for fam in ("Recon", "Spoofing"):
        X, bi, ai, nconf = _lowfloor_task(fam)
        for rep in range(reps):
            rng = np.random.default_rng([scenarios.S["seed"], rep, 31, len(fam)])
            A, B = rng.permutation(ai), rng.permutation(bi)
            a_te, a_tr, b_te, b_tr = A[:2000], A[2000:], B[:5000], B[5000:]
            for n in sizes:
                if n > min(len(a_tr), len(b_tr)):
                    continue
                Ztr, Zb, Za = _std(np.vstack([phi(X[b_tr[:n]]), phi(X[a_tr[:n]])]), phi(X[b_te]), phi(X[a_te]))
                y = np.r_[np.zeros(n), np.ones(n)]
                for name, clf in (("RF", _rf(rep)), ("HGB", _hgb(rep, n))):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(Ztr, y)
                    rows.append(dict(family=fam, rep=rep, learner=name, n=n, be=_be(clf, Zb, Za),
                                     unique_attack=len(ai), unique_benign=len(bi), conflict_vectors=nconf))
    return pd.DataFrame(rows)


GRID = {"RF": [dict(mf=mf, leaf=lf) for mf, lf in itertools.product(("sqrt", 0.5), (1, 5))],
        "HGB": [dict(it=it, leaves=lv, msl=ms) for it, lv, ms in itertools.product((200, 1000), (31, 255), (5, 20))]}


def capacity(reps=3):
    rows = []
    for fam in FAMS:
        Xa, _, _ = curves._get(tasks.task_name("ciciot", fam))
        Xb, _, _ = curves._get("ciciot__Benign")
        for rep in range(reps):
            a_tr, a_te, b_tr, b_te = curves._pools(P, "ciciot", fam, "main", "random", rep)
            n = 2 ** int(np.floor(np.log2(min(len(a_tr), len(b_tr), 16384))))
            Ztr, Zb, Za = _std(np.vstack([phi(Xb[b_tr[:n]]), phi(Xa[a_tr[:n]])]), phi(Xb[b_te]), phi(Xa[a_te]))
            y = np.r_[np.zeros(n), np.ones(n)]
            for name in ("RF", "HGB"):
                cv = []
                for cfg in GRID[name]:
                    errs = []
                    for tr, va in StratifiedKFold(3, shuffle=True, random_state=rep).split(Ztr, y):
                        clf = _rf(rep, trees=300, **cfg) if name == "RF" else _hgb(rep, n, **cfg)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            clf.fit(Ztr[tr], y[tr])
                        yv = y[va]
                        errs.append(_be(clf, Ztr[va][yv == 0], Ztr[va][yv == 1]))
                    cv.append(float(np.mean(errs)))
                best = GRID[name][int(np.argmin(cv))]
                clf = _rf(rep, trees=300, **best) if name == "RF" else _hgb(rep, n, **best)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(Ztr, y)
                base = _rf(rep) if name == "RF" else _hgb(rep, n)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    base.fit(Ztr, y)
                rows.append(dict(family=fam, rep=rep, learner=name, n=n, be_default=_be(base, Zb, Za),
                                 be_tuned=_be(clf, Zb, Za), cv_be_best=min(cv), best=str(best)))
    return pd.DataFrame(rows)


def run():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("subtypes", subtype_decomposition), ("more_data", more_data), ("capacity", capacity)):
        path = os.path.join(OUT, name + ".csv")
        if os.path.exists(path):
            print(name, "exists"); continue
        df = fn()
        df.to_csv(path, index=False)
        print(name, len(df), flush=True)
