"""Learning-curve experiment.

One unit of work = (task, order, variant, repeat). For each unit, a test set is
drawn once, the training pool is permuted once, and nested training sets of
n = 4, 8, ..., 16384 flows per class are taken as prefixes of that permutation.
Every learner is fitted on every training set.

order = random : test and training pools drawn at random from all flows
order = chrono : training pool from earlier flows, test from later flows
                 (UNSW-NB15: session 1 -> session 2; CICIDS2017: first 60 % -> last
                 40 % of the family's day, for the family and that day's benign flows)
order = rcontrol : the chrono union re-split at random into pools of the chrono sizes
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

from . import config, tasks

CURVE_DIR = os.path.join(config.RAW_RESULTS_DIR, "curves")
os.makedirs(CURVE_DIR, exist_ok=True)
_CACHE = {}


def phi(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_learner(name, n, seed, alt=False):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    if name == "LogReg":
        return LogisticRegression(C=1.0, max_iter=1000)
    if name == "kNN1":
        return KNeighborsClassifier(1)
    if name == "kNN5":
        return KNeighborsClassifier(10 if alt else 5)
    if name == "RF":
        return RandomForestClassifier(500 if alt else 200, max_features="sqrt", n_jobs=1, random_state=seed)
    if name == "HGB":
        return HistGradientBoostingClassifier(min_samples_leaf=min(20, max(1, (2 * n) // 20)),
                                              early_stopping=False, random_state=seed)
    if name == "MLP":
        return MLPClassifier((64, 32), max_iter=200, random_state=seed)
    raise ValueError(name)


def _get(name):
    if name not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.clear()
        _CACHE[name] = tasks.load(name)
    return _CACHE[name]


def _pools(P, ds, fam, variant, order, rep):
    """Return (attack train pool, attack test, benign train pool, benign test) as row
    indices into the task's attack and benign arrays."""
    Xa, oa, pa = _get(tasks.task_name(ds, fam, variant))
    Xb, ob, pb = _get(f"{ds}__Benign" + ("" if variant == "main" else f"__{variant}"))
    rng = np.random.default_rng([P["seed"], rep, abs(hash_str(f"{ds}|{fam}|{variant}|{order}")) % (2 ** 31)])
    if order == "random":
        A, B = np.arange(len(Xa)), np.arange(len(Xb))
        nta = min(P["n_test_attack_max"], int(P["test_frac_max"] * len(A)))
        ntb = min(P["n_test_benign"], int(P["test_frac_max"] * len(B)))
        A, B = rng.permutation(A), rng.permutation(B)
        return A[nta:], A[:nta], B[ntb:], B[:ntb]
    # chronological split (and its random control)
    if ds == "unsw":
        a_tr, a_te = np.flatnonzero(pa == "S1"), np.flatnonzero(pa == "S2")
        b_tr, b_te = np.flatnonzero(pb == "S1"), np.flatnonzero(pb == "S2")
    elif ds == "cicids":
        day = pa[0]
        a = np.argsort(oa, kind="stable")
        cut = int(P["chrono_train_frac_cicids"] * len(a))
        a_tr, a_te = a[:cut], a[cut:]
        bd = np.flatnonzero(pb == day)
        bd = bd[np.argsort(ob[bd], kind="stable")]
        cutb = int(P["chrono_train_frac_cicids"] * len(bd))
        b_tr, b_te = bd[:cutb], bd[cutb:]
    else:
        raise ValueError("no time order for " + ds)
    if order == "rcontrol":           # same union, same pool sizes, random assignment
        ua, ub = rng.permutation(np.r_[a_tr, a_te]), rng.permutation(np.r_[b_tr, b_te])
        a_tr, a_te = ua[:len(a_tr)], ua[len(a_tr):]
        b_tr, b_te = ub[:len(b_tr)], ub[len(b_tr):]
    nta = min(P["n_test_attack_max"], len(a_te))
    ntb = min(P["n_test_benign"], len(b_te))
    return rng.permutation(a_tr), rng.choice(a_te, nta, replace=False), rng.permutation(b_tr), \
        rng.choice(b_te, ntb, replace=False)


def hash_str(s):
    import zlib
    return zlib.crc32(s.encode())


def tpr_at_fpr(sb, sa, fpr=0.01):
    thr = np.quantile(sb, 1 - fpr, method="higher")
    return float((sa > thr).mean()), float((sb > thr).mean())


def run_unit(P, ds, fam, variant, order, rep, learners=None, alt=False):
    """Run every size and learner for one unit; returns a list of records."""
    learners = learners or P["learners"]
    Xa, _, _ = _get(tasks.task_name(ds, fam, variant))
    Xb, _, _ = _get(f"{ds}__Benign" + ("" if variant == "main" else f"__{variant}"))
    a_tr, a_te, b_tr, b_te = _pools(P, ds, fam, variant, order, rep)
    Za_te, Zb_te = phi(Xa[a_te]), phi(Xb[b_te])
    out = []
    nmax = min(len(a_tr), len(b_tr))
    for n in P["sizes"]:
        if n > nmax:
            break
        Ztr = np.vstack([phi(Xb[b_tr[:n]]), phi(Xa[a_tr[:n]])])
        ytr = np.r_[np.zeros(n), np.ones(n)]
        mu, sd = Ztr.mean(0), Ztr.std(0)
        sd[sd == 0] = 1.0
        Ztr = (Ztr - mu) / sd
        Zte = np.vstack([(Zb_te - mu) / sd, (Za_te - mu) / sd])
        nb = len(b_te)
        for L in learners:
            k_nn = {"kNN1": 1, "kNN5": 10 if alt else 5}.get(L, 0)
            if k_nn > 2 * n:          # fewer training points than neighbours: size skipped for this learner
                continue
            clf = make_learner(L, n, seed=rep, alt=alt)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(Ztr, ytr)
                p = clf.predict_proba(Zte)[:, 1]
            sec = time.perf_counter() - t0
            pred = p > 0.5
            fpr = float(pred[:nb].mean()); fnr = float(1 - pred[nb:].mean())
            tpr1, fpr1 = tpr_at_fpr(p[:nb], p[nb:])
            out.append(dict(dataset=ds, family=fam, variant=variant, order=order, rep=rep, learner=L,
                            alt=int(alt), n=n, n_test_a=len(a_te), n_test_b=nb, fpr=fpr, fnr=fnr,
                            be=0.5 * (fpr + fnr), tpr_at_1fpr=tpr1, fpr_realised=fpr1, sec=sec))
    return out


def unit_file(ds, fam, variant, order, alt=False):
    tag = f"{ds}__{fam}__{variant}__{order}" + ("__alt" if alt else "")
    return os.path.join(CURVE_DIR, tag + ".csv")


def run_cell(P, ds, fam, variant, order, alt=False, learners=None):
    """All repeats of one (task, variant, order); written atomically as one CSV."""
    path = unit_file(ds, fam, variant, order, alt)
    if os.path.exists(path):
        return path, "exists"
    t0 = time.time()
    recs = []
    for rep in range(P["repeats"]):
        recs += run_unit(P, ds, fam, variant, order, rep, learners=learners, alt=alt)
    tmp = path + ".tmp"
    pd.DataFrame(recs).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path, f"{time.time() - t0:.0f}s"
