"""Phase-1 computational audit of the existing results (no change to the original outputs).

A1  forecast outcome categories (measurable / both censored / observed censored only / forecast censored only)
A2  near-duplicate leakage: share of test flows whose nearest training flow (standardised signed-log features)
    lies within a tiny distance, per task (random split, n = 4,096 per class or the largest available)
A3  Kaplan-Meier estimate of the distribution of n*(1%) per learner, treating not-reached as right-censored
A4  dataset-clustered bootstrap for the H2 proportion (resampling datasets, then tasks within them)
A5  port shortcut in CICIDS2017: random-forest balanced error under the past->future split with and without
    'Destination Port' (10 repeats, largest common size)
Outputs: 06_Experiments/audit/*.csv and audit_summary.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "04_Code"))
from lcurve import curves, tasks  # noqa: E402

P = json.load(open(os.path.join(ROOT, "04_Code", "configs", "default.json")))
STAT = os.path.join(ROOT, "09_Statistical_Analysis")
out = {}

# ---------------------------------------------------------------- A1
f = pd.read_csv(os.path.join(STAT, "H6_forecast.csv"))
f = f[f.model == "M2"].copy()
oc = ~np.isfinite(f.observed)
fc = (~np.isfinite(f.forecast)) | (f.forecast > f.n_max)
cat = np.select([~oc & ~fc, oc & fc, oc & ~fc, ~oc & fc],
                ["measurable", "both_beyond_data", "observed_censored_forecast_finite",
                 "observed_finite_forecast_beyond"], "other")
f["category"] = cat
f["within2"] = f["within2"].astype(bool)
tab = f.groupby("category").agg(n=("within2", "size"), correct=("within2", "sum")).reset_index()
tab.to_csv(os.path.join(HERE, "A1_forecast_categories.csv"), index=False)
meas = f[f.category == "measurable"]
out["A1"] = {"categories": tab.to_dict("records"),
             "measurable_share_within2": float(meas.within2.mean()), "measurable_n": int(len(meas)),
             "measurable_median_abs_log2_error": float(np.median(np.abs(np.log2(meas.forecast / meas.observed))))}

# ---------------------------------------------------------------- A2
from sklearn.neighbors import NearestNeighbors  # noqa: E402

inv = tasks.inventory()
main = inv[(inv.variant == "main") & (inv.family != "Benign")]
rows = []
for r in main.itertuples():
    a_tr, a_te, b_tr, b_te = curves._pools(P, r.dataset, r.family, "main", "random", 0)
    Xa, _, _ = curves._get(tasks.task_name(r.dataset, r.family))
    Xb, _, _ = curves._get(f"{r.dataset}__Benign")
    n = min(4096, len(a_tr), len(b_tr))
    Ztr = np.vstack([curves.phi(Xb[b_tr[:n]]), curves.phi(Xa[a_tr[:n]])])
    mu, sd = Ztr.mean(0), Ztr.std(0); sd[sd == 0] = 1
    Ztr = (Ztr - mu) / sd
    ytr = np.r_[np.zeros(n), np.ones(n)]
    nn = NearestNeighbors(n_neighbors=1).fit(Ztr)
    for lab, idx, X in (("attack", a_te, Xa), ("benign", b_te, Xb)):
        Z = (curves.phi(X[idx]) - mu) / sd
        d, j = nn.kneighbors(Z)
        d = d[:, 0]; same = ytr[j[:, 0]] == (1 if lab == "attack" else 0)
        scale = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        rows.append(dict(dataset=r.dataset, family=r.family, cls=lab, n_train=n,
                         share_dist_lt_1e3=float((d < 1e-3).mean()),
                         share_dist_lt_1pct_median=float((d < 0.01 * scale).mean()),
                         share_nn_same_class=float(same.mean()), median_nn_dist=float(np.median(d))))
a2 = pd.DataFrame(rows)
a2.to_csv(os.path.join(HERE, "A2_near_duplicates.csv"), index=False)
out["A2"] = {"max_share_dist_lt_1e3": float(a2.share_dist_lt_1e3.max()),
             "tasks_with_share_gt_5pct": a2[a2.share_dist_lt_1e3 > 0.05][["dataset", "family", "cls",
                                                                           "share_dist_lt_1e3"]].to_dict("records")}

# ---------------------------------------------------------------- A3
s = pd.read_csv(os.path.join(STAT, "curve_summary.csv"))
s = s[(s.variant == "main") & (s.order == "random") & (s.alt == 0)]


def km_quantile(times, events, q=0.5):
    """Kaplan-Meier estimate of the q-quantile of n* (events = reached 1 %)."""
    t = np.asarray(times, float); e = np.asarray(events, bool)
    o = np.argsort(t); t, e = t[o], e[o]
    surv, n_at = 1.0, len(t)
    for ti in np.unique(t):
        d = int(((t == ti) & e).sum()); c = int((t == ti).sum())
        if d:
            surv *= 1 - d / n_at
            if surv <= 1 - q:
                return float(ti)
        n_at -= c
    return np.inf


km = []
for L, g in s.groupby("learner"):
    t = np.where(np.isfinite(g["nstar_0.01"]), g["nstar_0.01"], g.n_max)
    e = np.isfinite(g["nstar_0.01"])
    km.append(dict(learner=L, km_median=km_quantile(t, e), km_q25=km_quantile(t, e, 0.25),
                   km_q75=km_quantile(t, e, 0.75), reached=int(e.sum()), n=len(g)))
a3 = pd.DataFrame(km)
a3.to_csv(os.path.join(HERE, "A3_km_nstar.csv"), index=False)
out["A3"] = a3.to_dict("records")

# ---------------------------------------------------------------- A4
t2 = pd.read_csv(os.path.join(STAT, "H2_sample_complexity.csv"))
rng = np.random.default_rng(2026)
dss = t2.dataset.unique()
boot = []
for _ in range(5000):
    pick = rng.choice(dss, len(dss), replace=True)
    parts = []
    for d in pick:
        g = t2[t2.dataset == d]
        parts.append(g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 30))))
    b = pd.concat(parts)
    boot.append(float((b.nstar_tree <= 1024).mean()))
out["A4"] = {"share_le_1024": float((t2.nstar_tree <= 1024).mean()),
             "cluster_bootstrap_95ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}

# ---------------------------------------------------------------- A5
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

feats = json.load(open(os.path.join(tasks.TASK_DIR, "cicids__features.json")))["features"]
port = feats.index("Destination Port")
rows = []
for fam in ("BruteForce", "DoS", "WebAttack", "Botnet", "PortScan", "DDoS"):
    for order in ("chrono", "rcontrol"):
        Xa, _, _ = curves._get(tasks.task_name("cicids", fam))
        Xb, _, _ = curves._get("cicids__Benign")
        for rep in range(P["repeats"]):
            a_tr, a_te, b_tr, b_te = curves._pools(P, "cicids", fam, "main", order, rep)
            n = min(len(a_tr), len(b_tr), 16384)
            n = 2 ** int(np.floor(np.log2(n)))
            for drop in (False, True):
                keep = [j for j in range(Xa.shape[1]) if not (drop and j == port)]
                Ztr = np.vstack([curves.phi(Xb[b_tr[:n]][:, keep]), curves.phi(Xa[a_tr[:n]][:, keep])])
                mu, sd = Ztr.mean(0), Ztr.std(0); sd[sd == 0] = 1
                ytr = np.r_[np.zeros(n), np.ones(n)]
                clf = RandomForestClassifier(200, n_jobs=4, random_state=rep).fit((Ztr - mu) / sd, ytr)
                pb = clf.predict((curves.phi(Xb[b_te][:, keep]) - mu) / sd)
                pa = clf.predict((curves.phi(Xa[a_te][:, keep]) - mu) / sd)
                rows.append(dict(family=fam, order=order, rep=rep, n=n, drop_port=drop,
                                 be=0.5 * (pb.mean() + (1 - pa.mean()))))
a5 = pd.DataFrame(rows)
a5.to_csv(os.path.join(HERE, "A5_port_shortcut_raw.csv"), index=False)
a5s = a5.groupby(["family", "order", "drop_port"]).be.mean().unstack("drop_port")
a5s.columns = ["with_port", "without_port"]
a5s.reset_index().to_csv(os.path.join(HERE, "A5_port_shortcut.csv"), index=False)
out["A5"] = a5s.reset_index().round(5).to_dict("records")

json.dump(out, open(os.path.join(HERE, "audit_summary.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
