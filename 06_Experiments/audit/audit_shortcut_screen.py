"""A6  single-feature shortcut screen: balanced error of a depth-2 decision tree on each single feature, per task
(random split of the original protocol, repeat 0, n = min(4,096, pool) flows per class). A near-zero error from one
feature flags a possible shortcut (as found for TTL in UNSW-NB15). Output: A6_single_feature.csv"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "04_Code"))
from lcurve import curves, tasks  # noqa: E402

P = json.load(open(os.path.join(ROOT, "04_Code", "configs", "default.json")))
inv = tasks.inventory()
rows = []
for r in inv[(inv.variant == "main") & (inv.family != "Benign")].itertuples():
    feats = json.load(open(os.path.join(tasks.TASK_DIR, f"{r.dataset}__features.json")))["features"]
    a_tr, a_te, b_tr, b_te = curves._pools(P, r.dataset, r.family, "main", "random", 0)
    Xa, _, _ = curves._get(tasks.task_name(r.dataset, r.family))
    Xb, _, _ = curves._get(f"{r.dataset}__Benign")
    n = min(4096, len(a_tr), len(b_tr))
    Xtr = np.vstack([Xb[b_tr[:n]], Xa[a_tr[:n]]]); y = np.r_[np.zeros(n), np.ones(n)]
    res = []
    for j, f in enumerate(feats):
        t = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr[:, [j]], y)
        be = 0.5 * (t.predict(Xb[b_te][:, [j]]).mean() + 1 - t.predict(Xa[a_te][:, [j]]).mean())
        res.append((be, f))
    res.sort()
    rows.append(dict(dataset=r.dataset, family=r.family, best_feature=res[0][1], best_be=res[0][0],
                     second_feature=res[1][1], second_be=res[1][0], n_features_be_lt_1pct=sum(b < 0.01 for b, _ in res)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "A6_single_feature.csv"), index=False)
print(out.round(4).to_string())
