"""Data-integrity audit for the sample-complexity study.

Checks per dataset (as loaded by lcurve.data, i.e. after the documented cleaning):
rows, features, non-finite values, constant features, exact duplicate rows overall
and within class, feature vectors carrying both labels, family sizes before/after
de-duplication, identifiers excluded from features, and the UNSW-NB15 TTL artefact
(balanced error of a depth-2 tree on the TTL features alone, with vs without).
Outputs: data_audit_results.json, ../15_Supplementary/supp_table_integrity.tex
Run from 04_Code:  python ../05_Data/data_audit.py
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_Code"))
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from lcurve import data
from lcurve.tasks import row_hash

HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
for ds in ("nslkdd", "unsw", "cicids", "ciciot"):
    d = data.LOADERS[ds]()
    H = row_hash(d.X)
    is_b = d.family == "Benign"
    r = {"name": d.name, "rows": int(len(d)), "features": int(d.X.shape[1]),
         "nonfinite_values": int((~np.isfinite(d.X)).sum()),
         "constant_features": int((np.nanstd(d.X, axis=0) == 0).sum()),
         "benign_rows": int(is_b.sum()), "attack_rows": int((~is_b).sum()),
         "dup_rows_overall": int(len(H) - len(np.unique(H))),
         "dup_share_overall": float(1 - len(np.unique(H)) / len(H)),
         "dup_share_benign": float(1 - len(np.unique(H[is_b])) / max(1, is_b.sum())),
         "dup_share_attack": float(1 - len(np.unique(H[~is_b])) / max(1, (~is_b).sum())),
         "vectors_both_labels": int(len(np.intersect1d(np.unique(H[is_b]), np.unique(H[~is_b])))),
         "families": {f: {"rows": int((d.family == f).sum()), "unique": int(len(np.unique(H[d.family == f])))}
                      for f in sorted(set(d.family))},
         "feature_names": list(d.features)}
    if ds == "unsw":
        r["periods"] = {p: int((d.period == p).sum()) for p in sorted(set(d.period))}
        r["hosts"] = int(len(set(d.host)))
        meta = pd.read_parquet(os.path.join(HERE, "cache", "unsw_raw.parquet"), columns=["label", "host", "src_attacker"])
        prot = meta["host"].astype(str).isin(data.UNSW_PROTECTED_HOSTS)
        r["raw_rows"] = int(len(meta))
        r["rows_to_protected_hosts"] = int(prot.sum())
        r["benign_from_attacker_subnet_removed"] = int((prot & (meta["label"] == 0) & meta["src_attacker"]).sum())
        # TTL artefact: depth-2 tree on TTL features only, balanced error on a random half split
        dt = data.load_unsw_raw(drop_ttl=False)
        cols = [dt.features.index(c) for c in data.UNSW_TTL_FEATURES]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(dt), 400_000, replace=False)
        X, y = dt.X[idx][:, cols], dt.y[idx]
        h = len(idx) // 2
        clf = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=0).fit(X[:h], y[:h])
        p = clf.predict(X[h:])
        r["ttl_only_depth2_balanced_error"] = float(0.5 * ((p[y[h:] == 0] == 1).mean() + (p[y[h:] == 1] == 0).mean()))
        del dt
    if ds == "cicids":
        r["days"] = {p: int((d.period == p).sum()) for p in sorted(set(d.period))}
    res[ds] = r
    print(ds, {k: v for k, v in r.items() if k not in ("families", "feature_names")}, flush=True)
    del d, H
json.dump(res, open(os.path.join(HERE, "data_audit_results.json"), "w"), indent=1)

# Supplementary Table S1
L = [r"\begin{table}[ht]\centering\small",
     r"\caption{Data-integrity audit (05\_Data/data\_audit.py), after the documented cleaning rules. Dup.: share of rows "
     r"that exactly duplicate another row of the same dataset. Both labels: distinct feature vectors that occur with a "
     r"benign and an attack label.}",
     r"\label{tab:s_integrity}",
     r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrr}", r"\toprule",
     r"Dataset & Rows & Features & Non-finite & Constant & Dup. benign (\%) & Dup. attack (\%) & Both labels \\",
     r"\midrule"]
for ds, r in res.items():
    L.append(f"{r['name']} & {r['rows']:,} & {r['features']} & {r['nonfinite_values']:,} & {r['constant_features']} & "
             f"{100 * r['dup_share_benign']:.1f} & {100 * r['dup_share_attack']:.1f} & {r['vectors_both_labels']:,} \\\\")
L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
os.makedirs(os.path.join(HERE, "..", "15_Supplementary"), exist_ok=True)
open(os.path.join(HERE, "..", "15_Supplementary", "supp_table_integrity.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written")
