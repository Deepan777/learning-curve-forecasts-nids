"""Task construction: one task = one attack family against benign traffic of the
same dataset. Writes compact per-task arrays to 05_Data/tasks so that workers
never reload the raw datasets.

Variants
  main   exact duplicates removed within each class; vectors present in both
         classes removed (and counted)
  dup    duplicates kept (sensitivity)
  ttl    UNSW-NB15 with the TTL-derived features restored (sensitivity)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config, data

TASK_DIR = os.path.join(config.PROJECT_ROOT, "05_Data", "tasks")
os.makedirs(TASK_DIR, exist_ok=True)


def row_hash(X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    return pd.util.hash_array(X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel())


def conflict_floor(hb, ha):
    """Plug-in lower bound on the balanced Bayes error from vectors carrying both
    labels: 0.5 * sum_v min(p_B(v), p_F(v)), on the raw (non-de-duplicated) rows."""
    vb, cb = np.unique(hb, return_counts=True)
    va, ca = np.unique(ha, return_counts=True)
    common, ib, ia = np.intersect1d(vb, va, return_indices=True)
    if len(common) == 0:
        return 0.0, 0, 0
    pb = cb[ib] / len(hb); pa = ca[ia] / len(ha)
    return float(0.5 * np.minimum(pb, pa).sum()), int(len(common)), int(cb[ib].sum() + ca[ia].sum())


def _first_unique(h, order):
    """Index of the earliest occurrence (by `order`) of every distinct hash."""
    o = np.argsort(order, kind="stable")
    _, first = np.unique(h[o], return_index=True)
    return np.sort(o[first])


def task_name(ds, fam, variant="main"):
    return f"{ds}__{fam}" + ("" if variant == "main" else f"__{variant}")


def build(P, datasets=None, variants=("main", "dup", "ttl"), log=print):
    """Build every task file; returns the task inventory as a DataFrame."""
    rows = []
    for ds in datasets or P["datasets"]:
        for variant in variants:
            if variant == "ttl" and ds != "unsw":
                continue
            d = data.load_unsw_raw(drop_ttl=False) if variant == "ttl" else data.LOADERS[ds]()
            H = row_hash(d.X)
            is_b = d.family == "Benign"
            rng = np.random.default_rng([P["seed"], 11, list(P["datasets"]).index(ds)])
            # benign pool (shared by the dataset's tasks)
            b_all = np.flatnonzero(is_b)
            if variant == "dup":
                b_idx = np.sort(rng.choice(b_all, min(P["benign_pool"], len(b_all)), replace=False))
            else:
                bu = b_all[_first_unique(H[b_all], d.order[b_all])]
                b_idx = np.sort(rng.choice(bu, min(P["benign_pool"], len(bu)), replace=False))
            hb_raw = H[b_all]
            b_set = set(H[b_idx].tolist()) if variant != "dup" else None
            fams = sorted(set(d.family[~is_b]))
            keep_tasks = []
            for fam in fams:
                a_all = np.flatnonzero(d.family == fam)
                floor, n_conf_vec, n_conf_rows = conflict_floor(hb_raw, H[a_all])
                au = a_all[_first_unique(H[a_all], d.order[a_all])]
                n_unique = len(au)
                if n_unique < P["min_unique_attack"]:
                    log(f"  skip {ds}/{fam}: {n_unique} unique attack flows")
                    continue
                if variant == "dup":
                    a_idx = a_all
                else:
                    conf = np.isin(H[au], np.fromiter(b_set, dtype=H.dtype)) if b_set else np.zeros(len(au), bool)
                    a_idx = au[~conf]
                keep_tasks.append(fam)
                name = task_name(ds, fam, variant)
                np.savez_compressed(os.path.join(TASK_DIR, name + ".npz"),
                                    X=d.X[a_idx].astype(np.float32), order=d.order[a_idx].astype(np.int64),
                                    period=d.period[a_idx].astype(str))
                rows.append(dict(task=name, dataset=ds, family=fam, variant=variant, n_rows=len(a_all),
                                 n_unique=n_unique, n_used=len(a_idx), conflict_vectors=n_conf_vec,
                                 conflict_rows=n_conf_rows, conflict_floor=floor,
                                 periods=",".join(sorted(set(d.period[a_idx].astype(str))))))
                log(f"  {name}: rows {len(a_all)} unique {n_unique} used {len(a_idx)} conflicts {n_conf_vec}")
            # benign file: the pool minus vectors that also occur as any kept attack family (main/ttl)
            if variant != "dup":
                a_h = np.concatenate([H[d.family == f] for f in keep_tasks]) if keep_tasks else np.array([], H.dtype)
                b_idx = b_idx[~np.isin(H[b_idx], a_h)]
            bname = f"{ds}__Benign" + ("" if variant == "main" else f"__{variant}")
            np.savez_compressed(os.path.join(TASK_DIR, bname + ".npz"), X=d.X[b_idx].astype(np.float32),
                                order=d.order[b_idx].astype(np.int64), period=d.period[b_idx].astype(str))
            rows.append(dict(task=bname, dataset=ds, family="Benign", variant=variant, n_rows=len(b_all),
                             n_unique=len(np.unique(hb_raw)), n_used=len(b_idx)))
            json.dump({"features": list(d.features)},
                      open(os.path.join(TASK_DIR, f"{ds}__features" + ("" if variant == "main" else f"__{variant}")
                                        + ".json"), "w"))
            del d, H
    inv = pd.DataFrame(rows)
    path = os.path.join(TASK_DIR, "task_inventory.csv")
    if datasets is not None and os.path.exists(path):      # rebuild of a subset: keep the other datasets' rows
        old = pd.read_csv(path)
        inv = pd.concat([old[~old.dataset.isin(datasets)], inv], ignore_index=True)
    inv.to_csv(path, index=False)
    return inv


def load(name):
    z = np.load(os.path.join(TASK_DIR, name + ".npz"), allow_pickle=False)
    return z["X"], z["order"], z["period"]


def inventory():
    return pd.read_csv(os.path.join(TASK_DIR, "task_inventory.csv"))
