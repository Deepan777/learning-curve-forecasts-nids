"""Source -> target scenarios built on the original task files (05_Data/tasks, unchanged).

Every scenario returns, for one repeat, row indices into the family's attack array and the dataset's benign array:
  a_src, b_src  source training pools (permuted; the first n of each are used at size n)
  a_val, b_val  held-out source validation sets (the pilot curve is measured here)
  a_tgt, b_tgt  target test sets (the realised target curve is measured here)
Kinds: id, temporal, lovo (leave one variant out), benign (benign-period change). See configs/shift.json.
"""
from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lcurve import config, curves, tasks  # noqa: E402

from . import variants  # noqa: E402

CODE = os.path.join(config.PROJECT_ROOT, "04_Code")
P_ORIG = json.load(open(os.path.join(CODE, "configs", "default.json")))
S = json.load(open(os.path.join(CODE, "configs", "shift.json")))
SHIFT_DIR = os.path.join(config.PROJECT_ROOT, "06_Experiments", "shift")
CICIDS_DAY = {"BruteForce": "Tue", "DoS": "Wed", "WebAttack": "Thu-AM", "Botnet": "Fri-AM",
              "PortScan": "Fri-PM1", "DDoS": "Fri-PM2"}
_FINE = {}


def _seed(*parts):
    return [S["seed"]] + [zlib.crc32(str(p).encode()) % (2 ** 31) for p in parts]


def fine(ds, fam):
    key = (ds, fam)
    if key not in _FINE:
        _, order, _ = curves._get(tasks.task_name(ds, fam))
        _FINE[key] = variants.fine_labels(ds, order).astype(str)
    return _FINE[key]


def enumerate_scenarios():
    inv = tasks.inventory()
    main = inv[(inv.variant == "main") & (inv.family != "Benign")]
    rows = []
    for r in main.itertuples():
        rows.append(dict(sid=f"id__{r.dataset}__{r.family}", kind="id", dataset=r.dataset, family=r.family,
                         heldout=""))
    for r in main.itertuples():
        if r.dataset in ("unsw", "cicids"):
            rows.append(dict(sid=f"temporal__{r.dataset}__{r.family}", kind="temporal", dataset=r.dataset,
                             family=r.family, heldout=""))
            rows.append(dict(sid=f"benign__{r.dataset}__{r.family}", kind="benign", dataset=r.dataset,
                             family=r.family, heldout=""))
    for r in main.itertuples():
        if r.dataset not in ("cicids", "nslkdd", "ciciot"):
            continue
        v = fine(r.dataset, r.family)
        counts = pd.Series(v).value_counts()
        if len(counts) < 2:
            continue
        for var, c in counts.items():
            if c >= S["lovo_min_target"] and (len(v) - c) >= S["lovo_min_source"]:
                rows.append(dict(sid=f"lovo__{r.dataset}__{r.family}__{var}", kind="lovo", dataset=r.dataset,
                                 family=r.family, heldout=var, n_target_variant=int(c),
                                 n_source_variants=int(len(v) - c)))
    # artefact-free copies (amendment 2026-10-12): appended at the end so earlier row positions do not change
    for r in [dict(r) for r in rows if r["dataset"] in S.get("artefact_features", {})]:
        r["sid"] += "__noart"
        rows.append(r)
    sc = pd.DataFrame(rows)
    os.makedirs(SHIFT_DIR, exist_ok=True)
    sc.to_csv(os.path.join(SHIFT_DIR, "scenarios.csv"), index=False)
    return sc


def _carve_val(a_pool, b_pool):
    nva = min(S["val_attack_max"], int(S["val_attack_frac"] * len(a_pool)))
    nvb = min(S["val_benign"], int(0.2 * len(b_pool)))
    return a_pool[nva:], a_pool[:nva], b_pool[nvb:], b_pool[:nvb]


def pools(sc, rep):
    ds, fam, kind = sc["dataset"], sc["family"], sc["kind"]
    Xa, _, pa = curves._get(tasks.task_name(ds, fam))
    Xb, ob, pb = curves._get(f"{ds}__Benign")
    rng = np.random.default_rng(_seed(base_sid(sc["sid"]), rep))   # identical splits with and without artefacts
    if kind in ("id", "temporal"):
        a_tr, a_te, b_tr, b_te = curves._pools(P_ORIG, ds, fam, "main", "random" if kind == "id" else "chrono", rep)
        a_src, a_val, b_src, b_val = _carve_val(a_tr, b_tr)
        return a_src, a_val, a_te, b_src, b_val, b_te
    if kind == "lovo":
        v = fine(ds, fam)
        tgt = rng.permutation(np.flatnonzero(v == sc["heldout"]))
        src = rng.permutation(np.flatnonzero(v != sc["heldout"]))
        a_tgt = tgt[:S["test_attack_max"]]
        B = rng.permutation(len(Xb))
        b_tgt, rest = B[:S["test_benign"]], B[S["test_benign"]:]
        a_src, a_val, b_src, b_val = _carve_val(src, rest)
        return a_src, a_val, a_tgt, b_src, b_val, b_tgt
    if kind == "benign":
        A = rng.permutation(len(Xa))
        nta = min(S["test_attack_max"], int(0.3 * len(A)))
        a_tgt, a_rest = A[:nta], A[nta:]
        if ds == "unsw":
            bs, bt = np.flatnonzero(pb == "S1"), np.flatnonzero(pb == "S2")
        else:
            bs, bt = np.flatnonzero(pb == "Mon"), np.flatnonzero(pb == CICIDS_DAY[fam])
        bs, bt = rng.permutation(bs), rng.permutation(bt)
        b_tgt = bt[:S["test_benign"]]
        a_src, a_val, b_src, b_val = _carve_val(a_rest, bs)
        return a_src, a_val, a_tgt, b_src, b_val, b_tgt
    raise ValueError(kind)


def base_sid(sid):
    return sid[:-len("__noart")] if sid.endswith("__noart") else sid


def feature_names(ds):
    return json.load(open(os.path.join(tasks.TASK_DIR, f"{ds}__features.json")))["features"]


def arrays(sc):
    Xa, _, _ = curves._get(tasks.task_name(sc["dataset"], sc["family"]))
    Xb, _, _ = curves._get(f"{sc['dataset']}__Benign")
    if sc["sid"].endswith("__noart"):
        drop = set(S["artefact_features"][sc["dataset"]])
        keep = [j for j, f in enumerate(feature_names(sc["dataset"])) if f not in drop]
        assert len(keep) == Xa.shape[1] - len(drop)
        return Xa[:, keep], Xb[:, keep]
    return Xa, Xb
