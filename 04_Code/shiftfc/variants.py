"""Attack-variant (sub-label) lookups for the existing task files, and the CIC-IoT2023 caches with fine labels.

The original task files (05_Data/tasks) store, for every flow, its row position `order` in the dataset cache.
Sub-labels are recovered through that key, so the original files are used unchanged:

* CICIDS2017: `raw_label` column of 05_Data/cache/cicids2017_days.parquet (key: `order`).
* NSL-KDD: `class` column of 05_Data/source/KDDTrain_plus_full.parquet (key: row number).
* CIC-IoT2023: the original cache kept only the family. `build_ciciot_fine()` repeats the original sub-sampling
  (lcurve.data.build_ciciot_cache: same file, same seed, same random draws) while keeping the fine label, writes
  05_Data/cache/ciciot2023_families_finelabel.parquet, and checks that every feature value and the family of every
  row equal the original cache. In the same pass it keeps *all* rows of the benign, reconnaissance, spoofing, web and
  brute-force classes in 05_Data/cache/ciciot2023_lowfloor_full.parquet for the residual-error controls.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lcurve import config, data  # noqa: E402

FINE_CACHE = os.path.join(config.DATA_CACHE, "ciciot2023_families_finelabel.parquet")
LOWFLOOR_CACHE = os.path.join(config.DATA_CACHE, "ciciot2023_lowfloor_full.parquet")
LOWFLOOR_FAMILIES = ("Benign", "Recon", "Spoofing", "Web", "BruteForce")


def build_ciciot_fine(log=print):
    rng = np.random.default_rng(data.SUBSAMPLE_SEED)
    path = os.path.join(config.CICIOT_DIR, "train.csv")
    pool, full = [], []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunk["family"] = chunk["label"].astype(str).map(data.ciciot_family)
        keep = np.zeros(len(chunk), dtype=bool)
        for fam, g in chunk.groupby("family").groups.items():          # identical draws to the original
            g = np.asarray(list(g)) - chunk.index[0]
            frac = {"Benign": 1.0, "DDoS": 0.02, "DoS": 0.05, "Mirai": 0.10}.get(fam, 1.0)
            keep[g[rng.random(len(g)) < frac]] = True
        pool.append(chunk.loc[keep])
        full.append(chunk[chunk["family"].isin(LOWFLOOR_FAMILIES)])
        log(f"  chunk ending at row {chunk.index[-1] + 1}")
    # (1) replication of the original cache, with the fine label kept as `fine`
    df = pd.concat(pool, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    parts = []
    for fam, g in df.groupby("family"):
        cap = data.CICIOT_BENIGN_N if fam == "Benign" else data.CICIOT_FAMILY_CAP
        parts.append(g.sample(min(cap, len(g)), random_state=data.SUBSAMPLE_SEED))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=data.SUBSAMPLE_SEED).reset_index(drop=True)
    df["order"] = np.arange(len(df))
    fine = df["label"].astype(str)
    orig = pd.read_parquet(os.path.join(config.DATA_CACHE, "ciciot2023_families.parquet"))
    feat = [c for c in orig.columns if c not in ("label", "family", "period", "host", "order")]
    same_x = np.array_equal(orig[feat].to_numpy(np.float32), df[feat].to_numpy(np.float32))
    same_f = bool((orig["family"].astype(str).to_numpy() == df["family"].astype(str).to_numpy()).all())
    out = pd.DataFrame({"order": df["order"].to_numpy(), "family": df["family"].astype(str), "fine": fine})
    out.to_parquet(FINE_CACHE, compression="zstd")
    # (2) every row of the low-floor families (for the more-data control), deduplicated later per task
    fu = pd.concat(full, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    fu = fu.rename(columns={"label": "fine"})
    fu[feat] = fu[feat].astype(np.float32)
    fu["order"] = np.arange(len(fu))
    fu.to_parquet(LOWFLOOR_CACHE, compression="zstd")
    rep = {"rows_fine_cache": int(len(out)), "features_identical_to_original": bool(same_x),
           "families_identical_to_original": same_f, "lowfloor_rows": fu["family"].value_counts().to_dict(),
           "fine_label_counts": out.groupby(["family", "fine"]).size().astype(int).reset_index()
           .rename(columns={0: "n"}).to_dict("records")}
    json.dump(rep, open(os.path.join(config.PROJECT_ROOT, "05_Data", "ciciot_fine_verification.json"), "w"),
              indent=1)
    log(json.dumps({k: v for k, v in rep.items() if k != "fine_label_counts"}, indent=1))
    return rep


def fine_labels(ds: str, order: np.ndarray) -> np.ndarray:
    """Sub-label of each flow identified by its cache row `order`."""
    if ds == "cicids":
        c = pd.read_parquet(os.path.join(config.DATA_CACHE, "cicids2017_days.parquet"), columns=["raw_label", "order"])
        m = pd.Series(c["raw_label"].astype(str).to_numpy(), index=c["order"].to_numpy())
        return m.reindex(order).to_numpy()
    if ds == "nslkdd":
        c = pd.read_parquet(config.NSLKDD_PARQUET, columns=["class"])
        return c["class"].astype(str).to_numpy()[order]
    if ds == "ciciot":
        c = pd.read_parquet(FINE_CACHE)
        m = pd.Series(c["fine"].to_numpy(), index=c["order"].to_numpy())
        return m.reindex(order).to_numpy()
    raise ValueError(f"no sub-labels for {ds}")


if __name__ == "__main__":
    build_ciciot_fine()
