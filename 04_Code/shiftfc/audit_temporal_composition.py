"""Post hoc descriptive check (added 2026-10-13, after all results were known; not part of the follow-up protocol).

Question: in the temporal scenarios, does the later (target) period contain the same attack sub-types as the earlier
(source) period? Only CICIDS2017 records sub-types (fine labels); UNSW-NB15 families carry no sub-type label.
For every temporal scenario and repeat 0, the sub-type shares of the source training pool and of the target test
pool are compared with the total-variation distance (0 = identical composition, 1 = disjoint).
The unsafe share of verifiable temporal forecasts per scenario is joined from the forecast table.
Output: 06_Experiments/audit/temporal_composition.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shiftfc import scenarios  # noqa: E402
from lcurve import config  # noqa: E402

OUT = os.path.join(config.PROJECT_ROOT, "06_Experiments", "audit", "temporal_composition.csv")
FC = os.path.join(config.PROJECT_ROOT, "09_Statistical_Analysis", "shift", "forecasts.csv")


def main():
    sc = pd.read_csv(os.path.join(scenarios.SHIFT_DIR, "scenarios.csv"))
    sc = sc[(sc.kind == "temporal") & ~sc.sid.str.endswith("__noart")]
    fc = pd.read_csv(FC)
    rows = []
    for r in sc.to_dict("records"):
        a_src, a_val, a_tgt, *_ = scenarios.pools(r, 0)
        row = dict(sid=r["sid"], dataset=r["dataset"], family=r["family"], n_src=len(a_src), n_tgt=len(a_tgt))
        if r["dataset"] == "cicids":
            v = scenarios.fine(r["dataset"], r["family"])
            ps = pd.Series(v[a_src]).value_counts(normalize=True)
            pt = pd.Series(v[a_tgt]).value_counts(normalize=True)
            idx = ps.index.union(pt.index)
            ps, pt = ps.reindex(idx, fill_value=0), pt.reindex(idx, fill_value=0)
            row["tv_distance"] = 0.5 * float(np.abs(ps - pt).sum())
            row["subtypes_src"] = "; ".join(f"{k} {100 * x:.1f}%" for k, x in ps.items() if x > 0)
            row["subtypes_tgt"] = "; ".join(f"{k} {100 * x:.1f}%" for k, x in pt.items() if x > 0)
            row["subtypes_only_in_target"] = "; ".join(k for k in idx if ps[k] == 0 and pt[k] > 0)
        f = fc[fc.sid.isin([r["sid"], r["sid"] + "__noart"])]
        f = f[f.sid == r["sid"]] if (f.sid == r["sid"]).any() else f
        if "verifiable" in f.columns and "unsafe" in f.columns:
            fv = f[f.verifiable.astype(bool)]
            row["verifiable"] = int(len(fv))
            row["unsafe"] = int(fv.unsafe.astype(bool).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(out.drop(columns=["subtypes_src"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
