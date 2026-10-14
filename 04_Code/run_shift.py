"""Follow-up study: shift-aware forecasting of labelled-sample requirements (configs/shift.json).

  python run_shift.py scenarios                       enumerate scenarios -> 06_Experiments/shift/scenarios.csv
  python run_shift.py curves --worker W --nworkers K  run the scenario cells assigned to worker W (skip existing)
  python run_shift.py floors                          CIC-IoT2023 residual-error controls
  python run_shift.py analyze | figures | tables
  python run_shift.py smoke                           one repeat of one scenario, timing only

The original pipeline (run.py, lcurve/) and its outputs are not modified.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from shiftfc import scenarios  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--kinds", default="id,temporal,lovo,benign")
    ap.add_argument("--only", default="", help="keep scenario ids containing this text")
    ap.add_argument("--exclude", default="", help="drop scenario ids containing this text")
    ap.add_argument("--reverse", action="store_true", help="process the list from the end (extra helper worker)")
    a = ap.parse_args()
    if a.stage == "scenarios":
        sc = scenarios.enumerate_scenarios()
        print(sc.groupby(["kind", "dataset"]).size())
    elif a.stage == "smoke":
        from shiftfc import transfer
        sc = scenarios.enumerate_scenarios()
        for kind in ("id", "lovo"):
            row = sc[sc.kind == kind].iloc[0].to_dict()
            t0 = time.time()
            rec, sig = transfer.run_rep(row, 0, scenarios.S["learners"])
            print(kind, row["sid"], len(rec), "records", f"{time.time() - t0:.1f}s", sig)
    elif a.stage == "curves":
        from shiftfc import transfer
        path = os.path.join(scenarios.SHIFT_DIR, "scenarios.csv")
        sc = pd.read_csv(path, keep_default_na=False)
        sc = sc[sc.kind.isin(a.kinds.split(","))]
        if a.only:
            sc = sc[sc.sid.str.contains(a.only, regex=False)]
        if a.exclude:
            sc = sc[~sc.sid.str.contains(a.exclude, regex=False)]
        sc = sc.reset_index(drop=True)
        recs = sc.to_dict("records")[::-1] if a.reverse else sc.to_dict("records")
        mine = [r for i, r in enumerate(recs) if i % a.nworkers == a.worker]
        for r in mine:
            print(time.strftime("%H:%M:%S"), r["sid"], transfer.run_cell(r), flush=True)
    elif a.stage == "floors":
        from shiftfc import floors
        floors.run()
    elif a.stage == "analyze":
        from shiftfc import analysis
        analysis.run()
    elif a.stage == "robustness":
        from shiftfc import robustness
        robustness.run()
    elif a.stage == "figures":
        from shiftfc import figures
        figures.run()
    elif a.stage == "tables":
        from shiftfc import tables
        tables.run()
    else:
        raise SystemExit("unknown stage " + a.stage)


if __name__ == "__main__":
    main()
