"""Experiment driver.

  python run.py tasks                 # build per-task arrays (05_Data/tasks)
  python run.py curves  [--jobs 6]    # learning curves: main, chrono/rcontrol, sensitivities
  python run.py geometry              # intrinsic dimension and overlap descriptors
  python run.py analyze               # fits, hypothesis tests, Master_Results.csv
  python run.py figures               # figures and tables
  python run.py smoke                 # timing check on one task (not a result)

All settings come from configs/default.json (pre-specified)."""
import argparse, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joblib import Parallel, delayed
from lcurve import config

P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.json")))
LOG = os.path.join(config.LOG_DIR, "run_log.txt")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cells():
    """Every (dataset, family, variant, order, alt, learners) cell of the design."""
    from lcurve import tasks
    inv = tasks.inventory()
    main = inv[(inv.variant == "main") & (inv.family != "Benign")]
    out = []
    for r in main.itertuples():
        out.append((r.dataset, r.family, "main", "random", False, None))
        if r.dataset in ("unsw", "cicids"):
            out.append((r.dataset, r.family, "main", "chrono", False, None))
            out.append((r.dataset, r.family, "main", "rcontrol", False, None))
        out.append((r.dataset, r.family, "main", "random", True, ["RF", "kNN5"]))   # RF500 / kNN10
    for r in inv[(inv.variant == "dup") & (inv.family != "Benign")].itertuples():
        out.append((r.dataset, r.family, "dup", "random", False, None))
    for r in inv[(inv.variant == "ttl") & (inv.family != "Benign")].itertuples():
        out.append((r.dataset, r.family, "ttl", "random", False, None))
    return out


def _cell(c):
    from lcurve import curves
    ds, fam, variant, order, alt, learners = c
    try:
        return c, curves.run_cell(P, ds, fam, variant, order, alt=alt, learners=learners)[1]
    except Exception:
        return c, "ERROR " + traceback.format_exc()[-800:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tasks", "curves", "geometry", "analyze", "figures", "smoke"])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.stage == "tasks":
        from lcurve import tasks
        inv = tasks.build(P, log=log)
        log(f"tasks: {len(inv)} rows written")
    elif a.stage == "curves":
        todo = cells()
        # largest tasks first so that the pool stays busy at the end
        from lcurve import tasks
        size = tasks.inventory().set_index("task").n_used.to_dict()
        todo.sort(key=lambda c: -size.get(tasks.task_name(c[0], c[1], c[2]), 0))
        log(f"curves: {len(todo)} cells, jobs={a.jobs}")
        for c, msg in Parallel(n_jobs=a.jobs, batch_size=1)(delayed(_cell)(c) for c in todo):
            log(f"  {c[:5]} {msg}")
    elif a.stage == "geometry":
        from lcurve import geometry, tasks
        inv = tasks.inventory()
        tl = [(r.dataset, r.family) for r in inv[(inv.variant == "main") & (inv.family != "Benign")].itertuples()]
        # canonical order (dataset order of the config, then family) so that per-task seeds never depend on
        # the order in which tasks were (re)built
        tl.sort(key=lambda t: (P["datasets"].index(t[0]), t[1]))
        geometry.run(P, tl, log=log)
    elif a.stage == "smoke":
        from lcurve import curves
        t = time.time()
        recs = curves.run_unit(P, "nslkdd", "Probe", "main", "random", 0)
        import pandas as pd
        df = pd.DataFrame(recs)
        print(df.pivot_table(index="n", columns="learner", values="be").round(4))
        print(df.groupby("learner").sec.sum().round(1), f"total {time.time() - t:.0f}s")
    else:
        import importlib
        mod = {"analyze": "lcurve.analysis", "figures": "lcurve.figures"}[a.stage]
        getattr(importlib.import_module(mod), "stage")(P, log)
