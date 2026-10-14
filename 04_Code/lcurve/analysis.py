"""Aggregation, pre-specified hypothesis tests (H1-H6), sensitivity analyses and
13_Results/Master_Results.csv. The task (dataset x family) is the unit of inference."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from . import config, fitting, tasks
from .geometry import GEOM_CSV

STAT_DIR = os.path.join(config.PROJECT_ROOT, "09_Statistical_Analysis")
ROB_DIR = os.path.join(config.PROJECT_ROOT, "10_Robustness")
ERR_DIR = os.path.join(config.RESULTS_DIR, "error_analysis")
for _d in (STAT_DIR, ROB_DIR, ERR_DIR):
    os.makedirs(_d, exist_ok=True)
MASTER = []
DS_NAME = {"nslkdd": "NSL-KDD", "unsw": "UNSW-NB15", "cicids": "CICIDS2017", "ciciot": "CIC-IoT2023"}


def put(analysis, metric, value, dataset="all", family="all", learner="all", ci=(np.nan, np.nan), n=np.nan,
        note="", source=""):
    MASTER.append(dict(analysis=analysis, dataset=dataset, family=family, learner=learner, metric=metric,
                       value=value, ci_low=ci[0], ci_high=ci[1], n=n, note=note, source=source))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def spearman_boot(x, y, B, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho, p = stats.spearmanr(x, y)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            bs.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(rho), float(p), (float(lo), float(hi)), int(len(x))


def _summ_chunk(g, n_boot, seed):
    """seed is derived from the curve's own key, so adding or removing other curves
    never changes this curve's bootstrap draws."""
    import zlib
    key = "|".join(str(g[k].iloc[0]) for k in fitting.KEYS)
    return fitting.summarise(g, n_boot=n_boot, rng=np.random.default_rng([seed, zlib.crc32(key.encode())]))


def summarise_all(P, jobs=6, log=print):
    df = fitting.load_curves()
    df.to_parquet(os.path.join(STAT_DIR, "all_curve_records.parquet"))
    groups = [g for _, g in df.groupby(fitting.KEYS)]
    cached = os.path.join(STAT_DIR, "curve_summary.csv")
    if os.path.exists(cached):                    # reuse fits when the set of curves has not changed
        s = pd.read_csv(cached)
        have = set(map(tuple, s[fitting.KEYS].astype(str).values))
        need = {tuple(str(g[k].iloc[0]) for k in fitting.KEYS) for g in groups}
        if need <= have:
            log(f"  reusing {len(s)} fitted curves from curve_summary.csv")
            return df, s[s[fitting.KEYS].astype(str).apply(tuple, axis=1).isin(need)].reset_index(drop=True)
        done = [g for g in groups if tuple(str(g[k].iloc[0]) for k in fitting.KEYS) in have]
        groups = [g for g in groups if tuple(str(g[k].iloc[0]) for k in fitting.KEYS) not in have]
        log(f"  {len(done)} curves reused, {len(groups)} new")
    else:
        s = None
    log(f"  {len(groups)} curves; fitting with bootstrap on main/random curves")
    res = Parallel(n_jobs=jobs, batch_size=4)(
        delayed(_summ_chunk)(g, P["bootstrap"] if (g.variant.iloc[0] == "main" and g.order.iloc[0] == "random"
                                                   and g.alt.iloc[0] == 0) else 0, P["seed"])
        for g in groups)
    s = pd.concat(([s] if s is not None else []) + res, ignore_index=True)
    s.to_csv(os.path.join(STAT_DIR, "curve_summary.csv"), index=False)
    return df, s


def best_learner(s_main):
    """Per task, the learner with the lowest balanced error at the largest common n."""
    out = {}
    for (ds, fam), g in s_main.groupby(["dataset", "family"]):
        out[(ds, fam)] = g.sort_values(["e_max_n", "learner"]).learner.iloc[0]
    return out


def stage(P, log=print, jobs=6):
    MASTER.clear()
    rng = np.random.default_rng(P["seed"])
    df, s = summarise_all(P, jobs=jobs, log=log)
    inv = tasks.inventory()
    geo = pd.read_csv(GEOM_CSV)
    main = s[(s.variant == "main") & (s.order == "random") & (s.alt == 0)].copy()
    ntask = main[["dataset", "family"]].drop_duplicates().shape[0]
    put("design", "tasks", ntask, source="curve_summary.csv")
    put("design", "curve_records", len(df), source="all_curve_records.parquet")
    put("design", "learners", main.learner.nunique(), source="curve_summary.csv")
    for ds in P["datasets"]:
        put("design", "tasks", main[main.dataset == ds].family.nunique(), dataset=ds, source="curve_summary.csv")
    tinv = inv[(inv.variant == "main") & (inv.family != "Benign")]
    put("data", "conflict_vectors_max", int(tinv.conflict_vectors.max()), source="task_inventory.csv")
    put("data", "conflict_floor_max", float(tinv.conflict_floor.max()), source="task_inventory.csv")
    for r in tinv.itertuples():
        put("data", "dup_share", 1 - r.n_unique / r.n_rows, dataset=r.dataset, family=r.family,
            source="task_inventory.csv")
        put("data", "conflict_floor", r.conflict_floor, dataset=r.dataset, family=r.family, source="task_inventory.csv")

    # ---------------- H1: curve shape
    sh = main.dropna(subset=["M1_aicc", "M2_aicc", "M3_aicc"]).copy()
    sh["best_model"] = sh[["M1_aicc", "M2_aicc", "M3_aicc"]].idxmin(axis=1).str[:2]
    k = int((sh.best_model == "M2").sum()); n = len(sh)
    put("H1", "share_M2_best", k / n, ci=wilson(k, n), n=n, source="curve_summary.csv")
    for m in ("M1", "M3"):
        put("H1", f"share_{m}_best", float((sh.best_model == m).mean()), n=n, source="curve_summary.csv")
    for L, g in sh.groupby("learner"):
        put("H1", "share_M2_best", float((g.best_model == "M2").mean()), learner=L, n=len(g),
            source="curve_summary.csv")
    sh["dAIC_M2_M1"] = sh.M1_aicc - sh.M2_aicc
    put("H1", "median_dAICc_M1_minus_M2", float(sh.dAIC_M2_M1.median()), n=n, source="curve_summary.csv")
    sh[["dataset", "family", "learner", "best_model", "M1_aicc", "M2_aicc", "M3_aicc"]].to_csv(
        os.path.join(STAT_DIR, "H1_model_selection.csv"), index=False)

    # ---------------- H2: sample complexity
    bl = best_learner(main)
    rows = []
    for (ds, fam), g in main.groupby(["dataset", "family"]):
        gi = g.set_index("learner")
        tree = gi.loc[["RF", "HGB"], "nstar_0.01"].min()
        rows.append(dict(dataset=ds, family=fam, best=bl[(ds, fam)], nstar_tree=tree,
                         nstar_best=gi.loc[bl[(ds, fam)], "nstar_0.01"],
                         nstar5_tree=gi.loc[["RF", "HGB"], "nstar_0.05"].min(),
                         e_max_best=gi.loc[bl[(ds, fam)], "e_max_n"], n_max=gi.n_max.max(),
                         **{f"nstar_{L}": gi.loc[L, "nstar_0.01"] for L in gi.index},
                         **{f"emax_{L}": gi.loc[L, "e_max_n"] for L in gi.index}))
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(STAT_DIR, "H2_sample_complexity.csv"), index=False)
    k = int((t2.nstar_tree <= 1024).sum()); n = len(t2)
    put("H2", "share_tasks_nstar_tree_le_1024", k / n, ci=wilson(k, n), n=n,
        note="one-sided exact binomial vs 0.5: P=%.3g" % stats.binomtest(k, n, 0.5, alternative="greater").pvalue,
        source="H2_sample_complexity.csv")
    put("H2", "binom_p_vs_half", stats.binomtest(k, n, 0.5, alternative="greater").pvalue, n=n,
        source="H2_sample_complexity.csv")
    fin = t2.nstar_tree[np.isfinite(t2.nstar_tree)]
    put("H2", "median_nstar_tree_reached", float(fin.median()), n=len(fin), source="H2_sample_complexity.csv")
    put("H2", "tasks_never_1pct_tree", int((~np.isfinite(t2.nstar_tree)).sum()), n=n, source="H2_sample_complexity.csv")
    for r in t2.itertuples():
        put("H2", "nstar_tree_1pct", r.nstar_tree, dataset=r.dataset, family=r.family, learner="RF|HGB",
            source="H2_sample_complexity.csv")
        put("H2", "e_max_best", r.e_max_best, dataset=r.dataset, family=r.family, learner=r.best,
            source="H2_sample_complexity.csv")
    for L in main.learner.unique():
        v = t2[f"nstar_{L}"]
        put("H2", "share_tasks_nstar_le_1024", float((v <= 1024).mean()), learner=L, n=n,
            source="H2_sample_complexity.csv")
        put("H2", "median_nstar_all_tasks", float(np.median(np.where(np.isfinite(v), v, 1e9))), learner=L, n=n,
            note="censored values ranked last", source="H2_sample_complexity.csv")
    for ds, g in t2.groupby("dataset"):
        put("H2", "share_tasks_nstar_tree_le_1024", float((g.nstar_tree <= 1024).mean()), dataset=ds, n=len(g),
            source="H2_sample_complexity.csv")
    # detection-based outcome
    det = main.groupby(["dataset", "family"]).apply(lambda g: g[g.learner.isin(["RF", "HGB"])].nstar_det95.min())
    put("H2", "share_tasks_nstar_det95_tree_le_1024", float((det <= 1024).mean()), n=len(det),
        note="miss rate <= 5% at 1% FPR", source="curve_summary.csv")

    # ---------------- H3: floors vs Cover-Hart bounds
    rows = []
    for (ds, fam), g in main.groupby(["dataset", "family"]):
        gi = g.set_index("learner")
        b = bl[(ds, fam)]
        R1 = gi.loc["kNN1", "e_max_n"]
        lo = (1 - np.sqrt(max(0.0, 1 - 2 * R1))) / 2
        c, clo, chi = gi.loc[b, "M2_c"], gi.loc[b, "M2_c_lo"], gi.loc[b, "M2_c_hi"]
        gg = geo[(geo.dataset == ds) & (geo.family == fam)].iloc[0]
        cf = tinv[(tinv.dataset == ds) & (tinv.family == fam)].conflict_floor.iloc[0]
        rows.append(dict(dataset=ds, family=fam, best=b, c=c, c_lo=clo, c_hi=chi, R1=R1, CH_lower=lo, CH_upper=R1,
                         within=bool((chi >= lo) and (clo <= R1)), e_max_best=gi.loc[b, "e_max_n"],
                         conflict_floor=cf, nn_cross_attack=gg.nn_cross_attack, nn_cross_benign=gg.nn_cross_benign,
                         floor_gt_1pct=bool(c > 0.01)))
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(STAT_DIR, "H3_floors.csv"), index=False)
    k = int(t3.within.sum()); n = len(t3)
    put("H3a", "share_floor_within_CoverHart", k / n, ci=wilson(k, n), n=n, source="H3_floors.csv")
    put("H3a", "tasks_floor_gt_1pct", int(t3.floor_gt_1pct.sum()), n=n, source="H3_floors.csv")
    put("H3a", "tasks_floor_below_CoverHart", int((t3.c_hi < t3.CH_lower).sum()), n=n, source="H3_floors.csv")
    put("H3a", "tasks_floor_ci_includes_zero", int((t3.c_lo <= 1e-12).sum()), n=n,
        note="lower bootstrap limit of the fitted floor at the zero boundary", source="H3_floors.csv")
    put("H3a", "tasks_floor_above_CoverHart", int((t3.c_lo > t3.CH_upper).sum()), n=n, source="H3_floors.csv")
    put("H3a", "max_conflict_floor", float(t3.conflict_floor.max()), n=n, source="H3_floors.csv")
    t3["overlap"] = (t3.nn_cross_attack + t3.nn_cross_benign) / 2
    rho, p, ci, nn = spearman_boot(t3.overlap, t3.c, P["bootstrap_tasks"], rng)
    put("H3b", "spearman_floor_vs_overlap", rho, ci=ci, n=nn, note=f"P={p:.3g}", source="H3_floors.csv")
    put("H3b", "spearman_floor_vs_overlap_P", p, n=nn, source="H3_floors.csv")
    hi = t3[t3.floor_gt_1pct]; lo_ = t3[~t3.floor_gt_1pct]
    if len(hi) and len(lo_):
        u = stats.mannwhitneyu(hi.overlap, lo_.overlap, alternative="two-sided")
        put("H3b", "overlap_median_floor_gt_1pct", float(hi.overlap.median()), n=len(hi), source="H3_floors.csv")
        put("H3b", "overlap_median_floor_le_1pct", float(lo_.overlap.median()), n=len(lo_), source="H3_floors.csv")
        put("H3b", "mannwhitney_P", float(u.pvalue), n=len(t3), source="H3_floors.csv")
    for r in t3.itertuples():
        put("H3", "floor_c", r.c, dataset=r.dataset, family=r.family, learner=r.best, ci=(r.c_lo, r.c_hi),
            source="H3_floors.csv")
        put("H3", "R1_kNN1", r.R1, dataset=r.dataset, family=r.family, learner="kNN1", source="H3_floors.csv")
        put("H3", "nn_cross_attack", r.nn_cross_attack, dataset=r.dataset, family=r.family, source="H3_floors.csv")
        put("H3", "overlap", r.overlap, dataset=r.dataset, family=r.family, source="H3_floors.csv")

    # ---------------- H4: geometry
    rows = []
    for (ds, fam), g in main.groupby(["dataset", "family"]):
        gi = g.set_index("learner"); b = bl[(ds, fam)]
        gg = geo[(geo.dataset == ds) & (geo.family == fam)].iloc[0]
        rows.append(dict(dataset=ds, family=fam, best=b, b_exp=gi.loc[b, "M2_b"], c=gi.loc[b, "M2_c"],
                         nstar=gi.loc[b, "nstar_0.01"], nstar_tree=t2[(t2.dataset == ds) & (t2.family == fam)].nstar_tree.iloc[0],
                         **{k_: gg[k_] for k_ in geo.columns if k_ not in ("dataset", "family")},
                         **{f"b_{L}": gi.loc[L, "M2_b"] for L in gi.index},
                         **{f"nstar_{L}": gi.loc[L, "nstar_0.01"] for L in gi.index}))
    t4 = pd.DataFrame(rows)
    t4.to_csv(os.path.join(STAT_DIR, "H4_geometry.csv"), index=False)
    cap = 2 * main.n_max.max()
    lognstar = np.log2(np.where(np.isfinite(t4.nstar), t4.nstar, cap))
    r_a = spearman_boot(t4.twonn_attack, t4.b_exp, P["bootstrap_tasks"], rng)
    r_b = spearman_boot(t4.twonn_attack, lognstar, P["bootstrap_tasks"], rng)
    adj = holm([r_a[1], r_b[1]])
    put("H4a", "spearman_ID_attack_vs_exponent", r_a[0], ci=r_a[2], n=r_a[3], note=f"P={r_a[1]:.3g}; Holm P={adj[0]:.3g}",
        source="H4_geometry.csv")
    put("H4a", "P_holm", adj[0], n=r_a[3], source="H4_geometry.csv")
    put("H4b", "spearman_ID_attack_vs_log_nstar", r_b[0], ci=r_b[2], n=r_b[3], note=f"P={r_b[1]:.3g}; Holm P={adj[1]:.3g}",
        source="H4_geometry.csv")
    put("H4b", "P_holm", adj[1], n=r_b[3], source="H4_geometry.csv")
    # exploratory: other geometric descriptors and per learner
    for col in ("twonn_benign", "twonn_union", "mle10_attack", "mle20_attack", "twonn_attack_cont"):
        for tgt, y in (("exponent", t4.b_exp), ("log_nstar", lognstar), ("floor", t4.c)):
            r = spearman_boot(t4[col], y, 500, rng)
            put("H4-explore", f"spearman_{col}_vs_{tgt}", r[0], ci=r[2], n=r[3], note=f"P={r[1]:.3g}",
                source="H4_geometry.csv")
    r = spearman_boot(t4.twonn_attack, t4.c, 500, rng)
    put("H4-explore", "spearman_twonn_attack_vs_floor", r[0], ci=r[2], n=r[3], note=f"P={r[1]:.3g}", source="H4_geometry.csv")
    for L in main.learner.unique():
        r = spearman_boot(t4.twonn_attack, t4[f"b_{L}"], 500, rng)
        put("H4-explore", "spearman_ID_attack_vs_exponent", r[0], learner=L, ci=r[2], n=r[3], note=f"P={r[1]:.3g}",
            source="H4_geometry.csv")
        y = np.log2(np.where(np.isfinite(t4[f"nstar_{L}"]), t4[f"nstar_{L}"], cap))
        r = spearman_boot(t4.twonn_attack, y, 500, rng)
        put("H4-explore", "spearman_ID_attack_vs_log_nstar", r[0], learner=L, ci=r[2], n=r[3], note=f"P={r[1]:.3g}",
            source="H4_geometry.csv")
    r = spearman_boot(geo.twonn_attack, geo.mle20_attack, 500, rng)
    put("H4-explore", "spearman_TwoNN_vs_MLE20_attack", r[0], ci=r[2], n=r[3], source="geometry.csv")
    put("H4", "twonn_attack_min", float(geo.twonn_attack.min()), source="geometry.csv")
    put("H4", "twonn_attack_max", float(geo.twonn_attack.max()), source="geometry.csv")

    # ---------------- H5: time
    ch = s[(s.variant == "main") & (s.order.isin(["chrono", "rcontrol"])) & (s.alt == 0)]
    rows = []
    for (ds, fam, L), g in ch.groupby(["dataset", "family", "learner"]):
        if set(g.order) != {"chrono", "rcontrol"}:
            continue
        gi = g.set_index("order")
        # compare at the largest n reached by both
        sub = df[(df.dataset == ds) & (df.family == fam) & (df.variant == "main") & (df.alt == 0) & (df.learner == L)
                 & df.order.isin(["chrono", "rcontrol"])]
        nm = sub.groupby("order").n.max().min()
        e_at = sub[sub.n == nm].groupby("order").be.mean()
        rows.append(dict(dataset=ds, family=fam, learner=L, n_common=nm, be_chrono=e_at["chrono"],
                         be_rcontrol=e_at["rcontrol"], nstar_chrono=gi.loc["chrono", "nstar_0.01"],
                         nstar_rcontrol=gi.loc["rcontrol", "nstar_0.01"]))
    t5 = pd.DataFrame(rows)
    t5.to_csv(os.path.join(STAT_DIR, "H5_time.csv"), index=False)
    ps = []
    for L in ("RF", "HGB"):
        g = t5[t5.learner == L]
        w = stats.wilcoxon(g.be_chrono, g.be_rcontrol, alternative="two-sided")
        ps.append(w.pvalue)
        put("H5", "median_be_chrono", float(g.be_chrono.median()), learner=L, n=len(g), source="H5_time.csv")
        put("H5", "median_be_rcontrol", float(g.be_rcontrol.median()), learner=L, n=len(g), source="H5_time.csv")
        put("H5", "tasks_chrono_worse", int((g.be_chrono > g.be_rcontrol).sum()), learner=L, n=len(g),
            source="H5_time.csv")
        put("H5", "median_ratio_chrono_over_rcontrol", float(np.median(g.be_chrono / np.maximum(g.be_rcontrol, 1e-4))),
            learner=L, n=len(g), source="H5_time.csv")
        put("H5", "tasks_chrono_never_1pct", int((~np.isfinite(g.nstar_chrono)).sum()), learner=L, n=len(g),
            source="H5_time.csv")
        put("H5", "tasks_rcontrol_never_1pct", int((~np.isfinite(g.nstar_rcontrol)).sum()), learner=L, n=len(g),
            source="H5_time.csv")
    adj = holm(ps)
    for L, p, pa in zip(("RF", "HGB"), ps, adj):
        put("H5", "wilcoxon_P", p, learner=L, note=f"Holm P={pa:.3g}", source="H5_time.csv")
        put("H5", "wilcoxon_P_holm", pa, learner=L, source="H5_time.csv")
    for r in t5[t5.learner == "RF"].itertuples():
        put("H5", "be_chrono", r.be_chrono, dataset=r.dataset, family=r.family, learner="RF", source="H5_time.csv")
        put("H5", "be_rcontrol", r.be_rcontrol, dataset=r.dataset, family=r.family, learner="RF", source="H5_time.csv")

    # ---------------- H6: forecasting
    fc = main.copy()
    rows = []
    for r in fc.to_dict("records"):          # column names contain dots, so not itertuples
        for m in ("M1", "M2"):
            rows.append(dict(dataset=r["dataset"], family=r["family"], learner=r["learner"], model=m,
                             observed=r["nstar_0.01"], forecast=r.get(f"fc_{m}_0.01", np.nan), n_max=r["n_max"]))
    t6 = pd.DataFrame(rows)
    # geometry-informed: leave-one-task-out regression of log2 n* on attack TwoNN (per learner)
    gm = []
    for L in main.learner.unique():
        sub = main[main.learner == L].merge(geo[["dataset", "family", "twonn_attack"]], on=["dataset", "family"])
        y = np.log2(np.where(np.isfinite(sub["nstar_0.01"]), sub["nstar_0.01"], 2 * sub.n_max))
        x = sub.twonn_attack.values
        for i in range(len(sub)):
            m_ = np.ones(len(sub), bool); m_[i] = False
            sl, ic = np.polyfit(x[m_], y[m_], 1)
            gm.append(dict(dataset=sub.dataset.iloc[i], family=sub.family.iloc[i], learner=L, model="geometry",
                           observed=sub["nstar_0.01"].iloc[i], forecast=2 ** (ic + sl * x[i]), n_max=sub.n_max.iloc[i]))
    t6 = pd.concat([t6, pd.DataFrame(gm)], ignore_index=True)

    def correct(row):
        o, f, nm = row.observed, row.forecast, row.n_max
        oc, fcen = not np.isfinite(o), (not np.isfinite(f)) or f > nm
        if oc and fcen:
            return True
        if oc != fcen:
            return False
        return abs(np.log2(f / o)) <= 1
    t6["within2"] = t6.apply(correct, axis=1)
    t6.to_csv(os.path.join(STAT_DIR, "H6_forecast.csv"), index=False)
    for m, g in t6.groupby("model"):
        k = int(g.within2.sum()); n = len(g)
        put("H6", "share_within_factor2", k / n, learner="all", ci=wilson(k, n), n=n, note=m, source="H6_forecast.csv")
        r_ = g[np.isfinite(g.observed)]                       # curves that reached 1 %
        k2 = int(r_.within2.sum())
        put("H6", "share_within_factor2_reached", k2 / len(r_), learner="all", ci=wilson(k2, len(r_)), n=len(r_),
            note=m, source="H6_forecast.csv")
        f_ = r_[np.isfinite(r_.forecast) & (r_.forecast <= r_.n_max)]
        put("H6", "median_log2_forecast_over_observed", float(np.median(np.log2(f_.forecast / f_.observed))),
            learner="all", n=len(f_), note=m, source="H6_forecast.csv")
        put("H6", "both_censored", int((~np.isfinite(g.observed) & (~np.isfinite(g.forecast) | (g.forecast > g.n_max))).sum()),
            learner="all", n=n, note=m, source="H6_forecast.csv")
        miss = g[~g.within2]
        over = int((miss.forecast.fillna(np.inf) > miss.observed.fillna(np.inf)).sum())
        put("H6", "misses", len(miss), learner="all", n=n, note=m, source="H6_forecast.csv")
        put("H6", "misses_overestimate", over, learner="all", n=len(miss), note=m, source="H6_forecast.csv")
        for L, gl in g.groupby("learner"):
            put("H6", "share_within_factor2", float(gl.within2.mean()), learner=L, n=len(gl), note=m,
                source="H6_forecast.csv")
    a, b_ = t6[t6.model == "M2"].within2.values, t6[t6.model == "M1"].within2.values
    if len(a) == len(b_):
        n01 = int(((a == 1) & (b_ == 0)).sum()); n10 = int(((a == 0) & (b_ == 1)).sum())
        p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        put("H6", "mcnemar_M2_vs_M1_P", p, n=len(a), note=f"discordant {n01}/{n10}", source="H6_forecast.csv")
    a2 = t6[t6.model == "geometry"].sort_values(["dataset", "family", "learner"]).within2.values
    b2 = t6[t6.model == "M2"].sort_values(["dataset", "family", "learner"]).within2.values
    if len(a2) == len(b2):
        n01 = int(((a2 == 1) & (b2 == 0)).sum()); n10 = int(((a2 == 0) & (b2 == 1)).sum())
        p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        put("H6", "mcnemar_geometry_vs_M2_P", p, n=len(a2), note=f"discordant {n01}/{n10}", source="H6_forecast.csv")

    # ---------------- sensitivity analyses
    sens = []

    def compare(variant_mask, label):
        v = s[variant_mask]
        for (ds, fam, L), g in v.groupby(["dataset", "family", "learner"]):
            m_ = main[(main.dataset == ds) & (main.family == fam) & (main.learner == L)]
            if m_.empty:
                continue
            sens.append(dict(analysis=label, dataset=ds, family=fam, learner=L, nstar_main=m_["nstar_0.01"].iloc[0],
                             nstar_var=g["nstar_0.01"].iloc[0], emax_main=m_.e_max_n.iloc[0], emax_var=g.e_max_n.iloc[0]))
    compare((s.variant == "dup") & (s.order == "random"), "duplicates_kept")
    compare((s.variant == "ttl") & (s.order == "random"), "unsw_ttl_restored")
    compare((s.variant == "main") & (s.order == "random") & (s.alt == 1), "rf500_knn10")
    sv = pd.DataFrame(sens)
    sv.to_csv(os.path.join(ROB_DIR, "sensitivity_nstar.csv"), index=False)
    for lab, g in sv.groupby("analysis"):
        for L in ("RF", "HGB", "kNN5"):
            gl = g[g.learner == L]
            if gl.empty:
                continue
            cap_ = 2 * main.n_max.max()
            a_ = np.where(np.isfinite(gl.nstar_main), gl.nstar_main, cap_)
            b_ = np.where(np.isfinite(gl.nstar_var), gl.nstar_var, cap_)
            put("sensitivity", "median_log2_ratio_nstar_var_over_main", float(np.median(np.log2(b_ / a_))), learner=L,
                n=len(gl), note=lab, source="sensitivity_nstar.csv")
            if len(gl) >= 6 and np.any(a_ != b_):
                put("sensitivity", "wilcoxon_P_nstar", float(stats.wilcoxon(np.log2(a_), np.log2(b_)).pvalue),
                    learner=L, n=len(gl), note=lab, source="sensitivity_nstar.csv")
            put("sensitivity", "share_le_1024_var", float((gl.nstar_var <= 1024).mean()), learner=L, n=len(gl),
                note=lab, source="sensitivity_nstar.csv")
            put("sensitivity", "share_le_1024_main", float((gl.nstar_main <= 1024).mean()), learner=L, n=len(gl),
                note=lab, source="sensitivity_nstar.csv")
    # large tasks only
    big = t2.merge(tinv[["dataset", "family", "n_used"]], on=["dataset", "family"])
    bb = big[big.n_used >= 1000]
    k = int((bb.nstar_tree <= 1024).sum())
    put("sensitivity", "share_tasks_nstar_tree_le_1024", k / len(bb), ci=wilson(k, len(bb)), n=len(bb),
        note="tasks with >=1000 unique attack flows", source="H2_sample_complexity.csv")
    t4b = t4.merge(tinv[["dataset", "family", "n_used"]], on=["dataset", "family"])
    t4b = t4b[t4b.n_used >= 1000]
    r = spearman_boot(t4b.twonn_attack, np.log2(np.where(np.isfinite(t4b.nstar), t4b.nstar, cap)), 500, rng)
    put("sensitivity", "spearman_ID_attack_vs_log_nstar", r[0], ci=r[2], n=r[3], note=f">=1000 unique; P={r[1]:.3g}",
        source="H4_geometry.csv")
    # detection outcome vs balanced-error outcome ordering
    dd = main[main.learner == "RF"]
    a_ = np.where(np.isfinite(dd["nstar_0.01"]), dd["nstar_0.01"], cap)
    b_ = np.where(np.isfinite(dd.nstar_det95), dd.nstar_det95, cap)
    put("sensitivity", "spearman_nstar_BE_vs_nstar_det95", float(stats.spearmanr(a_, b_)[0]), learner="RF", n=len(dd),
        source="curve_summary.csv")

    # ---------------- descriptive: error with very few labels
    d4 = df[(df.variant == "main") & (df.order == "random") & (df.alt == 0) & (df.n == 4)]
    for L, g in d4.groupby("learner"):
        tm = g.groupby(["dataset", "family"]).be.mean()
        put("descriptive", "be_at_n4_min", float(tm.min()), learner=L, n=len(tm), source="all_curve_records.parquet")
        put("descriptive", "be_at_n4_max", float(tm.max()), learner=L, n=len(tm), source="all_curve_records.parquet")

    # ---------------- cost
    c = df[(df.variant == "main") & (df.order == "random") & (df.alt == 0)]
    for L, g in c.groupby("learner"):
        put("cost", "median_fit_seconds_n1024", float(g[g.n == 1024].sec.median()), learner=L, source="all_curve_records.parquet")
        put("cost", "median_fit_seconds_nmax", float(g[g.n == g.n.max()].sec.median()), learner=L,
            source="all_curve_records.parquet")
    put("cost", "total_fit_hours", float(df.sec.sum() / 3600), source="all_curve_records.parquet")

    out = pd.DataFrame(MASTER)
    out.to_csv(os.path.join(config.RESULTS_DIR, "Master_Results.csv"), index=False)
    log(f"Master_Results.csv: {len(out)} rows")
    return out
