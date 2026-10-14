"""Analysis of the follow-up study (F1-F6, operational prevalence, censoring). Labels: all F-analyses are follow-up
(specified 2026-10-12 before execution) or exploratory; none is part of the original pre-specified H1-H6.

Outputs
  09_Statistical_Analysis/shift/forecasts.csv     one row per forecast (scenario x learner)
  09_Statistical_Analysis/shift/signals_mean.csv  label-free shift scores averaged over repeats
  13_Results/Shift_Results.csv                    every number quoted from this study (section, metric, value, CI)
"""
from __future__ import annotations

import glob
import json
import os
import zlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lcurve import config
from lcurve.fitting import fit, invert, nstar

from . import scenarios

S = scenarios.S
STAT = os.path.join(config.PROJECT_ROOT, "09_Statistical_Analysis", "shift")
RES = os.path.join(config.PROJECT_ROOT, "13_Results", "Shift_Results.csv")
EPS = S["eps"]
SIGNALS = ["c2st_auc", "c2st_adj", "ks_share", "disagree_delta", "conf_drop", "pilot_width"]
SHIFT_KINDS = ["temporal", "lovo", "benign"]
B = 2000
rows = []
PFX = ""


def put(section, metric, value, lo=np.nan, hi=np.nan, n=np.nan, note="", source=""):
    rows.append(dict(section=PFX + section, metric=metric, value=value, ci_lo=lo, ci_hi=hi, n=n, note=note,
                     source=source))


def load():
    c = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""])
                   for f in glob.glob(os.path.join(scenarios.SHIFT_DIR, "curves", "*.csv"))], ignore_index=True)
    s = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""])
                   for f in glob.glob(os.path.join(scenarios.SHIFT_DIR, "signals", "*.csv"))], ignore_index=True)
    c["heldout"] = c["heldout"].fillna("").astype(str); s["heldout"] = s["heldout"].fillna("").astype(str)
    return c, s


def _interp_err(n, e, x):
    """Mean target error at size x (log-linear interpolation); nan outside the measured range."""
    if not np.isfinite(x) or x < n[0] or x > n[-1]:
        return np.nan
    return float(np.interp(np.log(x), np.log(n), e))


def forecasts(c):
    out = []
    for (sid, L), g in c.groupby(["sid", "learner"]):
        m = g.groupby("n").agg(be_val=("be_val", "mean"), be_tgt=("be_tgt", "mean"), auc=("auc_tgt", "mean"),
                               fpr_cal=("fpr_tgt_cal", "mean"), tpr_cal=("tpr_tgt_cal", "mean"),
                               tpr_val=("tpr_val_cal", "mean"))
        n = m.index.values.astype(float)
        ev, et = m.be_val.values, m.be_tgt.values
        e0 = 1.0 / (2 * g.n_val_a.iloc[0])
        sel = n <= S["pilot_max_n"]
        rec = dict(sid=sid, learner=L, kind=g.kind.iloc[0], dataset=g.dataset.iloc[0], family=g.family.iloc[0],
                   heldout=g.heldout.iloc[0], n_max=float(n.max()), n_val_a=int(g.n_val_a.iloc[0]),
                   n_tgt_a=int(g.n_tgt_a.iloc[0]))
        p_m2 = None
        for mdl in ("M1", "M2"):
            if sel.sum() >= 5:
                p, _, _ = fit(n[sel], ev[sel], e0, mdl)
                rec[f"fc_{mdl}"] = invert(mdl, p, EPS)
                p_m2 = p if mdl == "M2" else p_m2
            else:
                rec[f"fc_{mdl}"] = np.nan
        # pilot uncertainty (no target information): spread of M2 forecasts over a repeat-bootstrap of the pilot
        piv = g[g.n <= S["pilot_max_n"]].pivot_table(index="n", columns="rep", values="be_val")
        rng = np.random.default_rng([S["seed"], zlib.crc32(f"{sid}|{L}".encode())])
        fb = []
        for _ in range(100):
            pick = rng.choice(piv.columns.values, piv.shape[1], replace=True)
            p, _, _ = fit(piv.index.values.astype(float), piv[pick].mean(1).values, e0, "M2",
                          starts=[list(p_m2)] if p_m2 is not None else None)
            fb.append(np.log2(min(invert("M2", p, EPS), 1e9)))
        rec["pilot_width"] = float(np.percentile(fb, 90) - np.percentile(fb, 10))
        rec["nstar_src"] = nstar(n, ev, EPS)                 # full source-validation curve (no shift)
        rec["nstar_tgt"] = nstar(n, et, EPS)                 # realised target requirement
        rec["e_tgt_max"] = float(et[-1]); rec["e_val_max"] = float(ev[-1])
        rec["auc_tgt_max"] = float(m.auc.values[-1])
        rec["e_tgt_at_fc"] = _interp_err(n, et, rec["fc_M2"])
        k = np.flatnonzero(n == min(1024, n.max()))[0]
        rec.update(n_op=float(n[k]), fpr_cal_op=float(m.fpr_cal.values[k]), tpr_cal_op=float(m.tpr_cal.values[k]),
                   tpr_val_op=float(m.tpr_val.values[k]), be_tgt_op=float(et[k]))
        out.append(rec)
    f = pd.DataFrame(out)
    return classify(f, "fc_M2")


def classify(f, col):
    f = f.copy()
    fc, ob, nm = f[col].astype(float), f["nstar_tgt"].astype(float), f["n_max"]
    fc_in = np.isfinite(fc) & (fc <= nm)
    ob_in = np.isfinite(ob)
    f["category"] = np.select([fc_in & ob_in, ~fc_in & ~ob_in, fc_in & ~ob_in, ~fc_in & ob_in],
                              ["measurable", "both_beyond", "false_assurance", "conservative_miss"], default="other")
    f["within2"] = f.category.eq("measurable") & (np.abs(np.log2(fc / ob)) <= 1)
    # unsafe: labelling 2 x forecast would not reach eps on the target. Verifiable only if 2 x forecast <= n_max
    verif = fc_in & (2 * fc <= nm)
    f["verifiable"] = verif
    f["unsafe"] = verif & (~ob_in | (ob > 2 * fc))
    f["log2_ratio"] = np.where(f.category.eq("measurable"), np.log2(fc / ob), np.nan)
    return f


def cluster_ids(df):
    return (df.dataset + "|" + df.family).values


def cboot(df, stat, b=B, seed=0, level="family"):
    """Cluster bootstrap over dataset x family (or dataset) clusters; returns (estimate, lo, hi)."""
    if len(df) == 0:
        return np.nan, np.nan, np.nan, np.array([])
    key = cluster_ids(df) if level == "family" else df.dataset.values
    groups = {k: df[key == k] for k in np.unique(key)}
    ks = np.array(list(groups))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.choice(ks, len(ks), replace=True)
        v = stat(pd.concat([groups[k] for k in pick]))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float(stat(df)), np.nan, np.nan, np.array([])
    return float(stat(df)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), np.array(vals)


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def risk_coverage(score, unsafe):
    o = np.argsort(score, kind="stable")          # accept lowest scores first
    u = np.asarray(unsafe, float)[o]
    k = np.arange(1, len(u) + 1)
    return k / len(u), np.cumsum(u) / k


def aurc(score, unsafe):
    cov, risk = risk_coverage(score, unsafe)
    return float(np.mean(risk))


def lodo_threshold(df, sig, target=0.10):
    """Choose on the other datasets the largest threshold whose selective unsafe rate <= target; apply to each
    held-out dataset. Returns per-dataset realised unsafe rate and coverage."""
    res = []
    for d in sorted(df.dataset.unique()):
        tr, te = df[df.dataset != d], df[df.dataset == d]
        best = -np.inf
        for t in np.unique(tr[sig].values):
            acc = tr[tr[sig] <= t]
            if len(acc) and acc.unsafe.mean() <= target:
                best = max(best, t)
        acc = te[te[sig] <= best]
        res.append(dict(dataset=d, signal=sig, threshold=best, n=len(te), accepted=len(acc),
                        unsafe_accepted=int(acc.unsafe.sum()), unsafe_all=int(te.unsafe.sum())))
    return pd.DataFrame(res)


def core(f, s, sm, src):
    """F1-F5, censoring, operational and baselines for one analysis set."""
    # ---------- category table (F1/F2; four-way separation requested in the audit)
    for k, g in f.groupby("kind"):
        for cat, gc in g.groupby("category"):
            put("categories", f"{k}_{cat}", len(gc), n=len(g), source=src)
        put("categories", f"{k}_target_never_reaches", int((~np.isfinite(g.nstar_tgt)).sum()), n=len(g), source=src)

    # ---------- F1: accuracy among measurable forecasts, by kind, and paired differences vs ID
    for k, g in f.groupby("kind"):
        gm = g[g.category == "measurable"]
        est, lo, hi, _ = cboot(gm, lambda d: d.within2.mean(), seed=1)
        put("F1", f"within2_measurable_{k}", est, lo, hi, n=len(gm), source=src)
        est, lo, hi, _ = cboot(gm, lambda d: np.median(d.log2_ratio), seed=2)
        put("F1", f"median_log2_fc_over_obs_{k}", est, lo, hi, n=len(gm), source=src)
        est, lo, hi, _ = cboot(g, lambda d: (d.within2 | d.category.eq("both_beyond")).mean(), seed=3)
        put("F1", f"correct_incl_both_beyond_{k}", est, lo, hi, n=len(g), source=src)
    fam_id = f[f.kind == "id"]
    for k in SHIFT_KINDS:
        both = f[f.kind.isin(["id", k])]
        both = both[cluster_ids(both).astype(str) != ""]
        fams = set(cluster_ids(f[f.kind == k]))
        both = both[np.isin(cluster_ids(both), list(fams))]

        def diff(d, k=k):
            a = d[(d.kind == k) & d.category.eq("measurable")].within2.mean()
            b = d[(d.kind == "id") & d.category.eq("measurable")].within2.mean()
            return a - b
        est, lo, hi, v = cboot(both, diff, seed=4)
        p = float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))
        put("F1", f"diff_within2_{k}_minus_id_same_families", est, lo, hi, n=len(both),
            note=f"cluster bootstrap P = {p:.4g}", source=src)
    # decomposition of forecast error: extrapolation (pilot vs full source) and shift (source vs target)
    for k, g in f.groupby("kind"):
        ok = np.isfinite(g.nstar_src) & np.isfinite(g.nstar_tgt) & np.isfinite(g.fc_M2)
        if ok.sum():
            put("F1", f"median_log2_shift_tgt_over_src_{k}", float(np.median(np.log2(g.nstar_tgt[ok] / g.nstar_src[ok]))),
                n=int(ok.sum()), source=src)
            put("F1", f"median_log2_extrap_fc_over_src_{k}", float(np.median(np.log2(g.fc_M2[ok] / g.nstar_src[ok]))),
                n=int(ok.sum()), source=src)

    # ---------- F2: direction of failures
    for k, g in f.groupby("kind"):
        fail = g[(g.category == "measurable") & ~g.within2 | g.category.isin(["false_assurance", "conservative_miss"])]
        under = fail.category.eq("false_assurance") | (fail.category.eq("measurable") & (fail.log2_ratio < -1))
        put("F2", f"failures_{k}", len(fail), n=len(g), source=src)
        put("F2", f"failures_underestimate_{k}", int(under.sum()), n=len(fail), source=src)
        gv = g[g.verifiable]
        est, lo, hi, _ = cboot(gv, lambda d: d.unsafe.mean(), seed=5)
        put("F2", f"unsafe_rate_verifiable_{k}", est, lo, hi, n=len(gv), source=src)

    # ---------- F3: label-free signals rank unsafe forecasts (all kinds pooled; verifiable forecasts)
    v = f[f.verifiable].copy()
    put("F3", "verifiable_forecasts", len(v), n=len(f), source=src)
    put("F3", "unsafe_overall", float(v.unsafe.mean()), n=len(v), source=src)
    for sig in SIGNALS:
        vv = v[np.isfinite(v[sig])]
        est, lo, hi, _ = cboot(vv, lambda d, s_=sig: auroc(d[s_], d.unsafe), seed=6)
        put("F3", f"auroc_unsafe_{sig}", est, lo, hi, n=len(vv), source=src)
        est, lo, hi, _ = cboot(vv, lambda d, s_=sig: aurc(d[s_], d.unsafe), seed=7)
        put("F3", f"aurc_{sig}", est, lo, hi, n=len(vv), note="random abstention AURC = unsafe rate", source=src)
        for cov in (0.5, 0.75):
            def sel(d, s_=sig, cov=cov):
                cv, rk = risk_coverage(d[s_].values, d.unsafe.values)
                return float(rk[int(np.ceil(cov * len(rk))) - 1])
            est, lo, hi, _ = cboot(vv, sel, seed=8)
            put("F3", f"selective_unsafe_cov{int(cov * 100)}_{sig}", est, lo, hi, n=len(vv), source=src)
    # shift-only subset (does the score still separate within shifted scenarios?)
    vs = v[v.kind != "id"]
    for sig in SIGNALS:
        vv = vs[np.isfinite(vs[sig])]
        est, lo, hi, _ = cboot(vv, lambda d, s_=sig: auroc(d[s_], d.unsafe), seed=9)
        put("F3", f"auroc_unsafe_shift_only_{sig}", est, lo, hi, n=len(vv), source=src)
    put("F3", "unsafe_shift_only", float(vs.unsafe.mean()), n=len(vs), source=src)

    # ---------- F4: leave-one-dataset-out thresholds
    thr = {}
    for sig in ("c2st_auc", "c2st_adj", "ks_share", "disagree_delta", "conf_drop", "pilot_width"):
        r = lodo_threshold(v[np.isfinite(v[sig])], sig)
        thr[sig] = dict(zip(r.dataset, r.threshold))
        for k, g in v.groupby("kind"):            # descriptive: share accepted per scenario kind
            acc_k = g[sig] <= g.dataset.map(thr[sig])
            put("F4", f"lodo_accept_rate_{sig}_{k}", float(acc_k.mean()), n=len(g),
                note=f"unsafe among accepted {g.unsafe[acc_k].mean() if acc_k.any() else float('nan'):.3f}",
                source=f"09_Statistical_Analysis/shift/{PFX}lodo_{sig}.csv")
        r.to_csv(os.path.join(STAT, f"{PFX}lodo_{sig}.csv"), index=False)
        acc, un = r.accepted.sum(), r.unsafe_accepted.sum()
        put("F4", f"lodo_coverage_{sig}", acc / r.n.sum(), n=int(r.n.sum()), source=f"09_Statistical_Analysis/shift/{PFX}lodo_{sig}.csv")
        put("F4", f"lodo_unsafe_accepted_{sig}", un / acc if acc else np.nan, n=int(acc),
            note=f"no abstention: {r.unsafe_all.sum() / r.n.sum():.4f}", source=f"09_Statistical_Analysis/shift/{PFX}lodo_{sig}.csv")

    # ---------- F5: prevalence-only change (ID scenarios)
    sid_id = s[s.kind == "id"]
    for pt in S["prevalence_targets"]:
        a, b = sid_id.groupby("sid")[f"c2st_auc_prev{pt}"].mean(), sid_id.groupby("sid")[f"c2st_adj_prev{pt}"].mean()
        k_ = sid_id.groupby("sid")[f"ks_share_prev{pt}"].mean()
        put("F5", f"c2st_raw_prev{pt}_median", float(a.median()), float(a.min()), float(a.max()), n=len(a),
            note="ci columns = min, max", source="06_Experiments/shift/signals")
        put("F5", f"c2st_adj_prev{pt}_median", float(b.median()), float(b.min()), float(b.max()), n=len(b),
            note="ci columns = min, max", source="06_Experiments/shift/signals")
        put("F5", f"ks_share_prev{pt}_median", float(k_.median()), float(k_.min()), float(k_.max()), n=len(k_),
            note="ci columns = min, max", source="06_Experiments/shift/signals")
        st = wilcoxon(a.values, b.values)
        put("F5", f"wilcoxon_raw_vs_adj_prev{pt}_P", float(st.pvalue), n=len(a), source="06_Experiments/shift/signals")
    # false alarms of the transferred (leave-one-dataset-out) thresholds when only prevalence changes
    ds_of = sid_id.groupby("sid").dataset.first()
    for base in ("c2st_auc", "c2st_adj", "ks_share"):
        for suf, lab in (("", "no_change"), ("_prev0.05", "prev0.05"), ("_prev0.6", "prev0.6")):
            val = sid_id.groupby("sid")[base + suf].mean()
            alarm = val > ds_of.reindex(val.index).map(thr[base])
            put("F5", f"lodo_false_alarm_{base}_{lab}", float(alarm.mean()), n=len(val),
                note="share of in-distribution tasks abstained at the transferred threshold",
                source="06_Experiments/shift/signals")
    a0, b0 = sid_id.groupby("sid").c2st_auc.mean(), sid_id.groupby("sid").c2st_adj.mean()
    put("F5", "c2st_raw_no_change_median", float(a0.median()), float(a0.min()), float(a0.max()), n=len(a0),
        note="ci columns = min, max", source="06_Experiments/shift/signals")
    put("F5", "c2st_adj_no_change_median", float(b0.median()), float(b0.min()), float(b0.max()), n=len(b0),
        note="ci columns = min, max", source="06_Experiments/shift/signals")
    for k in SHIFT_KINDS:
        g = sm[sm.sid.str.startswith(k + "__")]
        put("F5", f"c2st_adj_median_{k}", float(g.c2st_adj.median()), float(g.c2st_adj.min()), float(g.c2st_adj.max()),
            n=len(g), note="ci columns = min, max", source="09_Statistical_Analysis/shift/signals_mean.csv")

    # ---------- Kaplan-Meier medians of target n* (right-censored) by kind
    for k, g in f.groupby("kind"):
        t = np.where(np.isfinite(g.nstar_tgt), g.nstar_tgt, g.n_max); e = np.isfinite(g.nstar_tgt).values
        put("censoring", f"km_median_nstar_tgt_{k}", km_quantile(t, e), n=len(g), note=f"events {int(e.sum())}",
            source=src)

    # ---------- operational prevalence (threshold fixed on source validation at 1 % FPR, n = 1024 or largest)
    for k, g in f.groupby("kind"):
        put("operational", f"median_target_fpr_at_source_1pct_{k}", float(g.fpr_cal_op.median()),
            float(g.fpr_cal_op.quantile(0.25)), float(g.fpr_cal_op.quantile(0.75)), n=len(g),
            note="ci columns = IQR", source=src)
        put("operational", f"share_target_fpr_gt_2pct_{k}", float((g.fpr_cal_op > 0.02).mean()), n=len(g), source=src)
        put("operational", f"median_target_tpr_at_source_1pct_{k}", float(g.tpr_cal_op.median()),
            float(g.tpr_cal_op.quantile(0.25)), float(g.tpr_cal_op.quantile(0.75)), n=len(g),
            note="ci columns = IQR", source=src)
        for pi in (0.001, 0.01, 0.1):
            ppv = g.tpr_cal_op * pi / (g.tpr_cal_op * pi + g.fpr_cal_op * (1 - pi))
            put("operational", f"median_ppv_pi{pi}_{k}", float(ppv.median()), float(ppv.quantile(0.25)),
                float(ppv.quantile(0.75)), n=len(g), note="ci columns = IQR", source=src)
        put("operational", f"median_auc_tgt_largest_{k}", float(g.auc_tgt_max.median()), n=len(g), source=src)
        hi_be = g[g.e_tgt_max > 0.05]
        put("operational", f"high_error_but_auc_ge_0.95_{k}", int((hi_be.auc_tgt_max >= 0.95).sum()), n=len(hi_be),
            note="target balanced error > 5 % at the largest size while ranking AUC >= 0.95", source=src)

    # ---------- correction-factor baseline (Mahmood et al. 2022): kappa from ID forecasts of other datasets
    cf = []
    for d in sorted(f.dataset.unique()):
        idm = f[(f.kind == "id") & (f.dataset != d) & f.category.eq("measurable")]
        kappa = float(np.quantile(idm.nstar_tgt / idm.fc_M2, 0.9)) if len(idm) else np.nan
        g = f[f.dataset == d].copy(); g["fc_kappa"] = g.fc_M2 * kappa; g["kappa"] = kappa
        cf.append(g)
    cf = classify(pd.concat(cf), "fc_kappa")
    cf.to_csv(os.path.join(STAT, f"{PFX}forecasts_correction_factor.csv"), index=False)
    for k, g in cf.groupby("kind"):
        gv = g[g.verifiable]
        put("baseline_kappa", f"unsafe_rate_verifiable_{k}", float(gv.unsafe.mean()) if len(gv) else np.nan, n=len(gv),
            note=f"kappa range {g.kappa.min():.2f}-{g.kappa.max():.2f}",
            source=f"09_Statistical_Analysis/shift/{PFX}forecasts_correction_factor.csv")
        gm = g[g.category == "measurable"]
        put("baseline_kappa", f"median_log2_fc_over_obs_{k}", float(np.median(gm.log2_ratio)) if len(gm) else np.nan,
            n=len(gm), source=f"09_Statistical_Analysis/shift/{PFX}forecasts_correction_factor.csv")
    # pure power law (M1) as alternative forecaster
    m1 = classify(f, "fc_M1")
    for k, g in m1.groupby("kind"):
        gm = g[g.category == "measurable"]
        put("baseline_M1", f"within2_measurable_{k}", float(gm.within2.mean()) if len(gm) else np.nan, n=len(gm),
            source=src)
        gv = g[g.verifiable]
        put("baseline_M1", f"unsafe_rate_verifiable_{k}", float(gv.unsafe.mean()) if len(gv) else np.nan, n=len(gv),
            source=src)

    # ---------- per-dataset breakdown (descriptive)
    for (k, d), g in f.groupby(["kind", "dataset"]):
        gv = g[g.verifiable]
        put("by_dataset", f"unsafe_rate_{k}_{d}", float(gv.unsafe.mean()) if len(gv) else np.nan, n=len(gv), source=src)
        put("by_dataset", f"target_never_{k}_{d}", float((~np.isfinite(g.nstar_tgt)).mean()), n=len(g), source=src)

    # ---------- per-learner breakdown
    for (k, L), g in f.groupby(["kind", "learner"]):
        gv = g[g.verifiable]
        put("by_learner", f"unsafe_rate_{k}_{L}", float(gv.unsafe.mean()) if len(gv) else np.nan, n=len(gv), source=src)


def run():
    global PFX
    os.makedirs(STAT, exist_ok=True)
    c, s = load()
    f = forecasts(c)
    sm = s.groupby("sid").mean(numeric_only=True).reset_index()
    sm.to_csv(os.path.join(STAT, "signals_mean.csv"), index=False)
    f = f.merge(sm[["sid"] + [x for x in SIGNALS if x in sm]], on="sid", how="left")
    f["features"] = np.where(f.sid.str.endswith("__noart"), "noart", "all")
    art = list(S.get("artefact_features", {}))
    prim = (f.dataset.isin(art) & (f.features == "noart")) | (~f.dataset.isin(art) & (f.features == "all"))
    f["primary"] = prim
    f.to_csv(os.path.join(STAT, "forecasts.csv"), index=False)
    src = "09_Statistical_Analysis/shift/forecasts.csv"
    noart_sids = set(f.sid[f.features == "noart"])
    s_prim = s[s.sid.isin(set(f.sid[prim]))]
    s_all = s[~s.sid.isin(noart_sids)]
    sm_prim = sm[sm.sid.isin(set(f.sid[prim]))]
    sm_all = sm[~sm.sid.isin(noart_sids)]

    # ---------- scenario inventory (primary set)
    fp = f[prim].copy()
    for k, g in fp.groupby("kind"):
        put("design", f"scenarios_{k}", g.sid.nunique(), n=len(g), note="forecasts = scenarios x learners", source=src)
    put("design", "forecasts_total", len(fp), source=src)
    put("design", "learner_fits_all_runs", int(len(c)), source="06_Experiments/shift/curves")
    put("design", "cpu_hours_all_runs", float(c.sec.sum() / 3600), source="06_Experiments/shift/curves")

    PFX = ""
    core(fp, s_prim, sm_prim, src)
    if (f.features == "noart").any():          # sensitivity: CIC-IoT2023 with all features (artefacts kept)
        PFX = "sens_allfeat_"
        core(f[f.features == "all"].copy(), s_all, sm_all, src)
        PFX = ""
        a = f[(f.dataset.isin(art)) & (f.kind == "id")].copy()
        a["base"] = a.sid.str.replace("__noart", "", regex=False)
        w = a.pivot_table(index=["base", "learner"], columns="features", values="nstar_tgt", aggfunc="first")
        w.to_csv(os.path.join(STAT, "artefact_effect_id_nstar.csv"))
        for L_, g in w.reset_index().groupby("learner"):
            if "noart" in g and "all" in g:
                put("artefact", f"ciciot_id_reached_all_vs_noart_{L_}", int(np.isfinite(g["all"]).sum()),
                    n=int(np.isfinite(g["noart"]).sum()), note="value = tasks reaching 1% with artefacts; n = without",
                    source="09_Statistical_Analysis/shift/artefact_effect_id_nstar.csv")
                ok = np.isfinite(g["all"]) & np.isfinite(g["noart"])
                if ok.any():
                    med = float(np.median(np.log2(g["noart"][ok] / g["all"][ok])))
                    put("artefact", f"ciciot_id_median_log2_noart_over_all_{L_}", med, n=int(ok.sum()),
                        source="09_Statistical_Analysis/shift/artefact_effect_id_nstar.csv")
                    put("artefact", f"ciciot_id_median_factor_noart_over_all_{L_}", float(2 ** med), n=int(ok.sum()),
                        source="09_Statistical_Analysis/shift/artefact_effect_id_nstar.csv")

    # ---------- floors (F6), if available
    fl = os.path.join(scenarios.SHIFT_DIR, "floors")
    if os.path.exists(os.path.join(fl, "more_data.csv")):
        md = pd.read_csv(os.path.join(fl, "more_data.csv"))
        for (fam, L, n), g in md.groupby(["family", "learner", "n"]):
            put("F6", f"more_data_be_{fam}_{L}_n{n}", float(g.be.mean()), float(g.be.min()), float(g.be.max()),
                n=len(g), note="ci columns = min, max over repeats", source="06_Experiments/shift/floors/more_data.csv")
    if os.path.exists(os.path.join(fl, "capacity.csv")):
        cp = pd.read_csv(os.path.join(fl, "capacity.csv"))
        for (fam, L), g in cp.groupby(["family", "learner"]):
            put("F6", f"capacity_default_{fam}_{L}", float(g.be_default.mean()), n=len(g),
                source="06_Experiments/shift/floors/capacity.csv")
            put("F6", f"capacity_tuned_{fam}_{L}", float(g.be_tuned.mean()), n=len(g),
                note="; ".join(sorted(set(g.best))), source="06_Experiments/shift/floors/capacity.csv")
    if os.path.exists(os.path.join(fl, "subtypes.csv")):
        st = pd.read_csv(os.path.join(fl, "subtypes.csv"))
        agg = st.groupby(["family", "learner", "subtype"]).agg(miss=("miss_rate", "mean"), misses=("misses", "sum"),
                                                               total=("total_misses", "sum")).reset_index()
        tot = st.groupby(["family", "learner", "rep"]).total_misses.first().groupby(["family", "learner"]).sum()
        agg["share_of_misses"] = agg.misses / agg.apply(lambda r: tot[(r.family, r.learner)], axis=1)
        agg.to_csv(os.path.join(STAT, "floors_subtypes.csv"), index=False)
        for r in agg.itertuples():
            put("F6", f"subtype_miss_{r.family}_{r.learner}_{r.subtype}", float(r.miss), n=int(r.misses),
                note=f"share of family misses {r.share_of_misses:.3f}", source="09_Statistical_Analysis/shift/floors_subtypes.csv")

    # ---------- computing share of the perceptron in the original main runs (reason for omitting it)
    oc = pd.concat([pd.read_csv(p) for p in
                    glob.glob(os.path.join(config.RAW_RESULTS_DIR, "curves", "*__main__random.csv"))])
    put("design", "mlp_share_cpu_original_main", float(oc[oc.learner == "MLP"].sec.sum() / oc.sec.sum()),
        n=len(oc), source="06_Experiments/raw/curves/*__main__random.csv")

    # ---------- re-audit of the original results (06_Experiments/audit)
    au = os.path.join(config.PROJECT_ROOT, "06_Experiments", "audit", "audit_summary.json")
    if os.path.exists(au):
        a = json.load(open(au))
        put("reaudit", "H2_share_le_1024", a["A4"]["share_le_1024"], *a["A4"]["cluster_bootstrap_95ci"], n=24,
            note="dataset-then-task cluster bootstrap", source="06_Experiments/audit/audit_summary.json")
        for r in a["A1"]["categories"]:
            put("reaudit", f"H6_category_{r['category']}", r["n"], n=144, note=f"counted correct: {r['correct']}",
                source="06_Experiments/audit/A1_forecast_categories.csv")
        put("reaudit", "H6_within2_measurable_only", a["A1"]["measurable_share_within2"], n=a["A1"]["measurable_n"],
            source="06_Experiments/audit/audit_summary.json")
        for r in a["A3"]:
            put("reaudit", f"km_median_nstar_{r['learner']}", r["km_median"], r["km_q25"], r["km_q75"], n=r["n"],
                note=f"reached {r['reached']}; ci columns = KM quartiles", source="06_Experiments/audit/A3_km_nstar.csv")
    pd.DataFrame(rows).to_csv(RES, index=False)
    print(f"{len(rows)} result rows -> {RES}")


def km_quantile(times, events, q=0.5):
    t = np.asarray(times, float); e = np.asarray(events, bool)
    o = np.argsort(t); t, e = t[o], e[o]
    surv, n_at = 1.0, len(t)
    for ti in np.unique(t):
        d = int(((t == ti) & e).sum()); cnt = int((t == ti).sum())
        if d:
            surv *= 1 - d / n_at
            if surv <= 1 - q:
                return float(ti)
        n_at -= cnt
    return np.inf
