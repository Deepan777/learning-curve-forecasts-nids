"""LaTeX tables generated from the result CSVs (no hand-typed numbers)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config, tasks
from .analysis import DS_NAME, ROB_DIR, STAT_DIR
from .geometry import GEOM_CSV

TABLE_DIR = config.TABLE_DIR
SUPP_DIR = os.path.join(config.PROJECT_ROOT, "15_Supplementary")
DS_ORDER = ["nslkdd", "unsw", "cicids", "ciciot"]


def _w(name, body, supp=False):
    for d in ([TABLE_DIR] + ([SUPP_DIR] if supp else [])):
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, name), "w", encoding="utf-8").write(body)


def fmt_n(v, nmax=None):
    if not np.isfinite(v):
        return r"$>$" + (f"{int(nmax):,}" if nmax else "max")
    if v <= 4:
        return r"$\le$4"
    return f"{int(round(v)):,}"


def pct(v, d=1):
    return "--" if not np.isfinite(v) else f"{100 * v:.{d}f}"


def table1():
    inv = tasks.inventory()
    m = inv[(inv.variant == "main") & (inv.family != "Benign")]
    b = inv[(inv.variant == "main") & (inv.family == "Benign")].set_index("dataset")
    feats = {ds: len(__import__("json").load(open(os.path.join(tasks.TASK_DIR, f"{ds}__features.json")))["features"])
             for ds in DS_ORDER}
    L = [r"\begin{tabular}{llrrrl}", r"\toprule",
         r"Dataset & Features & Benign pool & Families & Unique attack flows & Time order used \\", r"\midrule"]
    timeord = {"nslkdd": "none", "unsw": "session 1 $\\to$ session 2", "cicids": "within-day file order",
               "ciciot": "none"}
    for ds in DS_ORDER:
        g = m[m.dataset == ds]
        L.append(f"{DS_NAME[ds]} & {feats[ds]} & {int(b.loc[ds, 'n_used']):,} & {len(g)} & "
                 f"{int(g.n_used.min()):,}--{int(g.n_used.max()):,} & {timeord[ds]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("table1_tasks.tex", "\n".join(L) + "\n")


def table2(mr):
    """Hypothesis summary."""
    def g(analysis, metric, learner="all", note=None):
        r = mr[(mr.analysis == analysis) & (mr.metric == metric) & (mr.learner == learner)]
        if note is not None:
            r = r[r.note.astype(str).str.startswith(note)]
        return r.iloc[0] if len(r) else None

    rows = []
    r = g("H1", "share_M2_best")
    rows.append(("H1", "Saturating power law describes curves best (AICc)",
                 f"{pct(r.value)}\\% of {int(r.n)} curves [{pct(r.ci_low)}--{pct(r.ci_high)}]"))
    r = g("H2", "share_tasks_nstar_tree_le_1024"); p = g("H2", "binom_p_vs_half")
    rows.append(("H2", "Tree ensembles reach 1\\% error with $\\le$1,024 flows per class",
                 f"{int(round(r.value * r.n))}/{int(r.n)} tasks ({pct(r.value, 0)}\\%); one-sided binomial $P$={p.value:.2g}"))
    r = g("H3a", "share_floor_within_CoverHart")
    rows.append(("H3a", "Asymptote within nearest-neighbour Bayes-error interval",
                 f"{int(round(r.value * r.n))}/{int(r.n)} tasks [{pct(r.ci_low, 0)}--{pct(r.ci_high, 0)}\\%]"))
    r = g("H3b", "spearman_floor_vs_overlap"); p = g("H3b", "spearman_floor_vs_overlap_P")
    rows.append(("H3b", "Asymptote increases with class overlap (all tasks)",
                 f"$\\rho$={r.value:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}], $P$={p.value:.2g}"))
    a = g("H3b", "overlap_median_floor_gt_1pct"); b = g("H3b", "overlap_median_floor_le_1pct")
    p = g("H3b", "mannwhitney_P")
    rows.append(("", "Overlap in tasks with floor $>$1\\% vs the rest",
                 f"median {pct(a.value)}\\% ($n$={int(a.n)}) vs {pct(b.value, 2)}\\% ($n$={int(b.n)}); Mann--Whitney $P$={p.value:.2g}"))
    r = g("H4a", "spearman_ID_attack_vs_exponent"); p = g("H4a", "P_holm")
    rows.append(("H4a", "Exponent decreases with intrinsic dimension",
                 f"$\\rho$={r.value:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}], Holm $P$={p.value:.2g}"))
    r = g("H4b", "spearman_ID_attack_vs_log_nstar"); p = g("H4b", "P_holm")
    rows.append(("H4b", "$n^*$ increases with intrinsic dimension",
                 f"$\\rho$={r.value:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}], Holm $P$={p.value:.2g}"))
    for L in ("RF", "HGB"):
        a = g("H5", "median_be_chrono", L); b = g("H5", "median_be_rcontrol", L); p = g("H5", "wilcoxon_P_holm", L)
        w = g("H5", "tasks_chrono_worse", L)
        rows.append(("H5" if L == "RF" else "", f"Past$\\to$future error exceeds random split ({'random forest' if L == 'RF' else 'gradient boosting'})",
                     f"median {pct(a.value, 2)}\\% vs {pct(b.value, 2)}\\%; worse in {int(w.value)}/{int(w.n)}; Holm $P$={p.value:.2g}"))
    for m in ("M2", "M1", "geometry"):
        r = g("H6", "share_within_factor2", "all", note=m)
        r2 = g("H6", "share_within_factor2_reached", "all", note=m)
        lab = {"M2": "saturating power law", "M1": "pure power law", "geometry": "intrinsic dimension only"}[m]
        rows.append(("H6" if m == "M2" else "", f"Forecast of $n^*$ from $n\\le128$ within factor 2 ({lab})",
                     f"{pct(r.value)}\\% of {int(r.n)} [{pct(r.ci_low)}--{pct(r.ci_high)}]; "
                     f"curves reaching 1\\%: {pct(r2.value)}\\% of {int(r2.n)}"))
    L = [r"\begin{tabular}{llp{6.2cm}}", r"\toprule", r"& Pre-specified hypothesis & Result \\", r"\midrule"]
    for h, d, v in rows:
        L.append(f"{h} & {d} & {v} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("table2_hypotheses.tex", "\n".join(L) + "\n")


def table3(mr):
    """Sensitivity analyses."""
    sv = pd.read_csv(os.path.join(ROB_DIR, "sensitivity_nstar.csv"))
    lab = {"duplicates_kept": "Exact duplicates kept", "unsw_ttl_restored": "UNSW-NB15 TTL features restored",
           "rf500_knn10": "500 trees / $k$=10"}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Variant & Learner & Tasks & $\le$1,024 (main) & $\le$1,024 (variant) & Median $\log_2$ ratio of $n^*$ \\",
         r"\midrule"]
    for a in ("duplicates_kept", "unsw_ttl_restored", "rf500_knn10"):
        for Ln in ("RF", "HGB", "kNN5"):
            g = mr[(mr.analysis == "sensitivity") & (mr.note == a) & (mr.learner == Ln)]
            if g.empty:
                continue
            gm = g.set_index("metric").value
            n = int(g.n.iloc[0])
            L.append(f"{lab[a]} & {Ln} & {n} & {pct(gm['share_le_1024_main'], 0)}\\% & "
                     f"{pct(gm['share_le_1024_var'], 0)}\\% & {gm['median_log2_ratio_nstar_var_over_main']:+.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("table3_sensitivity.tex", "\n".join(L) + "\n")


def supp_tables(mr):
    t2 = pd.read_csv(os.path.join(STAT_DIR, "H2_sample_complexity.csv"))
    t3 = pd.read_csv(os.path.join(STAT_DIR, "H3_floors.csv"))
    geo = pd.read_csv(GEOM_CSV)
    learners = ["LogReg", "kNN1", "kNN5", "RF", "HGB", "MLP"]
    t2 = t2.assign(r=t2.dataset.map({d: i for i, d in enumerate(DS_ORDER)})).sort_values(["r", "family"])
    L = [r"\begin{table}[ht]\centering\small",
         r"\caption{Sample complexity $n^*(1\%)$ (labelled flows per class for 1\% balanced error, mean of 10 repeats) "
         r"for every task and learner, and balanced error at the largest training size of the best learner. $>$: not "
         r"reached at the largest available size.}", r"\label{tab:s_nstar}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{ll" + "r" * len(learners) + "rl}", r"\toprule",
         "Dataset & Family & " + " & ".join(learners) + r" & Best error (\%) & Best \\", r"\midrule"]
    for r in t2.itertuples():
        cells = [fmt_n(getattr(r, f"nstar_{x}"), r.n_max) for x in learners]
        L.append(f"{DS_NAME[r.dataset]} & {r.family} & " + " & ".join(cells) + f" & {pct(r.e_max_best, 2)} & {r.best} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    s2 = "\n".join(L) + "\n"
    t3 = t3.merge(geo.drop(columns=["nn_cross_attack", "nn_cross_benign"]), on=["dataset", "family"])
    t3 = t3.assign(r=t3.dataset.map({d: i for i, d in enumerate(DS_ORDER)})).sort_values(["r", "family"])
    L = [r"\begin{table}[ht]\centering\small",
         r"\caption{Floors and geometry per task. $c$: fitted asymptote of the best learner (95\% bootstrap CI). "
         r"$R_1$: 1-NN balanced error at the largest size; Cover--Hart interval $[\tfrac12(1-\sqrt{1-2R_1}),R_1]$. "
         r"Conflict: exact-conflict floor. ID: TwoNN intrinsic dimension (attack / benign). Overlap: mean share of cross-class nearest neighbours.}",
         r"\label{tab:s_floors}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"Dataset & Family & $c$ (\%) & 95\% CI & $R_1$ (\%) & CH lower (\%) & Conflict (\%) & ID attack & ID benign & Overlap (\%) \\",
         r"\midrule"]
    for r in t3.itertuples():
        L.append(f"{DS_NAME[r.dataset]} & {r.family} & {pct(r.c, 2)} & {pct(r.c_lo, 2)}--{pct(r.c_hi, 2)} & "
                 f"{pct(r.R1, 2)} & {pct(r.CH_lower, 2)} & {pct(r.conflict_floor, 3)} & {r.twonn_attack:.2f} & "
                 f"{r.twonn_benign:.2f} & {pct((r.nn_cross_attack + r.nn_cross_benign) / 2, 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    s3 = "\n".join(L) + "\n"
    h1 = mr[(mr.analysis == "H1") & (mr.metric == "share_M2_best") & (mr.learner != "all")]
    h6 = mr[(mr.analysis == "H6") & (mr.metric == "share_within_factor2") & (mr.learner != "all")]
    L = [r"\begin{table}[ht]\centering\small",
         r"\caption{Per-learner results. Share of curves best described by the saturating power law (AICc), and share "
         r"of $n^*(1\%)$ forecasts from $n\le128$ that fall within a factor of 2 of the observed value.}",
         r"\label{tab:s_learners}", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Learner & M2 best (\%) & Forecast M2 (\%) & Forecast M1 (\%) & Forecast ID (\%) \\", r"\midrule"]
    for Ln in learners:
        a = h1[h1.learner == Ln].value
        f = {m: h6[(h6.learner == Ln) & (h6.note == m)].value for m in ("M2", "M1", "geometry")}
        L.append(f"{Ln} & {pct(a.iloc[0], 0)} & {pct(f['M2'].iloc[0], 0)} & {pct(f['M1'].iloc[0], 0)} & "
                 f"{pct(f['geometry'].iloc[0], 0)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    s4 = "\n".join(L) + "\n"
    _w("supp_tables_results.tex", s2 + "\n" + s3 + "\n" + s4, supp=True)


def stage(P, log=print):
    mr = pd.read_csv(os.path.join(config.RESULTS_DIR, "Master_Results.csv"))
    table1(); table2(mr); table3(mr); supp_tables(mr)
    log("  tables written")
