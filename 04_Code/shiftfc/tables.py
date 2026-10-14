"""LaTeX tables of the revised manuscript, generated from result files only (no hand-typed numbers).
Written to 12_Tables/ with the prefix 'rev_' and copied to 14_Manuscript/tables/."""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

from lcurve import config, tasks

from . import analysis, scenarios

DS_ORDER = ["nslkdd", "unsw", "cicids", "ciciot"]
DS_NAME = {"nslkdd": "NSL-KDD", "unsw": "UNSW-NB15", "cicids": "CICIDS2017", "ciciot": "CIC-IoT2023"}
OUT = [config.TABLE_DIR, os.path.join(config.PROJECT_ROOT, "14_Manuscript", "tables"),
       os.path.join(config.PROJECT_ROOT, "15_Supplementary")]


def _w(name, body):
    for d in OUT:
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, name), "w", encoding="utf-8").write(body)


def pct(v, d=0):
    return "--" if v is None or not np.isfinite(v) else f"{100 * v:.{d}f}"


def table1():
    inv = tasks.inventory()
    m = inv[(inv.variant == "main") & (inv.family != "Benign")]
    b = inv[(inv.variant == "main") & (inv.family == "Benign")].set_index("dataset")
    sc = pd.read_csv(os.path.join(scenarios.SHIFT_DIR, "scenarios.csv"), keep_default_na=False)
    sc = sc[~sc.sid.str.endswith("__noart")]
    art = scenarios.S.get("artefact_features", {})
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r" & & & & \multicolumn{4}{c}{Scenarios} \\ \cmidrule(lr){5-8}",
         r"Dataset & Features & Families & Unique attack flows & In-distr. & Temporal & Benign change & Held-out variant \\",
         r"\midrule"]
    for ds in DS_ORDER:
        g = m[m.dataset == ds]
        nf = len(scenarios.feature_names(ds)) - len(art.get(ds, []))
        cnt = sc[sc.dataset == ds].kind.value_counts()
        feat = f"{nf}" + (r"$^{a}$" if ds in art else "")
        L.append(f"{DS_NAME[ds]} & {feat} & {len(g)} & {int(g.n_used.min()):,}--{int(g.n_used.max()):,} & "
                 + " & ".join(str(int(cnt.get(k, 0))) for k in ("id", "temporal", "benign", "lovo")) + r" \\")
    tot = sc.kind.value_counts()
    L += [r"\midrule", f"Total & & {len(m)} & & " + " & ".join(str(int(tot.get(k, 0))) for k in
                                                                ("id", "temporal", "benign", "lovo")) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    _w("rev_table1_tasks.tex", "\n".join(L) + "\n")


AUDIT = os.path.join(config.PROJECT_ROOT, "06_Experiments", "audit")
FAM = {"BruteForce": "Brute force", "WebAttack": "Web attack", "PortScan": "Port scan", "Recon": "Reconnaissance",
       "Web": "Web", "Botnet": "Botnet"}


def _fam(f):
    return FAM.get(f, f)


def _order(df):
    df = df.copy()
    df["_o"] = df.dataset.map({d: i for i, d in enumerate(DS_ORDER)})
    return df.sort_values(["_o", "family"]).drop(columns="_o")


def supp_single_feature():
    d = _order(pd.read_csv(os.path.join(AUDIT, "A6_single_feature.csv")))
    L = [r"\begin{tabular}{llllll}", r"\toprule",
         r"Dataset & Family & Best single feature & BE (\%) & Second feature & BE (\%) \\", r"\midrule"]
    for r in d.itertuples():
        L.append(f"{DS_NAME[r.dataset]} & {_fam(r.family)} & \\texttt{{{r.best_feature.replace('_', chr(92) + '_')}}} & "
                 f"{100 * r.best_be:.2f} & \\texttt{{{r.second_feature.replace('_', chr(92) + '_')}}} & {100 * r.second_be:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_single_feature.tex", "\n".join(L) + "\n")


def supp_near_dup():
    d = pd.read_csv(os.path.join(AUDIT, "A2_near_duplicates.csv"))
    a = _order(d[d.cls == "attack"]); b = d[d.cls == "benign"].set_index(["dataset", "family"])
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{Attack test flows} & \multicolumn{2}{c}{Benign test flows} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Dataset & Family & near-duplicate (\%) & same-class neighbour (\%) & near-duplicate (\%) & same-class neighbour (\%) \\",
         r"\midrule"]
    for r in a.itertuples():
        rb = b.loc[(r.dataset, r.family)]
        L.append(f"{DS_NAME[r.dataset]} & {_fam(r.family)} & {100 * r.share_dist_lt_1e3:.1f} & {100 * r.share_nn_same_class:.1f} & "
                 f"{100 * rb.share_dist_lt_1e3:.1f} & {100 * rb.share_nn_same_class:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_near_duplicates.tex", "\n".join(L) + "\n")


def supp_port():
    d = pd.read_csv(os.path.join(AUDIT, "A5_port_shortcut.csv"))
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Earlier $\rightarrow$ later (\%)} & \multicolumn{2}{c}{Random re-split (\%)} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Family & with port & without port & with port & without port \\", r"\midrule"]
    for fam in ("BruteForce", "DoS", "WebAttack", "Botnet", "PortScan", "DDoS"):
        c = d[(d.family == fam) & (d.order == "chrono")].iloc[0]; r = d[(d.family == fam) & (d.order == "rcontrol")].iloc[0]
        L.append(f"{_fam(fam)} & {100 * c.with_port:.2f} & {100 * c.without_port:.2f} & {100 * r.with_port:.2f} & "
                 f"{100 * r.without_port:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_port.tex", "\n".join(L) + "\n")


def supp_original_forecast_km():
    c = pd.read_csv(os.path.join(AUDIT, "A1_forecast_categories.csv")).set_index("category")
    k = pd.read_csv(os.path.join(AUDIT, "A3_km_nstar.csv"))
    name = {"measurable": "forecast and observation within the data",
            "both_beyond_data": "both beyond the data",
            "observed_finite_forecast_beyond": "observed, forecast beyond the data",
            "observed_censored_forecast_finite": "forecast within the data, never observed"}
    L = [r"\begin{tabular}{lrr}", r"\toprule", r"Category (saturating power law, $n\le128$) & Forecasts & Counted correct \\",
         r"\midrule"]
    for key in ("measurable", "both_beyond_data", "observed_finite_forecast_beyond", "observed_censored_forecast_finite"):
        if key in c.index:
            L.append(f"{name[key]} & {int(c.loc[key, 'n'])} & {int(c.loc[key, 'correct'])} \\\\")
    L += [r"\midrule", r"Learner & \multicolumn{2}{r}{Kaplan--Meier median $n^*(1\%)$ [quartiles]; reached} \\", r"\midrule"]
    lname = {"kNN1": "1-NN", "kNN5": "5-NN"}
    for r in k.itertuples():
        q75 = "not reached" if not np.isfinite(r.km_q75) else f"{r.km_q75:,.0f}"
        L.append(f"{lname.get(r.learner, r.learner)} & \\multicolumn{{2}}{{r}}{{{r.km_median:,.0f} [{r.km_q25:,.0f}--{q75}]; "
                 f"{r.reached}/{r.n}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_original_forecast_km.tex", "\n".join(L) + "\n")


def run():
    table1()
    for fn in (supp_single_feature, supp_near_dup, supp_port, supp_original_forecast_km):
        fn()
    if os.path.exists(os.path.join(analysis.STAT, "forecasts.csv")):
        for fn in (table2, supp_followup_by_learner, supp_floors, supp_robustness):
            try:
                fn()
            except Exception as ex:
                print(fn.__name__, "FAILED", repr(ex)[:300])
    print("tables written")


def table2():
    """Summary of pre-specified H1-H6 (first part) and follow-up F1-F6, every value read from the results files."""
    mr = pd.read_csv(os.path.join(config.PROJECT_ROOT, "13_Results", "Master_Results.csv"))
    sr = pd.read_csv(analysis.RES)
    sr = sr[~sr.section.str.startswith("sens_")]

    def m(an, metric, learner="all"):
        r = mr[(mr.analysis == an) & (mr.metric == metric) & (mr.learner == learner)]
        return r.iloc[0]

    def s(sec, metric):
        return sr[(sr.section == sec) & (sr.metric == metric)].iloc[0]

    def p(v, d=0):
        return f"{100 * v:.{d}f}\\%"

    def pv(note):
        return note.split("=")[-1].strip()

    RHO = "$\\rho$"

    def sci(v):
        mant, ex = f"{v:.1e}".split("e")
        return f"${mant}\\times10^{{{int(ex)}}}$"

    h1 = m("H1", "share_M2_best"); h2 = m("H2", "share_tasks_nstar_tree_le_1024"); h2p = m("H2", "binom_p_vs_half")
    h3a = m("H3a", "share_floor_within_CoverHart"); h3b = m("H3b", "spearman_floor_vs_overlap")
    ra = s("reaudit", "H2_share_le_1024")
    h6 = s("reaudit", "H6_within2_measurable_only")
    w = {k: s("F1", f"within2_measurable_{k}") for k in ("id", "temporal", "benign", "lovo")}
    d = {k: s("F1", f"diff_within2_{k}_minus_id_same_families") for k in ("temporal", "benign", "lovo")}
    fa = {k: (s("F2", f"failures_underestimate_{k}"), s("F2", f"failures_{k}")) for k in ("temporal", "lovo")}
    au = {k: s("F3", f"auroc_unsafe_{k}") for k in ("ks_share", "pilot_width")}
    lo = {k: (s("F4", f"lodo_coverage_{k}"), s("F4", f"lodo_unsafe_accepted_{k}")) for k in ("ks_share", "c2st_adj")}
    f5 = (s("F5", "c2st_raw_prev0.05_median"), s("F5", "c2st_adj_prev0.05_median"), s("F5", "wilcoxon_raw_vs_adj_prev0.05_P"))
    md = sr[(sr.section == "F6") & sr.metric.str.startswith("more_data_be_") & sr.metric.str.endswith("n32768")]
    ct = sr[(sr.section == "F6") & sr.metric.str.startswith("capacity_tuned_")]
    h4a, h4b = m("H4a", "spearman_ID_attack_vs_exponent"), m("H4b", "spearman_ID_attack_vs_log_nstar")
    h4p = m("H4a", "P_holm").value
    w5 = {L: (m("H5", "tasks_chrono_worse", L).value, m("H5", "wilcoxon_P_holm", L).value) for L in ("HGB", "RF")}
    n5 = int((mr[(mr.analysis == "H5") & (mr.metric == "be_chrono")]).shape[0])
    rows = [
        ("H1", "Saturating power law fits best", f"{p(h1.value)} of {int(h1.n)} curves ({p(h1.ci_low)}--{p(h1.ci_high)})", "not supported"),
        ("H2", r"Trees reach 1\% with $\le$1,024 flows per class in most tasks",
         f"{int(round(h2.value * h2.n))}/{int(h2.n)} tasks; $P$={h2p.value:.2g}; dependence-aware {p(ra.ci_lo)}--{p(ra.ci_hi)}", "supported"),
        ("H3a", "Floors within nearest-neighbour Bayes interval", f"{int(round(h3a.value * h3a.n))}/{int(h3a.n)} tasks", "not supported"),
        ("H3b", "Floors increase with class overlap", f"$\\rho$={h3b.value:.2f}, $P$={pv(str(h3b.note))}", "not supported"),
        ("H4", "Learning slows with intrinsic dimension",
         (RHO + f"={h4a.value:.2f} (exponent), {h4b.value:.2f} ($n^*$); Holm $P$={h4p:.2f}").replace("=-", "=$-$"), "not supported"),
        ("H5", "Earlier-to-later split raises error",
         f"gradient boosting {int(w5['HGB'][0])}/{n5} (Holm $P$={w5['HGB'][1]:.3f}); random forest {int(w5['RF'][0])}/{n5} "
         f"($P$={w5['RF'][1]:.3f})", "partly"),
        ("H6", r"Pilot forecasts within a factor of 2", f"{p(h6.value, 1)} of {int(h6.n)} measurable (shared test set)", "supported, optimistic"),
        ("F1", "Accuracy falls under temporal and variant shift",
         f"in-distr. {p(w['id'].value, 1)}; temporal {p(w['temporal'].value, 1)} ($P$={float(pv(str(d['temporal'].note))):.2f}); "
         f"variant {p(w['lovo'].value, 1)} ($P\\le0.001$); benign {p(w['benign'].value, 1)}", "variant only"),
        ("F2", "Failures under shift are mostly under-estimates",
         f"temporal {int(fa['temporal'][0].value)}/{int(fa['temporal'][1].value)}; variant {int(fa['lovo'][0].value)}/{int(fa['lovo'][1].value)}", "supported"),
        ("F3", "Label-free scores rank unsafe forecasts",
         f"AUROC {au['pilot_width'].value:.2f} (no target data) to {au['ks_share'].value:.2f} (KS share)", "supported"),
        ("F4", "Abstention thresholds transfer across benchmarks",
         f"KS {p(lo['ks_share'][1].value, 1)} unsafe at {p(lo['ks_share'][0].value, 0)} coverage; adj.\\ C2ST "
         f"{p(lo['c2st_adj'][1].value, 1)} at {p(lo['c2st_adj'][0].value, 0)}", "partly"),
        ("F5", "Adjusted C2ST insensitive to prevalence change",
         f"median {f5[1].value:.2f} vs {f5[0].value:.2f} unadjusted; $P$=" + sci(f5[2].value), "supported"),
        ("F6", "Residual errors persist with more data and tuning (exploratory)",
         f"{p(md.value.min(), 1)}--{p(md.value.max(), 1)} at 32,768 flows; tuned {p(ct.value.min(), 1)}--{p(ct.value.max(), 1)}", "supported"),
    ]
    L = [r"\begin{tabular}{p{0.75cm}>{\raggedright}p{4.3cm}>{\raggedright}p{6.15cm}>{\raggedright\arraybackslash}p{2.5cm}}", r"\toprule",
         r"& Hypothesis & Result & Verdict \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i == 7:
            L.append(r"\midrule \multicolumn{4}{l}{\textit{Follow-up study (specified before it was run; not pre-specified with H1--H6)}} \\ \midrule")
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("rev_table2_hypotheses.tex", "\n".join(L) + "\n")


KINDS = [("id", "In-distribution"), ("temporal", "Temporal"), ("benign", "Benign change"), ("lovo", "Held-out variant")]
SIGS = [("ks_share", "KS share"), ("conf_drop", "confidence drop"), ("disagree_delta", "disagreement change"),
        ("c2st_auc", "C2ST"), ("c2st_adj", "C2ST, label-shift adjusted"), ("pilot_width", "pilot width (no target data)")]


def _sr():
    return pd.read_csv(analysis.RES)


def _g(sr, sec, metric):
    r = sr[(sr.section == sec) & (sr.metric == metric)]
    return r.iloc[0] if len(r) else None


def _pc(r, d=1, ci=True):
    if r is None or not np.isfinite(r.value):
        return "--"
    s = f"{100 * r.value:.{d}f}"
    if ci and np.isfinite(r.ci_lo):
        s += f" ({100 * r.ci_lo:.{d}f}--{100 * r.ci_hi:.{d}f})"
    return s


def supp_followup_by_learner():
    """Follow-up outcomes per scenario kind: primary analysis, sensitivity with all features, comparators,
    and unsafe rates per classifier and per dataset."""
    sr = _sr()
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Kind & Forecasts & Never reached & False assurance & Within $\times2$ (\%) & Unsafe (\%) & Unsafe, all features (\%) & Unsafe, M1 / $\kappa$ (\%) \\",
         r"\midrule"]
    for k, name in KINDS:
        n = _g(sr, "categories", f"{k}_measurable")
        L.append(f"{name} & {int(n.n)} & {int(_g(sr, 'categories', f'{k}_target_never_reaches').value)} & "
                 f"{int(_g(sr, 'categories', f'{k}_false_assurance').value)} & {_pc(_g(sr, 'F1', f'within2_measurable_{k}'))} & "
                 f"{_pc(_g(sr, 'F2', f'unsafe_rate_verifiable_{k}'))} & "
                 f"{_pc(_g(sr, 'sens_allfeat_F2', f'unsafe_rate_verifiable_{k}'), ci=False)} & "
                 f"{_pc(_g(sr, 'baseline_M1', f'unsafe_rate_verifiable_{k}'), ci=False)} / "
                 f"{_pc(_g(sr, 'baseline_kappa', f'unsafe_rate_verifiable_{k}'), ci=False)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_followup_outcomes.tex", "\n".join(L) + "\n")
    learners = ["LogReg", "kNN1", "kNN5", "RF", "HGB"]
    lname = {"LogReg": "LogReg", "kNN1": "1-NN", "kNN5": "5-NN", "RF": "RF", "HGB": "HGB"}
    dss = ["nslkdd", "unsw", "cicids", "ciciot"]
    L = [r"\begin{tabular}{l" + "r" * (len(learners) + len(dss)) + "}", r"\toprule",
         r" & \multicolumn{5}{c}{Classifier} & \multicolumn{4}{c}{Dataset} \\ \cmidrule(lr){2-6}\cmidrule(lr){7-10}",
         "Kind & " + " & ".join(lname[x] for x in learners) + " & " + " & ".join(DS_NAME[d] for d in dss) + r" \\",
         r"\midrule"]
    for k, name in KINDS:
        cells = []
        for x in learners:
            r = _g(sr, "by_learner", f"unsafe_rate_{k}_{x}")
            cells.append("--" if r is None or not np.isfinite(r.value) else f"{100 * r.value:.0f} ({int(r.n)})")
        for d in dss:
            r = _g(sr, "by_dataset", f"unsafe_rate_{k}_{d}")
            cells.append("--" if r is None or not np.isfinite(r.value) else f"{100 * r.value:.0f} ({int(r.n)})")
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_unsafe_breakdown.tex", "\n".join(L) + "\n")
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Score & AUROC, all & AUROC, shifted only & AURC & Unsafe at 50\% cov. & LODO coverage & LODO unsafe \\",
         r"\midrule"]
    for s_, name in SIGS:
        a = _g(sr, "F3", f"auroc_unsafe_{s_}"); b = _g(sr, "F3", f"auroc_unsafe_shift_only_{s_}")
        c = _g(sr, "F3", f"aurc_{s_}"); e = _g(sr, "F3", f"selective_unsafe_cov50_{s_}")
        lc, lu = _g(sr, "F4", f"lodo_coverage_{s_}"), _g(sr, "F4", f"lodo_unsafe_accepted_{s_}")
        L.append(f"{name} & {a.value:.2f} ({a.ci_lo:.2f}--{a.ci_hi:.2f}) & {b.value:.2f} ({b.ci_lo:.2f}--{b.ci_hi:.2f}) & "
                 f"{c.value:.3f} & {100 * e.value:.1f} & {100 * lc.value:.1f} & {100 * lu.value:.1f} \\\\")
    u = _g(sr, "F3", "unsafe_overall")
    L += [r"\midrule", f"No abstention & 0.50 & 0.50 & {u.value:.3f} & {100 * u.value:.1f} & 100.0 & {100 * u.value:.1f} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    _w("supp_signals.tex", "\n".join(L) + "\n")
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Score & no change & attack share 0.05 & attack share 0.6 \\", r"\midrule"]
    for s_, name in SIGS[3:5] + SIGS[:1]:
        vals = [_g(sr, "F5", f"lodo_false_alarm_{s_}_{lab}") for lab in ("no_change", "prev0.05", "prev0.6")]
        L.append(name + " & " + " & ".join(f"{100 * v.value:.0f}" for v in vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_prevalence_alarms.tex", "\n".join(L) + "\n")


def supp_robustness():
    path = os.path.join(config.PROJECT_ROOT, "13_Results", "Shift_Robustness.csv")
    if not os.path.exists(path):
        return
    r = pd.read_csv(path)
    names = {"eps0.01_pilot128_tol2": r"primary: 1\%, $n\le128$, $\times2$", "eps0.05_pilot128_tol2": r"target 5\%",
             "eps0.01_pilot64_tol2": r"pilot $n\le64$", "eps0.01_pilot256_tol2": r"pilot $n\le256$",
             "eps0.01_pilot128_tol1.5": r"tolerance $\times1.5$", "eps0.01_pilot128_tol3": r"tolerance $\times3$"}
    L = [r"\begin{tabular}{l" + "rr" * 4 + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for _, n in KINDS) + r" \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         "Setting & " + " & ".join(["within (\\%) & unsafe (\\%)"] * 4) + r" \\", r"\midrule"]
    for key, name in names.items():
        cells = []
        for k, _ in KINDS:
            x = r[(r.setting == key) & (r.kind == k)].iloc[0]
            cells += [f"{100 * x.within_share:.1f}", f"{100 * x.unsafe_share:.1f}"]
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_robustness.tex", "\n".join(L) + "\n")


def supp_floors():
    sr = _sr()
    fams = [("Recon", "Reconnaissance"), ("Spoofing", "Spoofing"), ("Web", "Web"), ("BruteForce", "Brute force")]
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Family & Classifier & Default (\%) & Tuned (\%) & 16,384 flows, all rows (\%) & 32,768 flows, all rows (\%) \\",
         r"\midrule"]
    for f_, name in fams:
        for c_ in ("RF", "HGB"):
            d = _g(sr, "F6", f"capacity_default_{f_}_{c_}"); t = _g(sr, "F6", f"capacity_tuned_{f_}_{c_}")
            m16 = _g(sr, "F6", f"more_data_be_{f_}_{c_}_n16384"); m32 = _g(sr, "F6", f"more_data_be_{f_}_{c_}_n32768")
            L.append(f"{name} & {c_} & {100 * d.value:.2f} & {100 * t.value:.2f} & "
                     f"{'--' if m16 is None else f'{100 * m16.value:.2f}'} & {'--' if m32 is None else f'{100 * m32.value:.2f}'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _w("supp_floors.tex", "\n".join(L) + "\n")
