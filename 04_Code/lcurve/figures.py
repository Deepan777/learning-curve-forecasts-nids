"""Publication figures (PDF + SVG + 300-dpi PNG) and LaTeX tables.

Colour: categorical slots in fixed order (blue, orange, aqua, yellow, magenta,
green, violet, red); text in ink colours; scatter plots use one colour and marker
shapes for datasets (more than three colours are never used in a scatter)."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config
from .analysis import DS_NAME, ROB_DIR, STAT_DIR

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA = SLOTS[0], SLOTS[1], SLOTS[2]
DS_ORDER = ["nslkdd", "unsw", "cicids", "ciciot"]
DS_MARK = {"nslkdd": "o", "unsw": "s", "cicids": "^", "ciciot": "D"}
LEARN_ORDER = ["LogReg", "kNN1", "kNN5", "RF", "HGB", "MLP"]
SIZES_TICKS = [4, 16, 64, 256, 1024, 4096, 16384]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 1.0,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.major.width": 1.0, "ytick.major.width": 1.0, "lines.linewidth": 1.2,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    out = os.path.join(config.FIG_DIR, name)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def _panel(ax, letter):
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", color=INK)


def _logx(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES_TICKS)
    ax.set_xticklabels([f"{t}" if t < 1000 else f"{t // 1024}k" for t in SIZES_TICKS])


def _grid(ax):
    ax.grid(True, which="major", color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure 1
def fig_design():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 7.2); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f0efec", ec=MUTED):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=7.2, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=6.2, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=INK2, lw=1.0))

    w, gap = 4.5, 0.35
    xs = [0.1, 0.1 + w + gap, 0.1 + 2 * (w + gap)]
    box(xs[0], 4.95, w, 2.15, "Four public benchmarks",
        "NSL-KDD, UNSW-NB15,\nCICIDS2017, CIC-IoT2023\n24 attack families with\n" + r"$\geq$" + "200 unique flows")
    box(xs[1], 4.95, w, 2.15, "One task per attack family",
        "family vs benign traffic\nof the same dataset;\nexact duplicates removed,\nconflicting vectors counted")
    box(xs[2], 4.95, w, 2.15, "Learning curves",
        r"$n$ = 4 to 16,384 flows per class" + "\n6 learners " + r"$\times$" + " 10 repeats\nnested training sets\nbalanced error, held-out flows")
    arrow(xs[0] + w, 6.0, xs[1], 6.0); arrow(xs[1] + w, 6.0, xs[2], 6.0)
    box(0.1, 2.85, 14.2, 1.35, "Each curve gives",
        r"sample complexity $n^*(1\%)$,  floor $c$ and exponent $b$ of $e = c + a\,n^{-b}$,"
        + "  model choice by AICc", fc="#cde2fb", ec=BLUE)
    arrow(xs[2] + w / 2, 4.95, xs[2] + w / 2, 4.2)
    box(xs[0], 0.1, w, 2.15, "What sets the floor?",
        "exact label conflicts\nnearest-neighbour\nBayes-error bounds\nclass overlap  (H3)", fc="#cde2fb", ec=BLUE)
    box(xs[1], 0.1, w, 2.15, "What sets the speed?",
        "intrinsic dimension of\nthe attack class  (H4)\npast-to-future vs random\nsplit of the same flows  (H5)",
        fc="#cde2fb", ec=BLUE)
    box(xs[2], 0.1, w, 2.15, "Forecasting the label budget",
        r"fit on $n\leq128$, extrapolate $n^*(1\%)$" + "\nwithin a factor of 2?  (H6)\ngeometry-only forecast\nas comparator",
        fc="#cde2fb", ec=BLUE)
    for x in xs:
        arrow(x + w / 2, 2.85, x + w / 2, 2.25)
    return _save(fig, "fig1_design")


# ---------------------------------------------------------------- Figure 2
def fig_curves():
    df = pd.read_parquet(os.path.join(STAT_DIR, "all_curve_records.parquet"))
    m = df[(df.variant == "main") & (df.order == "random") & (df.alt == 0) & (df.learner == "RF")]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for k, ds in enumerate(DS_ORDER):
        ax = axs.flat[k]; _panel(ax, "abcd"[k]); _grid(ax)
        g = m[m.dataset == ds]
        fams = sorted(g.family.unique())
        ends = []
        for i, f in enumerate(fams):
            c = g[g.family == f].groupby("n").be.agg(["mean", "sem"])
            e0 = 1 / (2 * g[g.family == f].n_test_a.iloc[0])
            y = np.maximum(c["mean"].values, e0)
            ax.plot(c.index, y, color=SLOTS[i % 8], lw=1.3, marker="o", ms=2.5, zorder=3)
            ends.append([c.index[-1], y[-1], f, SLOTS[i % 8]])
        # direct labels in an aligned column right of the data, spaced in log units, with leader lines
        ends.sort(key=lambda t: t[1])
        ly = [np.log10(t[1]) for t in ends]
        for j in range(1, len(ly)):
            ly[j] = max(ly[j], ly[j - 1] + 0.19)
        shift = max(0.0, ly[-1] - np.log10(0.45))          # keep the top label inside the axes
        ly = [v - shift for v in ly]
        for j in range(len(ly) - 2, -1, -1):
            ly[j] = min(ly[j], ly[j + 1] - 0.19)
        xl = 16384 * 2.6
        for (x, y, f, col), yy in zip(ends, ly):
            ax.plot([x * 1.12, xl * 0.92], [y, 10 ** yy], color=col, lw=0.8, alpha=0.55, ls=(0, (1, 1.6)))
            ax.text(xl, 10 ** yy, f, fontsize=6.2, color=INK2, va="center")
        ax.axhline(0.01, color=INK, lw=1.0, ls=(0, (4, 2)))
        ax.set_yscale("log"); ax.set_ylim(1e-4, 0.6)
        _logx(ax); ax.set_xlim(3, 16384 * 22)
        ax.set_title(DS_NAME[ds], loc="left", fontsize=8, color=INK)
        if k % 2 == 0:
            ax.set_ylabel("balanced error (random forest)")
        if k >= 2:
            ax.set_xlabel("labelled flows per class, n")
    axs.flat[0].text(4.2, 0.0115, "1%", fontsize=6.5, color=INK, va="bottom")
    fig.tight_layout()
    return _save(fig, "fig2_curves")


# ---------------------------------------------------------------- Figure 3
def fig_nstar():
    t2 = pd.read_csv(os.path.join(STAT_DIR, "H2_sample_complexity.csv"))
    nmax = 16384
    right = nmax * 2.6
    t2["ds_rank"] = t2.dataset.map({d: i for i, d in enumerate(DS_ORDER)})
    t2["key"] = np.where(np.isfinite(t2.nstar_tree), t2.nstar_tree, 1e12)
    t2 = t2.sort_values(["ds_rank", "key"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    _grid(ax)
    y = np.arange(len(t2))[::-1]
    for yi, r in zip(y, t2.itertuples()):
        vals = np.array([getattr(r, f"nstar_{L}") for L in LEARN_ORDER], float)
        fin = vals[np.isfinite(vals)]
        lo = fin.min() if len(fin) else right
        hi = right if np.any(~np.isfinite(vals)) else fin.max()
        ax.plot([lo, hi], [yi, yi], color=MUTED, lw=3.0, solid_capstyle="round", alpha=0.55, zorder=2)
        v = r.nstar_tree
        if np.isfinite(v):
            ax.plot(v, yi, "o", ms=5.5, color=BLUE, mec="white", mew=1.0, zorder=4)
        else:
            ax.annotate("", xy=(right * 1.05, yi), xytext=(right / 1.6, yi),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4), zorder=4)
        ax.text(2.2, yi, f"{r.family}", fontsize=6.6, color=INK, va="center", ha="right")
    # dataset separators and labels
    for ds in DS_ORDER:
        yy = y[t2.dataset.values == ds]
        if len(yy):
            ax.text(2.2 / 9, yy.mean(), DS_NAME[ds], rotation=90, va="center", ha="center", fontsize=7, color=INK2,
                    fontweight="bold")
            ax.axhline(yy.min() - 0.5, color=GRID, lw=1.0)
    ax.axvline(1024, color=INK, lw=1.0, ls=(0, (4, 2)))
    ax.text(1024 * 1.08, len(t2) - 0.4, "1,024", fontsize=6.5, color=INK, va="bottom")
    ax.axvline(right, color=MUTED, lw=1.0, ls=(0, (1, 2)))
    ax.text(right, -1.4, "not reached\nwith the\navailable data", fontsize=6.0, color=INK2, ha="center", va="top")
    _logx(ax); ax.set_xlim(2.5, right * 1.3); ax.set_ylim(-1.0, len(t2) - 0.3)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("labelled flows per class needed for 1% balanced error, n*(1%)")
    ax.plot([], [], "o", color=BLUE, ms=5.5, label="best tree ensemble (random forest or gradient boosting)")
    ax.plot([], [], color=MUTED, lw=3.0, alpha=0.55, label="range over six learners")
    ax.legend(loc="lower right", fontsize=6.6, bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.subplots_adjust(left=0.2)
    return _save(fig, "fig3_nstar")


# ---------------------------------------------------------------- Figure 4
def fig_floors_geometry():
    t3 = pd.read_csv(os.path.join(STAT_DIR, "H3_floors.csv"))
    t4 = pd.read_csv(os.path.join(STAT_DIR, "H4_geometry.csv"))
    mr = pd.read_csv(os.path.join(config.RESULTS_DIR, "Master_Results.csv"))
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.85, wspace=0.3)
    ax = fig.add_subplot(gs[0, :]); _panel(ax, "a"); _grid(ax)
    t3 = t3.assign(ds_rank=t3.dataset.map({d: i for i, d in enumerate(DS_ORDER)})).sort_values(["ds_rank", "R1"])
    x = np.arange(len(t3))
    fl = 1e-4
    ax.vlines(x, np.maximum(t3.CH_lower, fl), np.maximum(t3.CH_upper, fl), color=MUTED, lw=4, alpha=0.55, zorder=2)
    yerr = [np.maximum(t3.c - np.maximum(t3.c_lo, fl), 0), np.maximum(t3.c_hi - t3.c, 0)]
    ax.errorbar(x, np.maximum(t3.c, fl), yerr=yerr, fmt="o", ms=4, color=BLUE, ecolor=BLUE, elinewidth=1.0, capsize=0,
                zorder=4, mec="white", mew=0.8)
    ax.axhline(0.01, color=INK, lw=1.0, ls=(0, (4, 2)))
    ax.set_yscale("log"); ax.set_ylim(0.8e-4, 0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{f}" for f in t3.family], rotation=60, ha="right", fontsize=6.2)
    for ds in DS_ORDER:
        xx = x[t3.dataset.values == ds]
        if len(xx):
            ax.text(xx.mean(), 1.01, DS_NAME[ds], transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6.8, color=INK2, fontweight="bold")
            ax.axvline(xx.max() + 0.5, color=GRID, lw=1.0)
    ax.set_ylabel("balanced error")
    ax.plot([], [], color=MUTED, lw=4, alpha=0.55, label="nearest-neighbour Bayes-error interval")
    ax.errorbar([], [], yerr=[], fmt="o", color=BLUE, label="fitted asymptote c (95% CI)")
    ax.legend(loc="upper left", fontsize=6.6, ncol=2)

    cap = 2 * 16384
    for k, (col, lab, letter) in enumerate((("b_exp", "fitted exponent b", "b"), ("nstar", "n*(1%), best learner", "c"))):
        ax = fig.add_subplot(gs[1, k]); _panel(ax, letter); _grid(ax)
        yv = t4[col].values.astype(float)
        if col == "nstar":
            cens = ~np.isfinite(yv)
            yv = np.where(cens, cap, yv)
        for ds in DS_ORDER:
            s_ = t4.dataset.values == ds
            ax.scatter(t4.twonn_attack[s_], yv[s_], marker=DS_MARK[ds], s=22, color=BLUE, edgecolor="white", lw=0.6,
                       label=DS_NAME[ds], zorder=3)
        if col == "nstar":
            ax.set_yscale("log", base=2); ax.set_yticks([4, 32, 256, 2048, cap])
            ax.set_yticklabels(["4", "32", "256", "2,048", ">16,384"])
            row = mr[(mr.analysis == "H4b") & (mr.metric == "spearman_ID_attack_vs_log_nstar")].iloc[0]
        else:
            row = mr[(mr.analysis == "H4a") & (mr.metric == "spearman_ID_attack_vs_exponent")].iloc[0]
        m_ = lambda v: f"{v:.2f}".replace("-", "−")
        ax.set_title(f"Spearman ρ = {m_(row.value)} (95% CI {m_(row.ci_low)} to {m_(row.ci_high)})",
                     loc="right", fontsize=6.6, color=INK)
        ax.set_xlabel("intrinsic dimension of the attack class (TwoNN)")
        ax.set_ylabel(lab)
        if k == 0:
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, -0.02), handletextpad=0.2)
    return _save(fig, "fig4_floors_geometry")


# ---------------------------------------------------------------- Figure 5
def fig_time_forecast():
    t5 = pd.read_csv(os.path.join(STAT_DIR, "H5_time.csv"))
    t6 = pd.read_csv(os.path.join(STAT_DIR, "H6_forecast.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
    ax = axs[0]; _panel(ax, "a"); _grid(ax)
    g = t5[t5.learner == "RF"]
    fl = 1e-4
    for r in g.itertuples():
        col = SLOTS[0] if r.dataset == "unsw" else SLOTS[1]
        ax.plot([0, 1], [max(r.be_rcontrol, fl), max(r.be_chrono, fl)], color=col, lw=1.1, marker="o", ms=3, alpha=0.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["random split\n(same flows)", "past → future"])
    ax.set_xlim(-0.3, 1.3); ax.set_yscale("log"); ax.set_ylim(0.8e-4, 0.6)
    ax.axhline(0.01, color=INK, lw=1.0, ls=(0, (4, 2)))
    ax.set_ylabel("balanced error at the largest common n\n(random forest)")
    ax.plot([], [], color=SLOTS[0], label="UNSW-NB15 (session 1 → 2)")
    ax.plot([], [], color=SLOTS[1], label="CICIDS2017 (within day)")
    ax.legend(loc="upper left", fontsize=6.4)
    ax = axs[1]; _panel(ax, "b"); _grid(ax)
    f = t6[t6.model == "M2"].copy()
    cap = 2 ** 17
    o = np.where(np.isfinite(f.observed), f.observed, cap)
    p = np.where(np.isfinite(f.forecast) & (f.forecast <= f.n_max), f.forecast, cap)
    p = np.clip(p, 2, cap)
    xs = np.array([2, cap])
    ax.fill_between(xs, xs / 2, xs * 2, color=BLUE, alpha=0.12, lw=0, label="within a factor of 2")
    ax.plot(xs, xs, color=INK2, lw=1.0)
    ax.scatter(o, p, s=10, color=BLUE, alpha=0.65, edgecolor="none", zorder=3)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ticks = [4, 64, 1024, 16384, cap]
    tl = ["4", "64", "1,024", "16,384", "not\nreached"]
    ax.set_xticks(ticks); ax.set_xticklabels(tl); ax.set_yticks(ticks); ax.set_yticklabels(tl)
    ax.set_xlim(2.5, cap * 1.6); ax.set_ylim(2.5, cap * 1.6)
    ax.set_xlabel("observed n*(1%)"); ax.set_ylabel("forecast from n ≤ 128")
    ax.legend(loc="upper left", fontsize=6.4)
    return _save(fig, "fig5_time_forecast")


# ---------------------------------------------------------------- Supplementary figures
SUPP_FIG_DIR = os.path.join(config.PROJECT_ROOT, "15_Supplementary", "figures")


def _save_supp(fig, name):
    os.makedirs(SUPP_FIG_DIR, exist_ok=True)
    out = os.path.join(SUPP_FIG_DIR, name)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def _task_grid(sel, colour_by, levels, labels, name, ylabel="balanced error", ncol=6):
    """Small multiples: one panel per task; one line per level of `colour_by`."""
    tasks_ = (sel[["dataset", "family"]].drop_duplicates()
              .assign(r=lambda d: d.dataset.map({x: i for i, x in enumerate(DS_ORDER)}))
              .sort_values(["r", "family"]))
    nrow = int(np.ceil(len(tasks_) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(7.2, 1.35 * nrow + 0.4), sharex=True, sharey=True)
    for ax in axs.flat[len(tasks_):]:
        ax.axis("off")
    for ax, (ds, fam) in zip(axs.flat, zip(tasks_.dataset, tasks_.family)):
        g = sel[(sel.dataset == ds) & (sel.family == fam)]
        for i, lv in enumerate(levels):
            c = g[g[colour_by] == lv].groupby("n").be.mean()
            if len(c):
                ax.plot(c.index, np.maximum(c.values, 1e-4), color=SLOTS[i], lw=1.0)
        ax.axhline(0.01, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(1e-4, 0.6)
        ax.set_xticks([4, 64, 1024, 16384]); ax.set_xticklabels(["4", "64", "1k", "16k"], fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_title(f"{DS_NAME[ds]}\n{fam}", fontsize=5.8, color=INK, pad=2)
    for i, (lv, lab) in enumerate(zip(levels, labels)):
        axs.flat[0].plot([], [], color=SLOTS[i], lw=1.2, label=lab)
    fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="lower center", ncol=len(levels), fontsize=6.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.005, 0.5, ylabel, rotation=90, va="center", fontsize=7)
    fig.text(0.5, 0.03, "labelled flows per class, n", ha="center", fontsize=7)
    fig.tight_layout(rect=(0.015, 0.05, 1, 1), h_pad=0.6, w_pad=0.3)
    return _save_supp(fig, name)


def figS_all_learners():
    df = pd.read_parquet(os.path.join(STAT_DIR, "all_curve_records.parquet"))
    m = df[(df.variant == "main") & (df.order == "random") & (df.alt == 0)]
    return _task_grid(m, "learner", LEARN_ORDER, ["logistic regression", "1-NN", "5-NN", "random forest",
                                                   "gradient boosting", "MLP"], "figS1_all_learners")


def figS_duplicates():
    df = pd.read_parquet(os.path.join(STAT_DIR, "all_curve_records.parquet"))
    m = df[(df.order == "random") & (df.alt == 0) & (df.learner == "RF") & df.variant.isin(["main", "dup"])]
    return _task_grid(m, "variant", ["main", "dup"], ["duplicates removed (main)", "duplicates kept"],
                      "figS2_duplicates")


def figS_time():
    df = pd.read_parquet(os.path.join(STAT_DIR, "all_curve_records.parquet"))
    m = df[(df.variant == "main") & (df.alt == 0) & (df.learner == "RF") & df.order.isin(["rcontrol", "chrono"])]
    return _task_grid(m, "order", ["rcontrol", "chrono"], ["random split of the same flows", "past → future"],
                      "figS3_time", ncol=5)


def stage(P, log=print):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    for fn in (fig_design, fig_curves, fig_nstar, fig_floors_geometry, fig_time_forecast,
               figS_all_learners, figS_duplicates, figS_time):
        log(f"  {fn.__name__} -> {fn()}")
    from . import tables
    tables.stage(P, log)
