"""Data figures of the revised manuscript (Figs. 2-6 and Supplementary Fig. S4), drawn only from the verified result
files (09_Statistical_Analysis/shift, 06_Experiments/shift). Figure 1 is a native TikZ diagram
(14_Manuscript/figures/tikz/fig1_framework.tikz) and is not produced here.

Style: 180 mm wide figures; DejaVu Sans (the same sans-serif face is used by the TikZ framework figure); lines and
markers >= 1 pt; lower-case bold panel letters; every colour is paired with a second encoding (line style, marker
shape or hatching) so that the figures remain readable in grayscale. Files go to 11_Figures/ with the prefix 'rev_'
(PDF and SVG vector, 300-dpi PNG)."""
from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lcurve import config  # noqa: E402

from . import analysis, scenarios  # noqa: E402

INK, INK2, MUTED, GRID = "#111111", "#4d4d4d", "#8c8c8c", "#e6e6e6"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
KIND_ORDER = ["id", "temporal", "benign", "lovo"]
KIND_NAME = {"id": "In-distribution", "temporal": "Temporal", "benign": "Benign change", "lovo": "Held-out variant"}
KIND_COL = {"id": SLOTS[0], "temporal": SLOTS[1], "benign": SLOTS[2], "lovo": SLOTS[6]}
KIND_MK = {"id": "o", "temporal": "s", "benign": "^", "lovo": "D"}
KIND_LS = {"id": "-", "temporal": (0, (4, 1.6)), "benign": (0, (1.2, 1.4)), "lovo": (0, (5, 1.5, 1.2, 1.5))}
SIG_NAME = {"c2st_adj": "C2ST, label-shift adjusted", "c2st_auc": "C2ST", "ks_share": "KS share",
            "disagree_delta": "Disagreement change", "conf_drop": "Confidence drop",
            "pilot_width": "Pilot width (no target data)"}
STAT = analysis.STAT
W = 7.09                                    # 180 mm
SIZES_TICKS = [4, 64, 1024, 16384]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
                     "axes.linewidth": 1.0, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
                     "xtick.minor.width": 1.0, "ytick.minor.width": 1.0, "lines.linewidth": 1.3,
                     "hatch.linewidth": 1.0, "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})


def _cap(s):
    """Journal lettering style: only the first letter of each label capitalised (maths and numbers untouched)."""
    return s[0].upper() + s[1:] if s and s[0].isalpha() and s[0].islower() else s


def _sentence_case(fig):
    from matplotlib.ticker import FixedFormatter
    for ax in fig.axes:
        texts = [ax.xaxis.label, ax.yaxis.label, ax.title] + list(ax.texts)
        if ax.get_legend() is not None:
            texts += list(ax.get_legend().get_texts())
        for t in texts:
            s = t.get_text()
            if len(s) == 1 and t.get_fontweight() == "bold":       # panel letters stay lower-case bold
                continue
            t.set_text(_cap(s))
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, FixedFormatter):
                fmt.seq = [_cap(str(s)) for s in fmt.seq]
    for leg in fig.legends:
        for t in leg.get_texts():
            t.set_text(_cap(t.get_text()))
    for t in fig.texts:
        t.set_text(_cap(t.get_text()))


def _save(fig, name):
    _sentence_case(fig)
    out = os.path.join(config.FIG_DIR, name)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


def _panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def _grid(ax, axis="both"):
    ax.grid(True, which="major", axis=axis, color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)


def _logx(ax, ticks=SIZES_TICKS):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])            # exact values, thousands separated by commas
    ax.minorticks_off()


def _primary():
    f = pd.read_csv(os.path.join(STAT, "forecasts.csv"), keep_default_na=False, na_values=[""])
    f = f[f.primary.astype(str) == "True"].copy() if "primary" in f else f
    for c in ("within2", "verifiable", "unsafe"):
        f[c] = f[c].astype(str) == "True"
    return f


# ---------------------------------------------------------------- Figure 2: requirements and in-distribution forecasts
def _km_curve(t, e, grid):
    """1 - Kaplan-Meier survival: share of curves that reached the target by size x (right-censored at n_max)."""
    t = np.asarray(t, float); e = np.asarray(e, bool)
    out = []
    for x in grid:
        surv, n_at = 1.0, len(t)
        for ti in np.unique(t[t <= x]):
            d = int(((t == ti) & e).sum()); c = int((t == ti).sum())
            if d:
                surv *= 1 - d / n_at
            n_at -= c
        out.append(1 - surv)
    return np.array(out)


def _scatter(ax, g, kind, top=2 ** 16, xlabel=True, ylabel=True, count=True):
    lo, hi = 2, top
    fc = g.fc_M2.astype(float).clip(lo, None); ob = g.nstar_tgt.astype(float)
    fcp = np.where(np.isfinite(fc) & (fc <= g.n_max), fc, hi * 0.8)
    obp = np.where(np.isfinite(ob), ob, hi * 0.8)
    xs = np.array([lo, hi])
    ax.fill_between(xs, xs / 2, xs * 2, color="#ececec", lw=0, zorder=1)
    ax.plot(xs, xs, color=MUTED, lw=1.0, zorder=2)
    ax.axhline(hi * 0.55, color=GRID, lw=1.0, zorder=1); ax.axvline(hi * 0.55, color=GRID, lw=1.0, zorder=1)
    ax.scatter(fcp, obp, s=11, marker=KIND_MK[kind], facecolor=KIND_COL[kind], edgecolor="white", linewidth=0.3,
               alpha=0.8, zorder=3)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xticks([4, 64, 1024, hi * 0.8]); ax.set_xticklabels(["4", "64", "1,024", "beyond"])
    ax.set_yticks([4, 64, 1024, 16384, hi * 0.8]); ax.set_yticklabels(["4", "64", "1,024", "16,384", "never"])
    ax.minorticks_off()
    if xlabel:
        ax.set_xlabel("forecast " + r"$\hat n$" + " (flows per class)")
    if ylabel:
        ax.set_ylabel("realised " + r"$n^{*}_{\mathrm{tgt}}$" + " (flows per class)")
    if count:
        ax.text(0.04, 0.96, f"{len(g)} forecasts", transform=ax.transAxes, fontsize=6.3, color=INK2, va="top")
    _grid(ax)


def fig2_indistribution():
    f = _primary(); f = f[f.kind == "id"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    grid = np.exp(np.linspace(np.log(4), np.log(16384), 160))
    styles = {"RF": (SLOTS[0], "-"), "HGB": (SLOTS[1], (0, (4, 1.6))), "LogReg": (SLOTS[2], (0, (1.2, 1.3))),
              "kNN1": (SLOTS[3], (0, (5, 1.5, 1.2, 1.5))), "kNN5": (SLOTS[6], (0, (2.5, 1.2)))}
    for L, (col, ls) in styles.items():
        g = f[f.learner == L]
        t = np.where(np.isfinite(g.nstar_tgt), g.nstar_tgt, g.n_max); e = np.isfinite(g.nstar_tgt).values
        ax.step(grid, _km_curve(t, e, grid), where="post", color=col, ls=ls, lw=1.5,
                label={"kNN1": "1-NN", "kNN5": "5-NN", "LogReg": "logistic regression",
                       "RF": "random forest", "HGB": "gradient boosting"}[L])
    _logx(ax); _grid(ax)
    ax.set_ylim(0, 1); ax.set_xlim(4, 16384)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.legend(loc="upper left", handlelength=2.6)
    ax.set_xlabel("labelled flows per class, " + r"$n$")
    ax.set_ylabel("tasks reaching 1% balanced error (%)")
    _panel(ax, "a", x=-0.13)
    ax = axs[1]
    _scatter(ax, f, "id")
    _panel(ax, "b", x=-0.2)
    fig.tight_layout(w_pad=2.2)
    return _save(fig, "rev_fig2_indistribution")


# ---------------------------------------------------------------- Figure 3: forecasts under source-target shift
def fig3_shift():
    f = _primary()
    fig = plt.figure(figsize=(W, 4.95))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.78], hspace=0.62, wspace=0.3)
    for i, k in enumerate(KIND_ORDER):
        ax = fig.add_subplot(gs[0, i])
        g = f[f.kind == k]
        _scatter(ax, g, k, xlabel=False, ylabel=(i == 0), count=False)
        ax.set_title(f"{KIND_NAME[k]}\n{len(g)} forecasts", color=INK, pad=4,
                     linespacing=1.25)
        if i:
            ax.set_yticklabels([])
        if i == 0:
            _panel(ax, "a", x=-0.36)
    fig.text(0.5, 0.455, "forecast " + r"$\hat n$" + " from the pilot (flows per class)", ha="center", fontsize=7.5)
    ax = fig.add_subplot(gs[1, :])
    cats = [("within " + r"$\times$" + "2", lambda g: g.within2, SLOTS[0], None),
            ("outside " + r"$\times$" + "2", lambda g: g.category.eq("measurable") & ~g.within2, "#a9c8ef", None),
            ("both beyond data", lambda g: g.category.eq("both_beyond"), "#c9c9c9", ".."),
            ("conservative miss", lambda g: g.category.eq("conservative_miss"), SLOTS[3], "//"),
            ("false assurance", lambda g: g.category.eq("false_assurance"), SLOTS[7], "xx")]
    y = np.arange(len(KIND_ORDER))[::-1]
    for j, k in enumerate(KIND_ORDER):
        g = f[f.kind == k]
        left = 0.0
        for lab, sel, col, hatch in cats:
            w = float(sel(g).mean())
            ax.barh(y[j], w, left=left, color=col, height=0.64, edgecolor="white", lw=0.8, hatch=hatch,
                    label=lab if j == 0 else None)
            if w >= 0.06:
                ax.text(left + w / 2, y[j], f"{100 * w:.0f}", ha="center", va="center", fontsize=6.5, zorder=6,
                        color="white" if (col == SLOTS[0] and hatch is None) else INK,
                        bbox=None if hatch is None else dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))
            left += w
    ax.set_yticks(y); ax.set_yticklabels([KIND_NAME[k] for k in KIND_ORDER])
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("forecasts in each outcome category (%)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.3), handlelength=1.6, columnspacing=1.2)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    _panel(ax, "b", x=-0.115)
    return _save(fig, "rev_fig3_shift_forecasts")


# ---------------------------------------------------------------- Figure 4: one family under the four scenario kinds
def fig4_examples(ds="cicids", fam="BruteForce", learner="RF"):
    c = pd.concat([pd.read_csv(p, keep_default_na=False) for p in
                   glob.glob(os.path.join(scenarios.SHIFT_DIR, "curves", f"*__{ds}__{fam}*.csv"))
                   if "__noart" not in p], ignore_index=True)
    f = _primary()
    sids = [s for s in sorted(c.sid.unique()) if s.split("__")[0] in KIND_ORDER]
    order = sorted(sids, key=lambda s: (KIND_ORDER.index(s.split("__")[0]), s))
    fig, axs = plt.subplots(1, len(order), figsize=(W, 2.3), sharey=True)
    for i, (ax, sid) in enumerate(zip(np.atleast_1d(axs), order)):
        kind = sid.split("__")[0]
        g = c[(c.sid == sid) & (c.learner == learner)].groupby("n")[["be_val", "be_tgt"]].mean()
        n = g.index.values.astype(float)
        ax.axvspan(4, 128, color="#efefef", lw=0, zorder=0)
        ax.plot(n, g.be_val, color=MUTED, lw=1.3, marker="o", ms=3, mfc="white", mew=1.0, zorder=3)
        ax.plot(n, g.be_tgt, color=KIND_COL[kind], lw=1.6, marker=KIND_MK[kind], ms=3.2, zorder=4)
        ax.axhline(0.01, color=INK2, lw=1.0, ls=(0, (2, 2)), zorder=2)
        r = f[(f.sid == sid) & (f.learner == learner)]
        if len(r) and np.isfinite(r.fc_M2.iloc[0]) and r.fc_M2.iloc[0] <= n.max():
            ax.axvline(r.fc_M2.iloc[0], color=INK, lw=1.0, zorder=2)
            ax.text(r.fc_M2.iloc[0] * 1.2, 2.2e-4, r"$\hat n$", fontsize=7, color=INK,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1); ax.set_xlim(4, 16384)
        _logx(ax, [4, 64, 1024, 16384]); _grid(ax)
        title = KIND_NAME[kind] if kind != "lovo" else "Variant: " + sid.split("__")[3].split("-")[0]
        ax.set_title(title, color=INK, pad=4)
        ax.set_xlabel(r"$n$" + " (flows per class)")
        if i == 0:
            ax.set_ylabel("balanced error")
            ax.text(5, 1.0e-3, "pilot", fontsize=6.3, color=INK2, rotation=90, va="bottom")
        _panel(ax, "abcdefgh"[i], x=-0.12 if i else -0.36)
    h = [plt.Line2D([], [], color=MUTED, lw=1.3, marker="o", ms=3, mfc="white", mew=1.0)] + \
        [plt.Line2D([], [], color=KIND_COL[k], lw=1.6, marker=KIND_MK[k], ms=3.2) for k in KIND_ORDER]
    fig.legend(h, ["source validation (pilot curve)"] + [f"target: {KIND_NAME[k].lower()}" for k in KIND_ORDER],
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.09, 1, 1))
    return _save(fig, "rev_fig4_example_family")


# ---------------------------------------------------------------- Figure 5: label-free reliability scores
def fig5_reliability():
    f = _primary(); v = f[f.verifiable].copy()
    res = pd.read_csv(analysis.RES)
    res = res[res.section == "F3"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.65), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    ax = axs[0]
    sigs = ["ks_share", "conf_drop", "disagree_delta", "c2st_auc", "c2st_adj", "pilot_width"]
    for i, s in enumerate(sigs):
        y = len(sigs) - 1 - i
        for dy, metric, filled in ((0.13, f"auroc_unsafe_{s}", True), (-0.13, f"auroc_unsafe_shift_only_{s}", False)):
            r = res[res.metric == metric]
            if not len(r):
                continue
            ax.plot([r.ci_lo.iloc[0], r.ci_hi.iloc[0]], [y + dy] * 2, color=INK2, lw=1.1)
            ax.plot(r.value.iloc[0], y + dy, "o" if filled else "s", ms=4.5,
                    mfc=(SLOTS[0] if filled else "white"), mec=SLOTS[0], mew=1.1)
    ax.axvline(0.5, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.set_yticks(range(len(sigs))); ax.set_yticklabels([SIG_NAME[s] for s in sigs][::-1])
    ax.set_xlim(0.15, 1.0); ax.set_xlabel("AUROC, unsafe vs safe forecasts")
    h = [plt.Line2D([], [], ls="", marker="o", ms=4.5, mfc=SLOTS[0], mec=SLOTS[0]),
         plt.Line2D([], [], ls="", marker="s", ms=4.5, mfc="white", mec=SLOTS[0], mew=1.1)]
    ax.legend(h, ["all scenarios", "shifted scenarios only"], loc="lower center", bbox_to_anchor=(0.45, 1.0),
              ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)
    _grid(ax, "x"); _panel(ax, "a", x=-0.86)
    ax = axs[1]
    base = v.unsafe.mean()
    lines = (("ks_share", SLOTS[3], "-"), ("conf_drop", SLOTS[2], (0, (4, 1.6))), ("c2st_auc", SLOTS[6], (0, (1.2, 1.3))),
             ("c2st_adj", SLOTS[0], (0, (5, 1.5, 1.2, 1.5))), ("pilot_width", MUTED, (0, (2.5, 1.2))))
    for s, col, ls in lines:
        vv = v[np.isfinite(v[s])]
        cov, risk = analysis.risk_coverage(vv[s].values, vv.unsafe.values)
        ax.plot(cov, risk, color=col, ls=ls, lw=1.4, label={"ks_share": "KS share", "conf_drop": "confidence drop",
                                                             "c2st_auc": "C2ST", "c2st_adj": "C2ST, adjusted",
                                                             "pilot_width": "pilot width"}[s])
    ax.axhline(base, color=INK2, lw=1.0, ls=(0, (2, 2)), label="no abstention")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("coverage (% of forecasts accepted)"); ax.set_ylabel("unsafe among accepted (%)")
    ax.legend(loc="upper right", handlelength=2.4, borderaxespad=0.2)
    _grid(ax); _panel(ax, "b", x=-0.24)
    ax = axs[2]
    r = pd.read_csv(os.path.join(STAT, "lodo_c2st_adj.csv"))
    x = np.arange(len(r))
    u0 = r.unsafe_all / r.n; u1 = (r.unsafe_accepted / r.accepted.replace(0, np.nan)).fillna(0)
    ax.bar(x - 0.19, u0, 0.36, color="#c9c9c9", edgecolor="white", label="all forecasts")
    ax.bar(x + 0.19, u1, 0.36, color=SLOTS[0], edgecolor="white", hatch="///", label="accepted forecasts")
    for xi, cov in zip(x, r.accepted / r.n):
        ax.text(xi + 0.19, u1.iloc[int(xi)] + 0.02, f"{100 * cov:.0f}%", ha="center", va="bottom", fontsize=6,
                color=INK2, rotation=90)
    names = {"nslkdd": "NSL-\nKDD", "unsw": "UNSW-\nNB15", "cicids": "CICIDS\n2017", "ciciot": "CIC-IoT\n2023"}
    ax.set_xticks(x); ax.set_xticklabels([names[d] for d in r.dataset], fontsize=6)
    ax.set_ylabel("unsafe on held-out benchmark (%)")
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.legend(loc="upper left", borderaxespad=0.2)
    _grid(ax, "y"); _panel(ax, "c", x=-0.26)
    fig.tight_layout(w_pad=1.0)
    return _save(fig, "rev_fig5_reliability")


# ---------------------------------------------------------------- Figure 6: prevalence change and operating thresholds
def fig6_operations():
    s = pd.concat([pd.read_csv(p, keep_default_na=False) for p in
                   glob.glob(os.path.join(scenarios.SHIFT_DIR, "signals", "id__*.csv"))], ignore_index=True)
    art = list(scenarios.S.get("artefact_features", {}))
    s = s[~(s.dataset.isin(art) & ~s.sid.str.endswith("__noart"))]
    m = s.groupby("sid").mean(numeric_only=True)
    f = _primary()
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1.1, 1]))
    ax = axs[0]
    conds = [("", "Unchanged"), ("_prev0.05", "5%"), ("_prev0.6", "60%")]
    for j, (suf, _) in enumerate(conds):
        for k, (col, dx, name, mk) in enumerate((("c2st_auc", -0.18, "C2ST", "o"), ("c2st_adj", 0.18, "C2ST, adjusted", "^"))):
            vals = m[col + suf].dropna().values
            xj = j + dx + (np.random.default_rng(j * 7 + k).random(len(vals)) - 0.5) * 0.14
            ax.scatter(xj, vals, s=10, marker=mk, facecolor=SLOTS[6] if k == 0 else SLOTS[0], edgecolor="white",
                       linewidth=0.3, alpha=0.85, label=name if j == 0 else None, zorder=3)
            ax.plot([j + dx - 0.11, j + dx + 0.11], [np.median(vals)] * 2, color=INK, lw=1.5, zorder=4)
    ax.axhline(0.5, color=INK2, lw=1.0, ls=(0, (2, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels([c_[1] for c_ in conds])
    ax.set_xlabel("target attack share (source unchanged)")
    ax.set_ylabel("two-sample AUC, source vs target"); ax.set_ylim(0.45, 0.7)
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    _grid(ax, "y"); _panel(ax, "a", x=-0.24)
    ax = axs[1]
    for i, k in enumerate(KIND_ORDER):
        g = f[f.kind == k].fpr_cal_op.values
        xi = i + (np.random.default_rng(i).random(len(g)) - 0.5) * 0.46
        ax.scatter(xi, np.clip(g, 1e-4, 1), s=7, marker=KIND_MK[k], facecolor=KIND_COL[k], edgecolor="none",
                   alpha=0.5, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(np.clip(g, 1e-4, 1))] * 2, color=INK, lw=1.5, zorder=4)
    ax.axhline(0.01, color=INK2, lw=1.0, ls=(0, (2, 2)))
    ax.set_yscale("log"); ax.set_ylim(1e-4, 0.3)
    short = {"id": "In-\ndistribution", "temporal": "Temporal", "benign": "Benign\nchange", "lovo": "Held-out\nvariant"}
    ax.set_xticks(range(4)); ax.set_xticklabels([short[k] for k in KIND_ORDER]); ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel("target false-positive rate")
    _grid(ax, "y"); _panel(ax, "b", x=-0.24)
    ax = axs[2]
    pis = np.array([0.001, 0.01, 0.1])
    for j, k in enumerate(KIND_ORDER):
        g = f[f.kind == k]
        ppv = [(g.tpr_cal_op * p / (g.tpr_cal_op * p + g.fpr_cal_op * (1 - p))) for p in pis]
        med = np.array([float(q.median()) for q in ppv])
        lo = np.array([float(q.quantile(0.25)) for q in ppv]); hi = np.array([float(q.quantile(0.75)) for q in ppv])
        xs = pis * (1.13 ** (j - 1.5))
        ax.errorbar(xs, med, yerr=[med - lo, hi - med], color=KIND_COL[k], marker=KIND_MK[k], ms=3.6, lw=1.3,
                    ls=KIND_LS[k], capsize=1.8, elinewidth=1.0, label=KIND_NAME[k])
    ax.set_xscale("log"); ax.set_xlim(0.0001, 0.17); ax.set_ylim(0, 1)   # empty left margin holds the legend
    ax.set_xticks(pis); ax.set_xticklabels(["0.1", "1", "10"]); ax.minorticks_off()
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("attack prevalence (%)"); ax.set_ylabel("precision of alerts (%)")
    ax.legend(loc="upper left", handlelength=1.9, borderaxespad=0.2, fontsize=6.3)
    _grid(ax); _panel(ax, "c", x=-0.24)
    fig.tight_layout(w_pad=1.0)
    return _save(fig, "rev_fig6_operations")


# ---------------------------------------------------------------- Supplementary Fig. S4
def figS_floors():
    fl = os.path.join(scenarios.SHIFT_DIR, "floors")
    md = pd.read_csv(os.path.join(fl, "more_data.csv"))
    st = pd.read_csv(os.path.join(STAT, "floors_subtypes.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7))
    ax = axs[0]
    sty = {("Recon", "RF"): (SLOTS[0], "-", "o"), ("Recon", "HGB"): (SLOTS[0], (0, (3, 2)), "s"),
           ("Spoofing", "RF"): (SLOTS[1], "-", "^"), ("Spoofing", "HGB"): (SLOTS[1], (0, (3, 2)), "D")}
    for (fam, L), g in md.groupby(["family", "learner"]):
        mm = 100 * g.groupby("n").be.agg(["mean", "min", "max"])      # balanced error in % (as in Table S14)
        col, ls, mk = sty[(fam, L)]
        ax.plot(mm.index, mm["mean"], color=col, lw=1.4, ls=ls, marker=mk, ms=3,
                label={"Recon": "reconnaissance", "Spoofing": "spoofing"}[fam] + ", " + L)
        ax.fill_between(mm.index, mm["min"], mm["max"], color=col, alpha=0.12, lw=0)
    _logx(ax, [1024, 4096, 16384, 32768]); ax.set_xlim(900, 36000)
    ax.legend(loc="upper right", handlelength=3.2)
    ax.set_ylabel("balanced error (%)"); ax.set_xlabel(r"$n$" + " (flows per class, all available rows)")
    _grid(ax); _panel(ax, "a", x=-0.16)
    ax = axs[1]
    st = st[st.learner == "RF"].sort_values(["family", "share_of_misses"])
    yy = np.arange(len(st))
    famcol = {"Recon": SLOTS[0], "Spoofing": SLOTS[1], "Web": SLOTS[2], "BruteForce": SLOTS[3]}
    famhatch = {"Recon": None, "Spoofing": "//", "Web": "..", "BruteForce": "xx"}
    for yi, r in zip(yy, st.itertuples()):
        ax.barh(yi, r.share_of_misses, color=famcol[r.family], hatch=famhatch[r.family], edgecolor="white", height=0.7)
        ax.text(r.share_of_misses + 0.01, yi, f"miss {100 * r.miss:.1f}%", fontsize=5.8, va="center", color=INK2)
    fam_name = {"Recon": "Recon.", "Spoofing": "Spoofing", "Web": "Web", "BruteForce": "Brute force"}
    ax.set_yticks(yy); ax.set_yticklabels([f"{fam_name[r.family]}: {r.subtype}" for r in st.itertuples()], fontsize=6)
    ax.set_xlabel("share of the family's missed attacks"); ax.set_xlim(0, 1.18)
    _grid(ax, "x"); _panel(ax, "b", x=-0.66)
    fig.tight_layout(w_pad=2)
    return _save(fig, "rev_figS_floors")


def run():
    for fn in (fig2_indistribution, fig3_shift, fig4_examples, fig5_reliability, fig6_operations, figS_floors):
        try:
            print(fn.__name__, fn())
        except Exception as ex:          # report and continue so one missing input does not block the others
            print(fn.__name__, "FAILED", repr(ex)[:300])
