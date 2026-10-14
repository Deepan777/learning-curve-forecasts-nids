"""Label-free shift scores computed from the labelled source pilot and UNLABELLED target traffic.

None of these functions sees a target label. Target flows enter only as an unlabelled mixture whose attack share is
fixed by the experimenter (main analysis: the same share as the source mixture; prevalence test: a different share).

c2st_auc       classifier two-sample test (Lopez-Paz & Oquab 2017): cross-validated AUC of a gradient-boosting
               domain classifier separating source from target mixtures; 0.5 = indistinguishable.
c2st_adj       the same after re-weighting the source mixture to the target attack share estimated with black-box
               shift estimation (Lipton et al. 2018), so that a pure change of prevalence is not counted as shift.
ks_share       share of features whose two-sample Kolmogorov-Smirnov test rejects at 0.05 / d (Bonferroni;
               univariate baseline of Rabanser et al. 2019).
disagree_delta increase in the disagreement rate of two pilot classifiers (random forest, gradient boosting) from
               the source to the target mixture.
conf_drop      drop in the random forest's mean maximum class probability from source to target mixture.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def c2st(Us, Ut, seed, folds=5):
    X = np.vstack([Us, Ut]); y = np.r_[np.zeros(len(Us)), np.ones(len(Ut))]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


def bbse_prevalence(clf, Zva_a, Zva_b, Ut):
    """Black-box shift estimate of the target attack share from the pilot classifier's rates on source validation."""
    tpr = float(clf.predict(Zva_a).mean()); fpr = float(clf.predict(Zva_b).mean())
    r = float(clf.predict(Ut).mean())
    if tpr - fpr < 0.05:
        return np.nan
    return float(np.clip((r - fpr) / (tpr - fpr), 0.0, 1.0))


def ks_share(Us, Ut):
    d = Us.shape[1]; k = 0
    for j in range(d):
        if np.ptp(Us[:, j]) == 0 and np.ptp(Ut[:, j]) == 0:
            continue
        if ks_2samp(Us[:, j], Ut[:, j]).pvalue < 0.05 / d:
            k += 1
    return k / d


def mixture(rng, Za, Zb, m_a, m_b):
    """Sample WITHOUT replacement (repeated rows would let the domain classifier memorise them across folds)."""
    ia = rng.choice(len(Za), min(m_a, len(Za)), replace=False)
    ib = rng.choice(len(Zb), min(m_b, len(Zb)), replace=False)
    return np.vstack([Za[ia], Zb[ib]])


def compute(Ztr, ytr, Zva_a, Zva_b, Zt_a, Zt_b, m_a, m_b, seed, prevalences=()):
    """All signals for one scenario and repeat. Z* are standardised with the pilot's training statistics."""
    rng = np.random.default_rng(seed)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf = RandomForestClassifier(200, n_jobs=1, random_state=seed).fit(Ztr, ytr)
        hgb = HistGradientBoostingClassifier(min_samples_leaf=min(20, max(1, len(ytr) // 20)), early_stopping=False,
                                             random_state=seed).fit(Ztr, ytr)
    Us = mixture(rng, Zva_a, Zva_b, m_a, m_b)

    def one(Ut, tag):
        out[f"c2st_auc{tag}"] = c2st(Us, Ut, seed)
        pi_s = m_a / (m_a + m_b)
        pi_t = bbse_prevalence(rf, Zva_a, Zva_b, Ut)
        out[f"pi_hat{tag}"] = pi_t
        if np.isfinite(pi_t):
            n_tot = m_a + m_b
            ma = int(round(pi_t * n_tot)); mb = n_tot - ma
            Us_adj = mixture(rng, Zva_a, Zva_b, max(ma, 1), max(mb, 1))
            out[f"c2st_adj{tag}"] = c2st(Us_adj, Ut, seed)
        else:
            out[f"c2st_adj{tag}"] = out[f"c2st_auc{tag}"]
        out[f"ks_share{tag}"] = ks_share(Us, Ut)
        dis_s = float((rf.predict(Us) != hgb.predict(Us)).mean())
        dis_t = float((rf.predict(Ut) != hgb.predict(Ut)).mean())
        out[f"disagree_delta{tag}"] = dis_t - dis_s
        out[f"conf_drop{tag}"] = float(rf.predict_proba(Us).max(1).mean() - rf.predict_proba(Ut).max(1).mean())
        out[f"pi_source{tag}"] = pi_s

    one(mixture(rng, Zt_a, Zt_b, m_a, m_b), "")
    for pt in prevalences:                       # same target features, different attack share
        n_tot = m_a + m_b
        ma = min(max(1, int(round(pt * n_tot))), len(Zt_a))
        mb = min(n_tot - ma, len(Zt_b))
        one(mixture(rng, Zt_a, Zt_b, ma, mb), f"_prev{pt}")
        out[f"pi_target_prev{pt}"] = ma / (ma + mb)
    return out
