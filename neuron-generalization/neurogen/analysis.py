"""Scoring predictions and the pre-registered verdict rule.

Nothing here looks at *how plausible* an explanation sounds. A hypothesis earns
support only through out-of-sample predictions, and preferentially through items
on which H1 and H2 disagree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value for k successes in n trials at p=0.5."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    obs = pmf[k]
    return float(min(1.0, sum(p for p in pmf if p <= obs + 1e-12)))


def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney)."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    p = np.asarray(pos_scores)[:, None]
    n = np.asarray(neg_scores)[None, :]
    return float(((p > n).mean() + 0.5 * (p == n).mean()))


def _metrics(pred: np.ndarray, act: np.ndarray, observed: np.ndarray) -> dict:
    """pred: 1 active / 0 inactive; observed: bool active; act: continuous signed activation."""
    if len(pred) == 0:
        return dict(n=0, accuracy=np.nan, balanced_accuracy=np.nan, auc=np.nan,
                    tpr=np.nan, tnr=np.nan)
    tp = ((pred == 1) & observed).sum(); fn = ((pred == 0) & observed).sum()
    tn = ((pred == 0) & ~observed).sum(); fp = ((pred == 1) & ~observed).sum()
    tpr = tp / (tp + fn) if tp + fn else np.nan
    tnr = tn / (tn + fp) if tn + fp else np.nan
    bal = np.nanmean([tpr, tnr]) if not (np.isnan(tpr) and np.isnan(tnr)) else np.nan
    return dict(n=int(len(pred)), accuracy=float((pred == observed).mean()),
                balanced_accuracy=float(bal), auc=auc(act[pred == 1], act[pred == 0]),
                tpr=float(tpr), tnr=float(tnr))


def score_hypothesis(rows: list[dict], h: str, threshold: float, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    defin = [r for r in rows if r[h] in ("active", "inactive")]
    pred = np.array([r[h] == "active" for r in defin], dtype=int)
    act = np.array([r["target_act"] for r in defin], dtype=float)
    obs = act > threshold
    m = _metrics(pred, act, obs)
    m["coverage"] = len(defin) / max(len(rows), 1)
    rng = np.random.default_rng(seed)
    bals = []
    for _ in range(n_boot if len(defin) > 1 else 0):
        ix = rng.integers(0, len(defin), len(defin))
        bals.append(_metrics(pred[ix], act[ix], obs[ix])["balanced_accuracy"])
    bals = np.array([b for b in bals if not np.isnan(b)])
    m["balanced_accuracy_ci"] = (tuple(np.percentile(bals, [2.5, 97.5]).tolist())
                                 if len(bals) else (np.nan, np.nan))
    cx = [r for r in defin if r["category"] == "counterexample"]
    m["counterexample_false_positive_rate"] = (
        float(np.mean([(r[h] == "inactive") and r["target_act"] > threshold for r in cx]))
        if cx else np.nan)
    return m


def discrimination(rows: list[dict], threshold: float) -> dict:
    """Items where H1 and H2 make opposite definite predictions."""
    d = [r for r in rows if {r["h1"], r["h2"]} == {"active", "inactive"}]
    h1_right = sum((r["h1"] == "active") == (r["target_act"] > threshold) for r in d)
    h2_right = len(d) - h1_right
    return dict(n_discriminating=len(d), h1_wins=h1_right, h2_wins=h2_right,
                sign_test_p=binom_two_sided_p(min(h1_right, h2_right), len(d)) if d else 1.0)


@dataclass
class VerdictRule:
    min_discriminating_items: int = 8
    support_balanced_acc: float = 0.75
    neither_balanced_acc: float = 0.65
    alpha: float = 0.05


def verdict(s1: dict, s2: dict, disc: dict, rule: VerdictRule) -> tuple[str, str]:
    """Pre-registered decision rule. Returns (label, reason).

    Labels: "supports H1", "supports H2", "neither", "inconclusive".
    """
    b1, b2 = s1["balanced_accuracy"], s2["balanced_accuracy"]
    b1 = 0.0 if np.isnan(b1) else b1
    b2 = 0.0 if np.isnan(b2) else b2
    if b1 < rule.neither_balanced_acc and b2 < rule.neither_balanced_acc:
        return "neither", f"both balanced accuracies below {rule.neither_balanced_acc} ({b1:.2f}, {b2:.2f})"
    if disc["n_discriminating"] < rule.min_discriminating_items:
        return "inconclusive", (f"only {disc['n_discriminating']} discriminating items "
                                f"(< {rule.min_discriminating_items})")
    sig = disc["sign_test_p"] < rule.alpha
    if sig and disc["h2_wins"] > disc["h1_wins"] and b2 >= rule.support_balanced_acc:
        return "supports H2", (f"H2 wins {disc['h2_wins']}/{disc['n_discriminating']} discriminating "
                               f"items (p={disc['sign_test_p']:.3g}); bal.acc {b2:.2f}")
    if sig and disc["h1_wins"] > disc["h2_wins"] and b1 >= rule.support_balanced_acc:
        return "supports H1", (f"H1 wins {disc['h1_wins']}/{disc['n_discriminating']} discriminating "
                               f"items (p={disc['sign_test_p']:.3g}); bal.acc {b1:.2f}")
    if not sig and min(b1, b2) >= rule.support_balanced_acc:
        return "inconclusive", "both hypotheses predict well and the discriminating items do not separate them"
    return "inconclusive", (f"no hypothesis met the support criterion (p={disc['sign_test_p']:.3g}, "
                            f"bal.acc H1={b1:.2f}, H2={b2:.2f})")


def evaluate_neuron(rows: list[dict], threshold: float, rule: VerdictRule,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    s1 = score_hypothesis(rows, "h1", threshold, n_boot, seed)
    s2 = score_hypothesis(rows, "h2", threshold, n_boot, seed)
    disc = discrimination(rows, threshold)
    label, reason = verdict(s1, s2, disc, rule)
    by_cat: dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"n": 0, "n_active": 0})
        c["n"] += 1
        c["n_active"] += int(r["target_act"] > threshold)
    return dict(threshold=threshold, h1=s1, h2=s2, discrimination=disc, by_category=by_cat,
                verdict=label, reason=reason)


# --------------------------------------------------------------------------- geometry
def two_cluster_separation(X: np.ndarray, seed: int = 0, n_iter: int = 50) -> dict:
    """Spherical 2-means on MLP inputs at top-activating tokens + silhouette.

    Descriptive only: well-separated clusters are *compatible* with H1 (distinct input
    directions both driving the neuron), a single cluster with H2, but neither
    observation is decisive (an abstraction can be computed from distinct inputs).
    """
    if len(X) < 4:
        return dict(n=len(X), silhouette=np.nan, sizes=[len(X)])
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
    rng = np.random.default_rng(seed)
    C = Xn[rng.choice(len(Xn), 2, replace=False)]
    for _ in range(n_iter):
        lab = (Xn @ C.T).argmax(1)
        newC = np.stack([Xn[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(2)])
        newC /= np.linalg.norm(newC, axis=1, keepdims=True).clip(1e-8)
        if np.allclose(newC, C):
            break
        C = newC
    D = 1 - Xn @ Xn.T
    sil = []
    for i in range(len(Xn)):
        same = lab == lab[i]
        if same.sum() <= 1 or (~same).sum() == 0:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = D[i, ~same].mean()
        sil.append((b - a) / max(a, b, 1e-12))
    return dict(n=len(X), silhouette=float(np.mean(sil)) if sil else np.nan,
                sizes=[int((lab == j).sum()) for j in range(2)], labels=lab.tolist())
