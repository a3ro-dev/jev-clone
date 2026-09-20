"""Classification / probability / risk-coverage metrics with grouped bootstrap."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

EPS = 1e-12


def accuracy(recs) -> float:
    return float(np.mean([r["correct"] for r in recs])) if recs else math.nan


def macro_f1(recs) -> float:
    classes = sorted({r["gold"] for r in recs} | {r["pred"] for r in recs})
    f1s = []
    for c in classes:
        tp = sum(r["gold"] == c and r["pred"] == c for r in recs)
        fp = sum(r["gold"] != c and r["pred"] == c for r in recs)
        fn = sum(r["gold"] == c and r["pred"] != c for r in recs)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else math.nan


def confusion(recs) -> dict[str, int]:
    return {f"{g}->{p}": n for (g, p), n in sorted(Counter((r["gold"], r["pred"]) for r in recs).items())}


def log_loss(recs) -> float:
    return float(-np.mean([math.log(max(r["probs"][r["gold"]], EPS)) for r in recs])) if recs else math.nan


def brier(recs) -> float:
    """Multiclass Brier: sum over options of (p - onehot)^2, averaged over items."""
    return float(np.mean([sum((p - (k == r["gold"])) ** 2 for k, p in r["probs"].items()) for r in recs])) if recs else math.nan


def reliability(recs, n_bins: int = 10) -> dict:
    """Equal-mass bins on top-label confidence. ECE is a *diagnostic*, not a gate."""
    if not recs:
        return {"bins": [], "ece": math.nan}
    conf = np.array([r["confidence"] for r in recs])
    corr = np.array([r["correct"] for r in recs], dtype=float)
    order = np.argsort(conf, kind="stable")
    n_bins = max(1, min(n_bins, len(recs) // 5))
    bins, ece = [], 0.0
    for idx in np.array_split(order, n_bins):
        c, a = float(conf[idx].mean()), float(corr[idx].mean())
        bins.append({"n": int(len(idx)), "conf_mean": c, "acc": a, "conf_min": float(conf[idx].min()),
                     "conf_max": float(conf[idx].max())})
        ece += len(idx) / len(recs) * abs(c - a)
    return {"bins": bins, "ece": float(ece)}


def risk_coverage(recs) -> dict:
    """Sort by confidence desc; risk(c) = error rate among the top c fraction. AURC = mean risk over the curve."""
    if not recs:
        return {"aurc": math.nan, "error_at_50": math.nan, "error_at_80": math.nan, "curve": []}
    order = sorted(recs, key=lambda r: -r["confidence"])
    err = np.cumsum([not r["correct"] for r in order]) / np.arange(1, len(order) + 1)
    n = len(order)

    def at(cov):
        return float(err[max(1, math.floor(cov * n)) - 1])  # most-confident <= cov fraction

    cov = np.arange(1, n + 1) / n
    return {"aurc": float(err.mean()), "error_at_50": at(0.5), "error_at_80": at(0.8),
            "curve": [{"coverage": float(c), "risk": float(e)} for c, e in zip(cov, err)]}


def summary(recs) -> dict:
    rc = risk_coverage(recs)
    rel = reliability(recs)
    return {"n": len(recs), "n_groups": len({r["group_id"] for r in recs}), "accuracy": accuracy(recs),
            "macro_f1": macro_f1(recs), "log_loss": log_loss(recs), "brier": brier(recs), "ece_diagnostic": rel["ece"],
            "reliability_bins": rel["bins"], "aurc": rc["aurc"], "error_at_50": rc["error_at_50"],
            "error_at_80": rc["error_at_80"], "confusion": confusion(recs),
            "n_truncated": int(sum(r.get("truncated", False) for r in recs))}


SCALARS = ("accuracy", "macro_f1", "log_loss", "brier", "ece_diagnostic", "aurc", "error_at_50", "error_at_80")


def bootstrap(recs, n_boot: int = 1000, seed: int = 0) -> dict:
    """Resample independent source groups with replacement; all rows of a group travel together."""
    by_group: dict[str, list] = {}
    for r in recs:
        by_group.setdefault(r["group_id"], []).append(r)
    gids = sorted(by_group)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in SCALARS}
    for _ in range(n_boot):
        sample = [r for g in rng.choice(gids, size=len(gids), replace=True) for r in by_group[g]]
        s = summary(sample)
        for k in SCALARS:
            draws[k].append(s[k])
    out = {}
    for k, v in draws.items():
        a = np.array(v, dtype=float)
        a = a[np.isfinite(a)]
        out[k] = {"ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else None,
                  "n_defined": int(len(a))}
    return {"n_boot": n_boot, "seed": seed, "n_groups": len(gids), **out}


def pair_scores(pair_recs: list[tuple[dict, dict]]) -> dict:
    """A pair is correct only if both halves are correct; halves are also reported separately."""
    if not pair_recs:
        return {"n": 0, "both_correct": math.nan, "a_correct": math.nan, "b_correct": math.nan}
    return {"n": len(pair_recs),
            "both_correct": float(np.mean([a["correct"] and b["correct"] for a, b in pair_recs])),
            "a_correct": float(np.mean([a["correct"] for a, b in pair_recs])),
            "b_correct": float(np.mean([b["correct"] for a, b in pair_recs]))}


def invariance(base: list[dict], other: list[dict]) -> dict:
    """Agreement and probability drift between two variants, matched on item_id (semantic option ids)."""
    o = {r["item_id"]: r for r in other}
    pairs = [(b, o[b["item_id"]]) for b in base if b["item_id"] in o]
    if not pairs:
        return {"n": 0}
    l1 = [sum(abs(b["probs"][k] - t["probs"][k]) for k in b["probs"]) for b, t in pairs]
    return {"n": len(pairs), "pred_agreement": float(np.mean([b["pred"] == t["pred"] for b, t in pairs])),
            "mean_abs_prob_change_l1": float(np.mean(l1)), "max_abs_prob_change_l1": float(np.max(l1)),
            "accuracy_base": accuracy([b for b, _ in pairs]), "accuracy_other": accuracy([t for _, t in pairs])}
