import math

import pytest

from jevc import metrics


def rec(gold, probs, gid="g"):
    pred = max(probs, key=probs.get)
    return {"item_id": f"{gid}-{gold}-{pred}", "group_id": gid, "gold": gold, "pred": pred, "probs": probs,
            "correct": pred == gold, "confidence": probs[pred], "truncated": False}


R = [rec("a", {"a": 0.9, "b": 0.1}, "g1"), rec("a", {"a": 0.4, "b": 0.6}, "g1"),
     rec("b", {"a": 0.2, "b": 0.8}, "g2"), rec("b", {"a": 0.7, "b": 0.3}, "g3")]


def test_accuracy_macro_f1_confusion():
    assert metrics.accuracy(R) == 0.5
    # per class: a: tp1 fp1 fn1 -> f1 .5 ; b: tp1 fp1 fn1 -> .5
    assert metrics.macro_f1(R) == pytest.approx(0.5)
    assert metrics.confusion(R) == {"a->a": 1, "a->b": 1, "b->a": 1, "b->b": 1}


def test_log_loss_and_brier():
    ll = -(math.log(0.9) + math.log(0.4) + math.log(0.8) + math.log(0.3)) / 4
    assert metrics.log_loss(R) == pytest.approx(ll)
    br = ((0.1**2 + 0.1**2) + (0.6**2 + 0.6**2) + (0.2**2 + 0.2**2) + (0.7**2 + 0.7**2)) / 4
    assert metrics.brier(R) == pytest.approx(br)
    assert metrics.brier([rec("a", {"a": 1.0, "b": 0.0})]) == 0.0


def test_risk_coverage():
    rc = metrics.risk_coverage(R)  # confidences desc: .9(ok) .8(ok) .7(err) .6(err)
    assert rc["error_at_50"] == 0.0 and rc["error_at_80"] == pytest.approx(1 / 3)
    assert rc["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def test_reliability_bins_equal_mass_and_ece():
    recs = [rec("a", {"a": c, "b": 1 - c}) if c >= 0.5 else rec("b", {"a": c, "b": 1 - c}) for c in [0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.51, 0.52, 0.53]]
    out = metrics.reliability(recs, n_bins=2)
    assert [b["n"] for b in out["bins"]] == [5, 5]
    assert all(b["acc"] == 1.0 for b in out["bins"])
    assert out["ece"] == pytest.approx(sum(0.5 * abs(b["conf_mean"] - 1.0) for b in out["bins"]))


def test_bootstrap_deterministic_and_grouped():
    b1, b2 = metrics.bootstrap(R, n_boot=50, seed=1), metrics.bootstrap(R, n_boot=50, seed=1)
    assert b1 == b2 and b1["n_groups"] == 3
    assert b1["accuracy"]["ci95"][0] >= 0.0 and b1["accuracy"]["ci95"][1] <= 1.0


def test_pair_scores_and_invariance():
    a, b = rec("a", {"a": 0.9, "b": 0.1}), rec("b", {"a": 0.6, "b": 0.4})
    ps = metrics.pair_scores([(a, a), (a, b)])
    assert ps["both_correct"] == 0.5 and ps["a_correct"] == 1.0 and ps["b_correct"] == 0.5
    inv = metrics.invariance([a], [dict(a, probs={"a": 0.8, "b": 0.2})])
    assert inv["pred_agreement"] == 1.0 and inv["mean_abs_prob_change_l1"] == pytest.approx(0.2)
