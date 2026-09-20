import pytest

from jevc.schema import Item, Option, Pair, Source, assign_pool, check_partition, finite_normalized, order_options


def mk(iid="x1", gid="g1", gold="a", pool="dev", **kw):
    d = dict(id=iid, group_id=gid, family="clinc", output_type="choice", template_id="t", domain="d",
             state="s", question="q", options=[Option("a", "A", "da"), Option("b", "B", "db"), Option("c", "C", "dc")],
             gold=gold, source=Source("ds", "cfg", "rev", "validation", 1, "cc-by-3.0", "overlap_unknown"), pool=pool)
    d.update(kw)
    return Item(**d)


def test_valid_item():
    mk().validate()


@pytest.mark.parametrize("bad", [
    dict(gold="zzz"), dict(options=[Option("a", "A", "d"), Option("a", "A2", "d")]), dict(options=[Option("a", "A", "")]),
    dict(family="nope"), dict(pool="staging"), dict(source=Source("ds", "c", "r", "s", 1, "lic", "unknown")),
])
def test_invalid_items(bad):
    with pytest.raises(AssertionError):
        mk(**bad).validate()


def test_order_options_deterministic_and_not_gold_first():
    opts = [Option("yes", "yes", "d"), Option("no", "no", "d")]
    firsts = [order_options(opts, f"item-{i}", 1)[0].id for i in range(200)]
    assert order_options(opts, "item-7", 1) == order_options(opts, "item-7", 1)
    assert 40 < firsts.count("yes") < 160  # not systematically first
    assert {o.id for o in order_options(opts, "item-7", 1)} == {"yes", "no"}


def test_assign_pool_deterministic_and_covers():
    fr = {"dev": 0.4, "cal": 0.2, "test": 0.4}
    pools = [assign_pool(f"g{i}", 3, fr) for i in range(2000)]
    assert pools == [assign_pool(f"g{i}", 3, fr) for i in range(2000)]
    assert 0.3 < pools.count("dev") / 2000 < 0.5 and 0.1 < pools.count("cal") / 2000 < 0.3


def test_check_partition_detects_group_leak():
    check_partition({"dev": [mk("a", "g1")], "test": [mk("b", "g2", pool="test")]})
    with pytest.raises(AssertionError):
        check_partition({"dev": [mk("a", "g1")], "test": [mk("b", "g1", pool="test")]})
    with pytest.raises(AssertionError):  # item claims a pool other than where it is stored
        check_partition({"dev": [mk("a", "g1", pool="test")]})


def test_pair_validation_and_ancestry():
    a, b = mk("p1a", "g1", gold="a"), mk("p1b", "g1", gold="b", state="s2")
    p = Pair("p1", "state_flip", "g1", "flip", None, "candidate", "", a, b)
    p.validate()
    with pytest.raises(AssertionError):  # same gold is not a flip
        Pair("p2", "state_flip", "g1", "", None, "candidate", "", a, mk("p1c", "g1", gold="a", state="s3")).validate()
    with pytest.raises(AssertionError):  # different parent group
        Pair("p3", "state_flip", "g1", "", None, "candidate", "", a, mk("p1d", "g9", gold="b", state="s3")).validate()
    with pytest.raises(AssertionError):  # pair ancestry crossing pools
        check_partition({"dev": [mk("z", "g1")], "test": []}, [Pair("p4", "state_flip", "g1", "", None, "candidate", "", a, mk("p1e", "g1", gold="b", state="s3", pool="test"))])


def test_finite_normalized():
    assert finite_normalized([0.25, 0.75]) and not finite_normalized([0.5, 0.6]) and not finite_normalized([float("nan"), 1])
