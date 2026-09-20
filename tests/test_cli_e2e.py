"""End-to-end `jevc run` with a fake scorer: exercises variants, pairs, metrics, bootstrap and the report without a model."""

import hashlib
import json

import torch

from jevc import cli, infer


class FakeRuntime:
    device = torch.device("cpu")
    dtype = "fake"


def fake_score(rt, views, input_mode="described_text"):
    out = []
    for q, state, opts in views:  # deterministic: prefer the option whose label shares the most words with the state
        scores = torch.tensor([float(len(set(o.label.split()) & set(state.lower().split()))) + 1e-6 * (int(hashlib.sha1(o.id.encode()).hexdigest(), 16) % 100003) for o in opts])  # order-independent, tie-free
        p = torch.softmax(scores, -1).tolist()
        out.append({"probs": dict(zip([o.id for o in opts], p)), "input_tokens": len(state.split()) + 3,
                    "full_tokens": len(state.split()) + 3, "truncated": False})
    return out


def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(infer, "load_runtime", lambda device: FakeRuntime())
    monkeypatch.setattr(infer, "score_batch", fake_score)
    cli.main(["run", "--run-id", "t1", "--no-probe", "--n-boot", "20"])
    out = tmp_path / "reports" / "t1"
    assert {p.name for p in out.iterdir()} == {"run.json", "predictions.jsonl", "metrics.json", "report.md"}
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for fam in ("boolq", "clinc"):
        r = m["families"][fam]
        assert r["base"]["n"] == 50 and r["bootstrap"]["n_boot"] == 20
        assert set(r["variants"]) == {"no_state", "no_question", "reordered"}
        assert r["reorder_invariance"]["pred_agreement"] == 1.0  # fake scorer is order-independent -> mapping must be exact
        assert r["pairs"]["candidate"]["n"] == 3 and "verified" not in r["pairs"]
    preds = [json.loads(l) for l in (out / "predictions.jsonl").open(encoding="utf-8")]
    assert len(preds) == 2 * (50 * 4 + 6)
    assert all(abs(sum(p["probs"].values()) - 1) < 1e-6 for p in preds)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["paid_spend_usd"] == 0.0 and run["model_revision"] == infer.MODEL_REVISION
    # a second run with the same id must be refused (no overwrites)
    try:
        cli.main(["run", "--run-id", "t1", "--no-probe", "--n-boot", "1"])
        raise SystemExit("expected refusal")
    except AssertionError:
        pass


def test_ingested_dev_sets_are_deterministic_and_leak_free():
    from jevc.schema import check_partition, load_items
    fams = {f: load_items(cli.ingest.EVAL_DIR / f / "dev.jsonl") for f in ("boolq", "clinc")}
    for f, items in fams.items():
        check_partition({"dev": items}, cli._load_pairs(f))
        assert all(i.source.exposure == "overlap_unknown" for i in items)
        assert all(i.gold not in (i.options[0].id,) for i in items) or sum(i.options[0].id == i.gold for i in items) < len(items) * 0.7
    assert not ({i.group_id for i in fams["boolq"]} & {i.group_id for i in fams["clinc"]})
