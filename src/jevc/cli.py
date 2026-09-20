"""CLI: `jevc ingest` builds the dev probe sets; `jevc run` evaluates the untouched checkpoint and writes a report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import ingest, metrics
from .schema import check_partition, load_items, load_pairs, write_jsonl

ROOT = ingest.ROOT
PAIRS_DIR = ROOT / "data" / "pairs"
VARIANTS = ("full", "no_state", "no_question", "reordered")


def cmd_ingest(_):
    for fam, items in (("boolq", ingest.ingest_boolq()), ("clinc", ingest.ingest_clinc())):
        path = ingest.EVAL_DIR / fam / "dev.jsonl"
        write_jsonl(path, items)
        check_partition({"dev": load_items(path)})
        print(f"{fam}: {len(items)} items, {len({i.group_id for i in items})} groups -> {path.relative_to(ROOT)}")


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _env(rt) -> dict:
    import datasets, gliclass, torch, transformers
    cuda = torch.cuda.is_available()
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda, "cuda_available": cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_total_mem_mb": torch.cuda.get_device_properties(0).total_memory / 2**20 if cuda else None,
            "transformers": transformers.__version__, "gliclass": gliclass.__version__, "datasets": datasets.__version__,
            "device_used": str(rt.device), "precision": rt.dtype}


def _load_pairs(fam):
    p = PAIRS_DIR / f"{fam}_pairs_dev.jsonl"
    return load_pairs(p) if p.exists() else []


def cmd_run(a):
    from . import infer

    out = ROOT / "reports" / a.run_id
    assert not out.exists(), f"run dir exists: {out}"
    out.mkdir(parents=True)
    fams = {f: load_items(ingest.EVAL_DIR / f / "dev.jsonl") for f in ("boolq", "clinc")}
    pairs = {f: _load_pairs(f) for f in fams}
    for f in fams:
        check_partition({"dev": fams[f]}, pairs[f])
        for p in pairs[f]:
            assert p.parent_group in {i.group_id for i in fams[f]}, f"pair {p.pair_id} parent not in dev set"

    rt = infer.load_runtime(a.device)
    run_info = {"run_id": a.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "mode": "dev",
                "model": infer.MODEL_ID, "model_revision": infer.MODEL_REVISION, "tokenizer_revision": infer.MODEL_REVISION,
                "max_length": infer.MAX_LENGTH, "batch_size": a.batch_size, "classification_type": "single-label",
                "seeds": {"ingest": ingest.SEED, "bootstrap": 0, "reorder": "reversed presentation order"},
                "code_commit": _git("rev-parse", "HEAD"), "code_dirty": bool(_git("status", "--porcelain")),
                "env": _env(rt), "paid_spend_usd": 0.0}

    preds, results = [], {}
    for f, items in fams.items():
        by_var = {v: infer.run_items(rt, items, v, a.batch_size) for v in VARIANTS}
        for v in VARIANTS:
            preds += by_var[v]
        base = by_var["full"]
        res = {"base": metrics.summary(base), "bootstrap": metrics.bootstrap(base, a.n_boot, 0),
               "variants": {v: metrics.summary(by_var[v]) for v in VARIANTS if v != "full"},
               "ablation_full_minus_variant_accuracy": {v: metrics.accuracy(base) - metrics.accuracy(by_var[v])
                                                        for v in ("no_state", "no_question")},
               "reorder_invariance": metrics.invariance(base, by_var["reordered"]),
               "input_tokens": {"max": max(r["input_tokens"] for r in base), "mean": sum(r["input_tokens"] for r in base) / len(base),
                                "n_truncated": sum(r["truncated"] for r in base)},
               "failures": [{k: r[k] for k in ("item_id", "gold", "pred", "confidence", "truncated")}
                            for r in sorted(base, key=lambda r: -r["confidence"]) if not r["correct"]][:30]}
        # pairs: both halves run as full items, scored by status bucket
        pr = {}
        for status in ("verified", "candidate"):
            ps = [p for p in pairs[f] if p.status == status]
            if ps:
                ra = infer.run_items(rt, [p.a for p in ps], "full", a.batch_size)
                rb = infer.run_items(rt, [p.b for p in ps], "full", a.batch_size)
                for r, p in zip(ra + rb, ps + ps):
                    r["variant"], r["pair_id"] = "pair", p.pair_id
                preds += ra + rb
                pr[status] = {**metrics.pair_scores(list(zip(ra, rb))),
                              "per_pair": [{"pair_id": p.pair_id, "a": x["correct"], "b": y["correct"]} for p, x, y in zip(ps, ra, rb)]}
        res["pairs"] = pr
        results[f] = res

    probe_rows = []
    if not a.no_probe:
        short, long = fams["clinc"][0], max(fams["boolq"], key=lambda i: len(i.state))
        probe_rows += infer.probe(rt, short, long)
        if rt.device.type == "cuda":
            rt.model.to("cpu"); rt.device = infer.torch.device("cpu"); rt.pipe.device = rt.device
            probe_rows += infer.probe(rt, short, long)
    write_jsonl(out / "predictions.jsonl", preds)
    (out / "metrics.json").write_text(json.dumps({"families": results, "probe": probe_rows}, indent=1), encoding="utf-8")
    (out / "run.json").write_text(json.dumps(run_info, indent=1), encoding="utf-8")
    (out / "report.md").write_text(render_report(run_info, results, probe_rows, fams, pairs), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


def _f(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def render_report(run, results, probe_rows, fams, pairs) -> str:
    e = run["env"]
    L = [f"# Development report — run `{run['run_id']}`", "",
         "Development-only probe data. Nothing here is a confirmatory or final-test result, and passing two families is not evidence of cross-family generalization.", "",
         "## Environment and model provenance", "",
         f"- Model `{run['model']}` @ `{run['model_revision']}` (tokenizer same revision), license apache-2.0 (model card). Untouched, no fine-tuning.",
         f"- Backbone per config.json: microsoft/deberta-v3-base uni-encoder, `prompt_first=true`, `problem_type=multi_label_classification`, `class_token_pooling=average`, `use_segment_embeddings=true`.",
         f"- Scoring: `classification_type=single-label` ⇒ softmax over the per-label logits (gliclass `_postprocess_logits`). The checkpoint was trained with a multi-label (sigmoid/focal) objective, so the softmax vector is a renormalization, not a trained categorical head.",
         f"- Input assembly (gliclass `UniEncoderZeroShotClassificationPipeline.prepare_input`): `<<LABEL>>opt1<<LABEL>>opt2…<<SEP>>{{question}}{{descriptions}} {{state}}`; descriptions use the training-time `label: description` form and are placed before the state so tail truncation cannot remove them.",
         f"- max_length {run['max_length']}, batch size {run['batch_size']}, precision {e['precision']}, device `{e['device_used']}`, CUDA available: {e['cuda_available']}" + (f" ({e['gpu']}, {e['gpu_total_mem_mb']:.0f} MB)" if e['cuda_available'] else ""),
         f"- Python {e['python']}, torch {e['torch']} (CUDA build {e['torch_cuda_build']}), transformers {e['transformers']}, gliclass {e['gliclass']}, datasets {e['datasets']}; {e['platform']}",
         f"- Code commit `{run['code_commit']}` (dirty: {run['code_dirty']}); seeds {run['seeds']}; paid spend ${run['paid_spend_usd']:.2f}", "",
         "## Dataset provenance and licensing", ""]
    for f, items in fams.items():
        s = items[0].source
        L.append(f"- **{f}**: `{s.dataset}` config `{s.config}` split `{s.split}` @ `{s.revision}`, license `{s.license}`, exposure `{s.exposure}`; "
                 f"{len(items)} items in {len({i.group_id for i in items})} source groups, all in the dev pool (group-hashed, seed {run['seeds']['ingest']}).")
    L += ["- Exposure is `overlap_unknown` for both: the model card lists tau/commonsense_qa, knowledgator/gliclass-v3-logic-dataset and BioMike/formal-logic-reasoning-gliclass-2k, none of which is BoolQ or CLINC, but the full lineage of the GLiClass v3 training mix is not published, so absence of overlap is not established.",
          "- CLINC candidate groups are the 10 official domains (15 intents each, from clinc/oos-eval `data/domains.json`); each item sees all intents of its gold domain, so candidate construction is independent of the gold label. `oos` is excluded in 1A.", ""]
    for f, r in results.items():
        b, bs = r["base"], r["bootstrap"]
        L += [f"## {f}: base metrics (full input)", "", "| metric | value | 95% CI (group bootstrap, n={}) |".format(bs['n_boot']), "|---|---|---|"]
        for k in metrics.SCALARS:
            ci = bs[k]["ci95"]
            L.append(f"| {k} | {_f(b[k])} | {_f(ci[0]) if ci else 'n/a'} – {_f(ci[1]) if ci else 'n/a'} |")
        L += [f"| n / groups | {b['n']} / {b['n_groups']} | |", f"| truncated inputs | {b['n_truncated']} | max tokens {r['input_tokens']['max']}, mean {r['input_tokens']['mean']:.0f} |", "",
              "Confusion (gold->pred): " + ", ".join(f"{k}: {v}" for k, v in b["confusion"].items()), "",
              "Reliability (equal-mass bins on top-label confidence; ECE is diagnostic only):", "", "| n | conf range | conf mean | acc |", "|---|---|---|---|"]
        L += [f"| {x['n']} | {x['conf_min']:.2f}–{x['conf_max']:.2f} | {x['conf_mean']:.3f} | {x['acc']:.3f} |" for x in b["reliability_bins"]]
        L += ["", f"### {f}: diagnostic variants", "", "| variant | acc | macro_f1 | log_loss | brier | full−variant acc |", "|---|---|---|---|---|---|"]
        for v, s in r["variants"].items():
            d = r["ablation_full_minus_variant_accuracy"].get(v)
            L.append(f"| {v} | {_f(s['accuracy'])} | {_f(s['macro_f1'])} | {_f(s['log_loss'])} | {_f(s['brier'])} | {_f(d) if d is not None else '—'} |")
        inv = r["reorder_invariance"]
        L += ["", f"Reordered options (semantic remap): prediction agreement {_f(inv.get('pred_agreement'))}, mean L1 probability change {_f(inv.get('mean_abs_prob_change_l1'))}, max {_f(inv.get('max_abs_prob_change_l1'))}.", ""]
        L += [f"### {f}: state-flip pairs", ""]
        if not r["pairs"]:
            L.append("No pairs.")
        for status, ps in r["pairs"].items():
            L.append(f"- **{status}** pairs (n={ps['n']}): both-correct {_f(ps['both_correct'])}, half A {_f(ps['a_correct'])}, half B {_f(ps['b_correct'])}. "
                     + ("Candidate pairs were authored in-session and are NOT verified gold; they are reported for pipeline validation only." if status == "candidate" else ""))
            L += [f"  - {x['pair_id']}: a={'✓' if x['a'] else '✗'} b={'✓' if x['b'] else '✗'}" for x in ps["per_pair"]]
        L += ["", f"### {f}: highest-confidence errors (up to 30)", ""]
        L += [f"- {x['item_id']}: gold `{x['gold']}` pred `{x['pred']}` p={x['confidence']:.3f}{' (truncated)' if x['truncated'] else ''}" for x in r["failures"]] or ["none"]
        L.append("")
    L += ["## Latency and memory (measured; no estimates)", ""]
    if probe_rows:
        L += ["Warm-up 3 iterations, median of 5. `model` = forward pass on pre-encoded batch (CUDA-synchronized); `e2e` = build+tokenize+transfer+forward+softmax.", "",
              "| device | input | batch | seq_len | model ms | e2e ms | GPU peak alloc MB | GPU peak reserved MB | process RSS MB |", "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {p['device']} | {p['input']} | {p['batch']} | {p['seq_len']} | {_f(p['model_ms_median'])} | {_f(p['e2e_ms_median'])} | {_f(p['gpu_peak_alloc_mb'])} | {_f(p['gpu_peak_reserved_mb'])} | {_f(p['process_rss_mb'])} |" + (f" {p['error']}" if p.get("error") else "") for p in probe_rows]
    else:
        L.append("Probe skipped.")
    L += ["", "## Known limitations", "",
          "- ~50 items per family: intervals are wide; use them to catch pipeline bugs, not to rank models.",
          "- Softmax over multi-label-trained logits; calibration is measured, not assumed from the model card.",
          "- Pairs are in-session candidates until a human verifies both answers and the intended flip.",
          "- Exposure `overlap_unknown`; no contamination check beyond the model card was possible.",
          "- Question is passed through the gliclass `prompt` slot and descriptions through the text; other label representations were not compared (deferred to 1B, dev-only).",
          "- No final-test data exists or was inspected; `eval-v0.1.0` is not frozen."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jevc")
    sub = ap.add_subparsers(required=True)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--no-probe", action="store_true")
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
