"""Deterministic development-only ingestion of BoolQ and CLINC150 probe sets."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from .schema import Item, Option, Source, assign_pool, normalize_text, order_options, stable_hash

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL_DIR = ROOT / "eval" / "dev" / "families"

SEED = 20260920
# Pool fractions are applied to *groups*; Milestone 1A materializes only the dev pool.
POOL_FRACTIONS = {"dev": 0.4, "cal": 0.2, "test": 0.4}

# Revisions pinned on 2026-09-20 from the Hugging Face API (`sha` field).
BOOLQ = dict(dataset="google/boolq", config="default", revision="35b264d03638db9f4ce671b711558bf7ff0f80d5",
             split="validation", license="cc-by-sa-3.0", exposure="overlap_unknown")
CLINC = dict(dataset="clinc/clinc_oos", config="plus", revision="155b9c710419136e17307b80d0a13e68cd46b4ec",
             split="validation", license="cc-by-3.0", exposure="overlap_unknown")

BOOLQ_OPTIONS = [
    Option("yes", "yes", "the passage supports answering the question with yes"),
    Option("no", "no", "the passage does not support a yes; the answer is no"),
]
CLINC_QUESTION = "Which intent does the user's message express?"


def _load(spec: dict):
    return load_dataset(spec["dataset"], spec["config"], split=spec["split"], revision=spec["revision"])


def _source(spec: dict, row_id: int) -> Source:
    return Source(spec["dataset"], spec["config"], spec["revision"], spec["split"], row_id, spec["license"], spec["exposure"])


def _pick_groups(groups: dict[str, list], target: int) -> list[str]:
    """Deterministically take whole groups (hash order) until >= target rows."""
    out, n = [], 0
    for g in sorted(groups, key=lambda g: stable_hash("pick", str(SEED), g)):
        out.append(g)
        n += len(groups[g])
        if n >= target:
            break
    return out


def ingest_boolq(target: int = 50) -> list[Item]:
    ds = _load(BOOLQ)
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(ds):
        gid = "boolq:" + stable_hash(normalize_text(row["passage"]))
        if assign_pool(gid, SEED, POOL_FRACTIONS) == "dev":
            groups.setdefault(gid, []).append(i)
    items = []
    for gid in _pick_groups(groups, target):
        for i in groups[gid]:
            row = ds[i]
            iid = f"boolq-{i}"
            items.append(Item(
                id=iid, group_id=gid, family="boolq", output_type="boolean", template_id="boolq_v1",
                domain="wikipedia", state=row["passage"], question=row["question"].strip().rstrip("?") + "?",
                options=order_options(BOOLQ_OPTIONS, iid, SEED), gold="yes" if row["answer"] else "no",
                source=_source(BOOLQ, i), pool="dev"))
    for it in items:
        it.validate()
    return items


def clinc_candidate_groups() -> tuple[dict[str, list[str]], dict[str, str]]:
    domains = json.loads((DATA / "clinc_domains.json").read_text(encoding="utf-8"))
    desc = json.loads((DATA / "clinc_intent_descriptions.json").read_text(encoding="utf-8"))
    desc.pop("_meta", None)
    assert set(desc) == {i for v in domains.values() for i in v}, "descriptions must cover exactly the 150 intents"
    return domains, desc


def ingest_clinc(per_domain: int = 5) -> list[Item]:
    domains, desc = clinc_candidate_groups()
    intent_domain = {i: d for d, v in domains.items() for i in v}
    ds = _load(CLINC)
    names = ds.features["intent"].names
    groups: dict[str, dict[str, list[int]]] = {d: {} for d in domains}
    for i, row in enumerate(ds):
        intent = names[row["intent"]]
        if intent == "oos":
            continue
        gid = "clinc:" + stable_hash(normalize_text(row["text"]))
        if assign_pool(gid, SEED, POOL_FRACTIONS) == "dev":
            groups[intent_domain[intent]].setdefault(gid, []).append(i)
    items = []
    for domain in domains:  # fixed candidate group = every intent of the gold intent's domain, gold-independent
        options = [Option(intent, intent.replace("_", " "), desc[intent]) for intent in domains[domain]]
        for gid in _pick_groups(groups[domain], per_domain):
            for i in groups[domain][gid]:
                row = ds[i]
                iid = f"clinc-{i}"
                items.append(Item(
                    id=iid, group_id=gid, family="clinc", output_type="choice", template_id="clinc_domain_v1",
                    domain=domain, state=row["text"], question=CLINC_QUESTION,
                    options=order_options(options, iid, SEED), gold=names[row["intent"]],
                    source=_source(CLINC, i), pool="dev"))
    for it in items:
        it.validate()
    return items
