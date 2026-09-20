"""Item / pair schema, validation, JSONL IO, grouped partitioning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

FAMILIES = {"boolq", "clinc"}
OUTPUT_TYPES = {"boolean", "choice"}
EXPOSURE = {"known_overlap", "previously_benchmarked", "overlap_unknown"}
POOLS = ("train", "dev", "cal", "test")


@dataclass
class Option:
    id: str
    label: str  # model-visible label text
    description: str  # model-visible description


@dataclass
class Source:
    dataset: str
    config: str
    revision: str
    split: str
    row_id: int
    license: str
    exposure: str


@dataclass
class Item:
    id: str
    group_id: str
    family: str
    output_type: str
    template_id: str
    domain: str
    state: str
    question: str
    options: list[Option]
    gold: str
    source: Source
    pool: str = "dev"
    notes: str = ""

    def validate(self) -> None:
        assert self.id and self.group_id, "id/group_id required"
        assert self.family in FAMILIES, f"bad family {self.family}"
        assert self.output_type in OUTPUT_TYPES, f"bad output_type {self.output_type}"
        assert self.template_id and self.domain, "template_id/domain required"
        assert self.pool in POOLS, f"bad pool {self.pool}"
        assert isinstance(self.question, str) and isinstance(self.state, str)
        assert len(self.options) >= 2, "need >=2 options"
        ids = [o.id for o in self.options]
        assert len(set(ids)) == len(ids), f"duplicate option ids in {self.id}"
        labels = [o.label for o in self.options]
        assert len(set(labels)) == len(labels), f"duplicate option labels in {self.id}"
        assert all(o.label and o.description for o in self.options), "options need label+description"
        assert self.gold in ids, f"gold {self.gold} not in options for {self.id}"
        if self.output_type == "boolean":
            assert len(self.options) == 2
        assert self.source.exposure in EXPOSURE, f"bad exposure {self.source.exposure}"
        assert self.source.dataset and self.source.revision and self.source.split and self.source.license
        assert isinstance(self.source.row_id, int)


@dataclass
class Pair:
    pair_id: str
    kind: str  # "state_flip" | "question_flip"
    parent_group: str
    intended_change: str
    verifier: str | None
    status: str  # "verified" | "candidate" | "rejected"
    notes: str
    a: Item
    b: Item

    def validate(self) -> None:
        assert self.pair_id and self.kind in {"state_flip", "question_flip"}
        assert self.status in {"verified", "candidate", "rejected"}
        self.a.validate()
        self.b.validate()
        assert self.a.family == self.b.family
        assert self.a.group_id == self.b.group_id == self.parent_group, "pair halves must share parent group"
        assert self.a.gold != self.b.gold, "a label-flip pair must change the gold"
        assert [o.id for o in self.a.options] == [o.id for o in self.b.options], "same candidate set"
        if self.kind == "state_flip":
            assert self.a.question == self.b.question and self.a.state != self.b.state
        else:
            assert self.a.state == self.b.state and self.a.question != self.b.question


# ---------- construction helpers ----------

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def stable_hash(*parts: str, n: int = 16) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:n]


def order_options(options: list[Option], item_id: str, seed: int) -> list[Option]:
    """Deterministic per-item permutation. Sort key is a hash, so gold is never systematically first."""
    return sorted(options, key=lambda o: stable_hash(str(seed), item_id, o.id))


def assign_pool(group_id: str, seed: int, fractions: dict[str, float]) -> str:
    """Hash-based deterministic assignment of a *group* to a pool."""
    assert abs(sum(fractions.values()) - 1.0) < 1e-9
    u = int(stable_hash(str(seed), group_id, n=12), 16) / 16**12
    acc = 0.0
    for pool, frac in fractions.items():
        acc += frac
        if u < acc:
            return pool
    return list(fractions)[-1]


# ---------- IO ----------

def item_from_dict(d: dict) -> Item:
    d = dict(d)
    d["options"] = [Option(**o) for o in d["options"]]
    d["source"] = Source(**d["source"])
    return Item(**d)


def pair_from_dict(d: dict) -> Pair:
    d = dict(d)
    d["a"], d["b"] = item_from_dict(d["a"]), item_from_dict(d["b"])
    return Pair(**d)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r) if hasattr(r, "__dataclass_fields__") else r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(path: Path) -> list[Item]:
    items = [item_from_dict(d) for d in read_jsonl(path)]
    for it in items:
        it.validate()
    return items


def load_pairs(path: Path) -> list[Pair]:
    pairs = [pair_from_dict(d) for d in read_jsonl(path)]
    for p in pairs:
        p.validate()
    return pairs


def check_partition(items_by_pool: dict[str, list[Item]], pairs: list[Pair] = ()) -> None:
    """No group may appear in two pools; pair ancestry must stay inside one pool."""
    seen: dict[str, str] = {}
    for pool, items in items_by_pool.items():
        for it in items:
            assert it.pool == pool, f"{it.id} says pool={it.pool} but stored in {pool}"
            prev = seen.setdefault(it.group_id, pool)
            assert prev == pool, f"group {it.group_id} leaks across {prev}/{pool}"
    for p in pairs:
        pool = seen.get(p.parent_group)
        assert pool is None or pool == p.a.pool == p.b.pool, f"pair {p.pair_id} crosses pools"


def finite_normalized(probs: list[float], tol: float = 1e-4) -> bool:
    return all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs) and abs(sum(probs) - 1.0) < tol
