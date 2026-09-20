"""GLiClass inference: verified preprocessing, single-label softmax, full probability vectors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .schema import Item, Option

MODEL_ID = "knowledgator/gliclass-instruct-base-v1.0"
MODEL_REVISION = "4f6a108b08a5537f395521d19b5073e197923dd3"  # HF API sha, 2026-09-20
MAX_LENGTH = 1024  # gliclass pipeline default; DeBERTa-v3 uses relative positions (position_biased_input=false)


@dataclass
class Runtime:
    model: object
    tokenizer: object
    pipe: object
    device: torch.device
    dtype: str


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu" or (pref == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda:0")


def load_runtime(device: str = "auto") -> Runtime:
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    dev = pick_device(device)
    model = GLiClassModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    # We use the pipeline only for its verified input assembly (`prepare_input`); scoring is done here so that
    # untruncated lengths and full probability vectors are recorded.
    pipe = ZeroShotClassificationPipeline(model, tok, classification_type="single-label", device=str(dev),
                                          max_length=MAX_LENGTH, progress_bar=False).pipe  # inner uni-encoder pipeline
    assert model.config.prompt_first is True and model.config.architecture_type == "uni-encoder"
    assert type(pipe).__name__ == "UniEncoderZeroShotClassificationPipeline", type(pipe)
    return Runtime(model, tok, pipe, dev, str(next(model.parameters()).dtype).replace("torch.", ""))


# ---------- variants ----------

def variant_view(item: Item, variant: str) -> tuple[str, str, list[Option]]:
    """Return (question, state, options) for a diagnostic variant."""
    if variant == "full":
        return item.question, item.state, item.options
    if variant == "no_state":
        return item.question, "", item.options
    if variant == "no_question":
        return "", item.state, item.options
    if variant == "reordered":
        return item.question, item.state, list(reversed(item.options))  # always a different order, deterministic
    raise ValueError(variant)


def build_text(state: str, options: list[Option]) -> str:
    """Training-time description format is `label: description` joined by spaces, placed before/after the text.
    We always place it *before* the state so truncation (which cuts the tail) never removes option semantics."""
    desc = " ".join(f"{o.label}: {o.description}" for o in options)
    return f"{desc} {state}".strip()


def encode(rt: Runtime, views: list[tuple[str, str, list[Option]]]):
    raws = [rt.pipe.prepare_input(build_text(state, opts), [o.label for o in opts], None, q or None)
            for q, state, opts in views]
    enc = rt.tokenizer(raws, truncation=True, max_length=MAX_LENGTH, padding="longest", return_tensors="pt")
    full = [len(x) for x in rt.tokenizer(raws, truncation=False)["input_ids"]]
    return enc.to(rt.device), full


@torch.no_grad()
def score_batch(rt: Runtime, views: list[tuple[str, str, list[Option]]]) -> list[dict]:
    enc, full_lens = encode(rt, views)
    n_max = max(len(v[2]) for v in views)
    logits = rt.model(**enc, max_num_classes=n_max).logits.float()
    out = []
    for i, (_, _, opts) in enumerate(views):
        p = torch.softmax(logits[i, : len(opts)], dim=-1).cpu().tolist()
        used = int(enc["attention_mask"][i].sum().item())
        out.append({"probs": dict(zip([o.id for o in opts], p)), "input_tokens": used,
                    "full_tokens": full_lens[i], "truncated": full_lens[i] > MAX_LENGTH})
    return out


def run_items(rt: Runtime, items: list[Item], variant: str = "full", batch_size: int = 8) -> list[dict]:
    """One prediction record per item; probabilities are keyed by stable option id, so reordering maps back."""
    recs = []
    for s in range(0, len(items), batch_size):
        chunk = items[s : s + batch_size]
        views = [variant_view(it, variant) for it in chunk]
        for it, view, sc in zip(chunk, views, score_batch(rt, views)):
            probs = sc["probs"]
            assert all(math.isfinite(v) for v in probs.values()) and abs(sum(probs.values()) - 1) < 1e-4, it.id
            pred = max(probs, key=probs.get)
            recs.append({"item_id": it.id, "group_id": it.group_id, "family": it.family, "variant": variant,
                         "presented_order": [o.id for o in view[2]], "probs": probs, "pred": pred, "gold": it.gold,
                         "correct": pred == it.gold, "confidence": probs[pred], **{k: sc[k] for k in
                         ("input_tokens", "full_tokens", "truncated")}})
    return recs


# ---------- latency / memory probe ----------

def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def probe(rt: Runtime, item_short: Item, item_long: Item, batch_sizes=(1, 4, 16), warmup=3, reps=5) -> list[dict]:
    """Measured only. Model time = forward pass on pre-encoded inputs; e2e = build+tokenize+transfer+forward+softmax."""
    import psutil

    proc = psutil.Process()
    rows = []
    for name, item in (("short", item_short), ("long", item_long)):
        for bs in batch_sizes:
            views = [variant_view(item, "full")] * bs
            enc, _ = encode(rt, views)
            n = len(item.options)
            row = {"device": str(rt.device), "input": name, "batch": bs, "seq_len": int(enc["input_ids"].shape[1])}
            if rt.device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(rt.device)
            rss0 = proc.memory_info().rss
            model_t, e2e_t = [], []
            try:
                with torch.no_grad():
                    for _ in range(warmup):
                        rt.model(**enc, max_num_classes=n)
                    _sync(rt.device)
                    for _ in range(reps):
                        _sync(rt.device); t0 = time.perf_counter()
                        rt.model(**enc, max_num_classes=n)
                        _sync(rt.device); model_t.append(time.perf_counter() - t0)
                for _ in range(reps):
                    _sync(rt.device); t0 = time.perf_counter()
                    score_batch(rt, views)
                    _sync(rt.device); e2e_t.append(time.perf_counter() - t0)
            except Exception as e:  # torch raises AcceleratorError/OutOfMemoryError; a measured OOM is a result, not a crash
                if "out of memory" not in str(e).lower():
                    raise
                if rt.device.type == "cuda":
                    torch.cuda.empty_cache()
                rows.append({**row, "error": "out of memory", "model_ms_median": None, "e2e_ms_median": None,
                             "gpu_peak_alloc_mb": None, "gpu_peak_reserved_mb": None, "process_rss_mb": None, "rss_delta_mb": None})
                continue
            rows.append({**row, "model_ms_median": 1000 * sorted(model_t)[len(model_t) // 2],
                         "e2e_ms_median": 1000 * sorted(e2e_t)[len(e2e_t) // 2],
                         "gpu_peak_alloc_mb": (torch.cuda.max_memory_allocated(rt.device) / 2**20) if rt.device.type == "cuda" else None,
                         "gpu_peak_reserved_mb": (torch.cuda.max_memory_reserved(rt.device) / 2**20) if rt.device.type == "cuda" else None,
                         "process_rss_mb": proc.memory_info().rss / 2**20, "rss_delta_mb": (proc.memory_info().rss - rss0) / 2**20})
    return rows
