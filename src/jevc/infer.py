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


INPUT_MODES = ("official_labels", "described_text", "described_labels", "pairwise_descriptions")


def build_text(state: str, options: list[Option], input_mode: str = "described_text") -> str:
    """Training-time description format is `label: description` joined by spaces, placed before/after the text.
    We always place it *before* the state so truncation (which cuts the tail) never removes option semantics."""
    if input_mode == "official_labels":
        return state
    if input_mode == "described_text":
        desc = " ".join(f"{o.label}: {o.description}" for o in options)
        return f"{desc} {state}".strip()
    if input_mode in {"described_labels", "pairwise_descriptions"}:
        return state
    raise ValueError(f"unknown input mode: {input_mode}")


def model_labels(options: list[Option], input_mode: str) -> list[str]:
    if input_mode in {"official_labels", "described_text"}:
        return [o.label for o in options]
    if input_mode in {"described_labels", "pairwise_descriptions"}:
        return [o.description for o in options]
    raise ValueError(f"unknown input mode: {input_mode}")


def encode(rt: Runtime, views: list[tuple[str, str, list[Option]]], input_mode: str = "described_text"):
    raws = [rt.pipe.prepare_input(build_text(state, opts, input_mode), model_labels(opts, input_mode), None, q or None)
            for q, state, opts in views]
    enc = rt.tokenizer(raws, truncation=True, max_length=MAX_LENGTH, padding="longest", return_tensors="pt")
    full = [len(x) for x in rt.tokenizer(raws, truncation=False)["input_ids"]]
    return enc.to(rt.device), full


@torch.no_grad()
def score_batch(rt: Runtime, views: list[tuple[str, str, list[Option]]], input_mode: str = "described_text") -> list[dict]:
    if input_mode == "pairwise_descriptions":
        return score_pairwise_batch(rt, views)
    enc, full_lens = encode(rt, views, input_mode)
    n_max = max(len(v[2]) for v in views)
    logits = rt.model(**enc, max_num_classes=n_max).logits.float()
    out = []
    for i, (_, _, opts) in enumerate(views):
        p = torch.softmax(logits[i, : len(opts)], dim=-1).cpu().tolist()
        used = int(enc["attention_mask"][i].sum().item())
        out.append({"probs": dict(zip([o.id for o in opts], p)), "input_tokens": used,
                    "full_tokens": full_lens[i], "truncated": full_lens[i] > MAX_LENGTH})
    return out


@torch.no_grad()
def score_pairwise_batch(rt: Runtime, views: list[tuple[str, str, list[Option]]]) -> list[dict]:
    """Score each option in isolation, then normalize its raw logit across that item's options.

    This is deliberately an audit view, not a claim that the model learned a joint choice
    distribution. Its value is removing candidate ordering and inter-option attention.
    """
    flat: list[tuple[int, tuple[str, str, list[Option]]]] = []
    for item_idx, (question, state, options) in enumerate(views):
        flat.extend((item_idx, (question, state, [option])) for option in options)
    enc, full_lens = encode(rt, [view for _, view in flat], "pairwise_descriptions")
    logits = rt.model(**enc, max_num_classes=1).logits[:, 0].float().cpu()
    grouped: list[list[tuple[Option, float, int]]] = [[] for _ in views]
    for (item_idx, (_, _, options)), logit, length, mask in zip(flat, logits, full_lens, enc["attention_mask"]):
        grouped[item_idx].append((options[0], float(logit), length, int(mask.sum().item())))
    out = []
    for rows in grouped:
        scores = torch.softmax(torch.tensor([row[1] for row in rows]), dim=0).tolist()
        out.append({"probs": {row[0].id: score for row, score in zip(rows, scores)},
                    "input_tokens": max(row[3] for row in rows),
                    "full_tokens": max(row[2] for row in rows),
                    "truncated": any(row[2] > MAX_LENGTH for row in rows)})
    return out


def run_items(rt: Runtime, items: list[Item], variant: str = "full", batch_size: int = 8,
              input_mode: str = "described_text") -> list[dict]:
    """One prediction record per item; probabilities are keyed by stable option id, so reordering maps back."""
    recs = []
    for s in range(0, len(items), batch_size):
        chunk = items[s : s + batch_size]
        views = [variant_view(it, variant) for it in chunk]
        for it, view, sc in zip(chunk, views, score_batch(rt, views, input_mode)):
            probs = sc["probs"]
            assert all(math.isfinite(v) for v in probs.values()) and abs(sum(probs.values()) - 1) < 1e-4, it.id
            pred = max(probs, key=probs.get)
            recs.append({"item_id": it.id, "group_id": it.group_id, "family": it.family, "variant": variant,
                         "input_mode": input_mode,
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
