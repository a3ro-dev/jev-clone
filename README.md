# jevc — open Jev-style decision model (Milestone 1A)

Evaluation-first pipeline for the untouched `knowledgator/gliclass-instruct-base-v1.0` checkpoint on two
development-only probe families: BoolQ (boolean, passage-conditioned) and CLINC150 (intent choice with fixed
domain candidate groups). See [PLAN.md](PLAN.md) for the specification.

Nothing here is fine-tuned, frozen, or final-test. Paid spend: $0.

## Setup (Python 3.11 via uv)

```bash
uv sync --python 3.11 --frozen
```

`pyproject.toml` pins torch 2.14.0, transformers 5.17.0, gliclass 0.1.20, datasets 5.0.1; `uv.lock` is the
cross-platform lock. On Windows torch comes from the PyTorch `cu130` index; on Linux the PyPI wheel already bundles
CUDA 13. CPU is used automatically when CUDA is unavailable.

## Runpod (where inference runs)

The local laptop is used only for tests and data preparation. On a Runpod pod (Linux, NVIDIA driver supporting
CUDA 13.0+, any GPU with >= 8 GB), from the repo root:

```bash
bash scripts/runpod.sh
```

That installs uv if needed, syncs the lock, prints the torch/CUDA check, runs the tests, ingests the pinned datasets,
and runs `jevc run` (evaluation + probe). Copy `reports/<run_id>/` back before stopping the pod.

## Run

```bash
uv run pytest -q
```

```bash
uv run jevc ingest
```

```bash
uv run jevc run --run-id dev-$(date -u +%Y%m%d-%H%M%S)
```

`ingest` downloads the pinned dataset revisions and writes `eval/dev/families/{boolq,clinc}/dev.jsonl`
(about 50 items each, whole source groups, dev pool only). `run` evaluates the full input plus three diagnostic
variants (empty state, empty question, deterministically reordered options), scores the hand-written state-flip
pairs in `data/pairs/`, runs the latency/memory probe on GPU (if present) and CPU, and writes
`reports/<run_id>/{run.json,predictions.jsonl,metrics.json,report.md}`. Options: `--device cpu|cuda|auto`,
`--batch-size`, `--n-boot`, `--no-probe`. A run directory is never overwritten. The probe records an out-of-memory
result per (device, input, batch) instead of aborting; on the 6 GB RTX 4050 the batch-16 x ~1024-token case OOMs.

Tests (`uv run pytest -q`) need no model: the end-to-end test drives `jevc run` with a fake scorer.

## Layout

```text
src/jevc/schema.py    item/pair schema, validation, grouped partition, JSONL IO
src/jevc/ingest.py    BoolQ + CLINC dev ingestion (pinned revisions, seeds, candidate groups)
src/jevc/infer.py     GLiClass loading, verified input assembly, single-label softmax, probe
src/jevc/metrics.py   accuracy, macro-F1, confusion, log loss, Brier, reliability/ECE, risk-coverage, bootstrap
src/jevc/cli.py       `jevc ingest` / `jevc run` and the markdown report
data/clinc_domains.json              official CLINC150 domain -> intents map (candidate groups)
data/clinc_intent_descriptions.json  fixed model-visible intent descriptions
data/pairs/*_pairs_dev.jsonl         hand-written development state-flip pairs (candidate status)
eval/dev/families/<family>/dev.jsonl generated development items
tests/                               schema, partition, mapping, metrics, determinism
```

## Provenance (verified 2026-09-20)

| artifact | id / revision | license |
|---|---|---|
| model + tokenizer | `knowledgator/gliclass-instruct-base-v1.0` @ `4f6a108b08a5537f395521d19b5073e197923dd3` | apache-2.0 |
| gliclass code | PyPI 0.1.20 (GitHub main `40baa67c` inspected) | Apache-2.0 |
| BoolQ | `google/boolq` default/validation @ `35b264d03638db9f4ce671b711558bf7ff0f80d5` | cc-by-sa-3.0 |
| CLINC150 | `clinc/clinc_oos` plus/validation @ `155b9c710419136e17307b80d0a13e68cd46b4ec` | cc-by-3.0 |
| CLINC domains | clinc/oos-eval `data/domains.json` @ `828f8093` | (repo, CC-BY-3.0 data) |

Input format (from gliclass source, `prompt_first=true` uni-encoder):
`<<LABEL>>opt1<<LABEL>>opt2…<<SEP>>{question}{"label: description" …} {state}`; single-label scoring is a softmax
over the per-label logits of a checkpoint trained with a multi-label objective.
