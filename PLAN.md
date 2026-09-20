# Open Jev-style decision model: evaluation-first plan (v0.1, 2026-09-20)

Deliverable of milestone 1: a frozen, versioned evaluation package (`eval/v0.1.0/`) and a base-model report. No training spend before the report exists.

## 0. Fixed facts this plan depends on

| Fact | Value | Source |
|---|---|---|
| Base checkpoint | `knowledgator/gliclass-instruct-base-v1.0`, ~0.2B params, Apache-2.0 | HF model card |
| Its fine-tuning data | `tau/commonsense_qa`, `knowledgator/gliclass-v3-logic-dataset` (built from FineWeb, CommonsenseQA, MultiNLI, GLiClass-2k), `BioMike/formal-logic-reasoning-gliclass-2k` | HF model card, dataset card |
| Parent pretraining data | `MoritzLaurer/synthetic_zeroshot_mixtral_v0.1` (synthetic) | gliclass-base-v1.0 card |
| Parent zero-shot benchmark sets (not training) | imdb, ag_news, emotions, CR, sst2, sst5, 20NG, spam, rotten_tomatoes, massive, banking, yahoo, financial_phrasebank, capsotu | gliclass-base-v1.0 card |
| Input construction (uni-encoder) | `<<LABEL>>l1<<LABEL>>l2...<<SEP>>{prompt}{text}` | gliclass/pipeline.py |
| Single-label scoring | softmax over per-label logits | gliclass/pipeline.py |
| Pipeline call | `pipe(texts, labels, threshold, batch_size, classification_type, examples, prompt)` | gliclass/pipeline.py |
| Training format | JSON list of `{"text","all_labels","true_labels"}`; `true_labels` may be `{label: prob}` soft targets | GLiClass README |
| train.py defaults | encoder_lr 1e-5, others_lr 3e-5, batch 8, epochs 3, max_length 1024, focal loss, optional EWC | train.py |
| Local hardware | RTX 4050 Laptop 6 GB, Python 3.14 system (too new for torch; use uv-managed 3.11) | this machine |

Contamination rule derived from the above: **exclude MultiNLI, CommonsenseQA, and anything derived from them. Avoid NLI-style sources entirely for verification (SNLI/ANLI are separate corpora but same task genre).** Parent benchmark sets are permitted but flagged in the manifest as `base_benchmarked: true`.

## 1. Repo layout

```
jev-clone/
  PLAN.md
  pyproject.toml                 # uv project, python 3.11
  src/jevc/
    schema.py                    # Item / Option / Pair dataclasses + JSON (de)serialization + validation
    ingest/<family>.py           # one module per family: source -> list[Item]
    templates.py                 # 5 question templates + option-description variants per family
    partition.py                 # balance + deterministic split assignment
    counterfactual.py            # candidate generation prompts, CSV queue I/O, acceptance
    variants.py                  # ablation / relabel / reorder variants from frozen items
    infer_gliclass.py            # Item -> probabilities via GLiClass; latency harness
    metrics.py                   # all metrics + bootstrap
    report.py                    # per-family tables -> reports/<model>/<eval-version>/
    freeze.py                    # writes manifest.json with sha256 of every file
  eval/v0.1.0/
    manifest.json
    families/<family>/{dev,cal,test}.jsonl
    families/<family>/pairs_{dev,test}.jsonl
    families/<family>/verification_queue.csv     # human-verified, status column
    variants/<family>/{state_ablated,question_ablated,opaque_labels,reordered,distractor,dropped_gold}_{split}.jsonl
    DECISION_RULE.md
  reports/<model_id>/<eval-version>/{metrics.json, per_family.md, reliability/*.png, latency.json}
  train/                         # milestone 2 only
```

## 2. Environment (exact)

```bash
uv init --python 3.11
uv add gliclass transformers torch datasets scikit-learn numpy pandas pyyaml matplotlib
uv add --dev pytest
```
Pin `torch` to a CUDA 12.x wheel index for the 4050. A CPU wheel is acceptable for base eval: ~0.2B model, 10 families x ~1.2k items = ~12k forward passes, expect under 30 min on CPU and under 3 min on the 4050.

## 3. Item schema

```json
{
  "id": "clinc.t3.000412",
  "family": "clinc_intent",
  "output_type": "choice | boolean | score",
  "template_id": "t3",
  "domain": "travel",
  "state": "...user utterance / passage / summary...",
  "question": "Which intent does this message express?",
  "options": [{"id": "book_flight", "desc": "User wants to book a flight"}],
  "level_values": null,
  "gold": "book_flight",
  "pair_id": null,
  "pair_role": null,
  "source": "clinc_oos", "source_id": "train:1234", "license": "CC-BY-3.0",
  "base_benchmarked": false
}
```
Rules: `options` length 2 for boolean, 3 to 8 for choice, equal to the number of rubric levels for score with `level_values` such as `[1,2,3,4,5]`. `gold` is an option id. `pair_role` is `a` or `b` for counterfactual halves. Validation in `schema.py` rejects any item violating these.

## 4. Families (10)

| # | family | type | source (HF id) | options per item | license (verify at ingest) | contamination flag |
|---|---|---|---|---|---|---|
| F1 | clinc_intent | choice | `clinc_oos` (plus) | gold + 4 sampled from same domain; `oos` rows used only as distractor variant | CC-BY-3.0 | none |
| F2 | banking_intent | choice | `banking77` | gold + 5 sampled from same coarse group (hand-map 77 to 9 groups) | CC-BY-4.0 | base_benchmarked |
| F3 | boolq | boolean | `boolq` | yes / no | CC-BY-SA-3.0 | none |
| F4 | vitaminc_verify | choice | `tals/vitaminc` | supports / refutes / not enough info | verify | none; contrastive by construction, supplies built-in pairs |
| F5 | toxicity | boolean | `civil_comments` (toxicity >= 0.5 is yes, <= 0.1 is no, drop middle) | yes / no | CC0 | none |
| F6 | sms_spam | boolean | `sms_spam` | spam / ham | verify (UCI) | base_benchmarked |
| F7 | tweet_sentiment | choice | `tweet_eval` config `sentiment` | negative / neutral / positive | verify | none |
| F8 | summ_coherence | score | SummEval expert annotations (coherence, mean of 3 raters, rounded) | 5 levels, values 1 to 5 | verify (MIT expected) | none |
| F9 | urgency | choice | constructed (Claude-generated, 100% human-verified) | low / medium / high | ours, CC-BY-4.0 | constructed, disclosed |
| F10 | paws_paraphrase | boolean | `paws` config `labeled_final` | yes / no | verify | none |

If a license forbids redistribution, the family stays in the eval but the package ships an ingest script plus hashes rather than the rows.

## 5. Ingestion rules (per family)

Common: strip whitespace, drop states under 3 tokens or over 400 words (keep one length-bucket list per family for latency tests), dedupe by exact normalized text, then dedupe near-duplicates with MinHash (5-gram shingles, Jaccard >= 0.8) across the whole family before splitting. Record `source_id` for provenance.

- F1: options = gold intent + 4 others sampled without replacement from the same domain using `random.Random(sha256(source_id))`. Descriptions: one sentence per intent, hand-written once in `templates.py` (150 lines).
- F2: hand-map 77 intents to 9 coarse groups; sample 5 distractors within group. Descriptions hand-written (77 lines).
- F3: state = passage; question = the BoolQ question text; options yes/no with descriptions like "The passage supports answering yes".
- F4: state = evidence; question = "Does the evidence support the claim: {claim}?"; options supports/refutes/NEI. Keep VitaminC's own contrastive pairs tagged with `pair_id` for the pairs file.
- F5: state = comment. Balance to 50/50 after thresholding.
- F6: state = SMS.
- F7: state = tweet.
- F8: state = source article (truncated to 350 words) + summary; question = rubric text for coherence; options = 5 level descriptions taken from the SummEval annotation guidelines; gold = round(mean expert coherence).
- F9: generated per section 7 with three domains (support tickets, ops alerts, internal chat), 3 levels, 5 templates.
- F10: state = "Sentence 1: ... Sentence 2: ...".

## 6. Templates

Five question phrasings per family (`t1..t5`), written by hand, each with a paired option-description variant. Rules: no shared 4-gram between any two templates of a family (checked by test); t5 is the most unlike the others (imperative form). Template assignment: t1 to t3 train-eligible (milestone 2), t4 dev and cal, t5 test. This gives the paraphrase-held-out level (L1) for free.

Opaque-label variant (L2): option ids replaced by `A, B, C...` in random order (seeded per item), descriptions kept. This is a variant file, not a separate family.

## 7. Balancing and partitioning (exact algorithm)

1. Group items by `(family, template_id)`.
2. Within each group, downsample to `n_min = min(count per gold option)`, capped at the quota. Score family: balance across the 5 levels the same way.
3. Quotas per family: test 500 on t5, dev 250 on t4, cal 250 on t4 (disjoint from dev), train-eligible pool on t1 to t3: everything left, capped 3000.
4. Split assignment: `bucket = int(sha256(salt + source_id).hexdigest(), 16) % 100`; under 50 train-pool, 50 to 62 dev, 63 to 75 cal, 76 and above test; salt fixed in manifest. Assignment is done on `source_id` **before** template attachment so no source row appears in two partitions.
5. Domain hold-out (L4): F1, F2, F9 have domains; hold one domain out of the train pool entirely, present only in test. Record which.
6. Write `splits.json`: `source_id -> partition`, plus counts table.

## 8. Counterfactual pairs

Quotas: 50 verified pairs per family in test, 20 in dev. 10 families gives 500 test pairs and 200 dev pairs. F4 pairs come from VitaminC itself (still verified). F9 pairs are generated with the items.

Kinds, equal thirds where possible:
- `state_flip`: same question and options; minimal state edit; gold changes.
- `question_flip`: same state and options; question changes so that a different option is correct (only where the state supports two questions: F3, F4, F8, F9; skip elsewhere).
- `invariance`: same state and question; options reordered and ids renamed; gold maps to the same description. Generated deterministically, no human verification needed, scored as consistency not accuracy.

Generation: for each seed test item, prompt Claude with the item JSON and the instruction "Produce the smallest edit to `state` such that the correct option becomes `{target}`; return JSON with `state_b`, `gold_b`, `edit_rationale`." Target chosen uniformly among non-gold options.

Verification queue CSV columns: `pair_id, family, kind, state_a, gold_a, state_b, gold_b, verifier, status{approved,rejected,edited}, corrected_gold_b, notes`. A pair enters `pairs_test.jsonl` only with status approved or edited. Every test pair is verified; dev pairs may be sampled at 50%. Budget: ~30 s per pair, ~4.5 h for 500 test pairs. Rejection rate above 40% for a family means the generation prompt is rewritten before continuing.

Scoring: pair correct iff both halves correct. Report pair accuracy, and also half-A accuracy vs half-B accuracy (B is the edited half; a gap there indicates edit-artifact sensitivity).

## 9. Ablation variants (generated from frozen test, deterministic)

| variant | change | what it measures | expected for a state-reading model |
|---|---|---|---|
| state_ablated | `state = ""` | question+options-only baseline | near chance on balanced data |
| question_ablated | `question = ""` | question conditioning | drop relative to full |
| opaque_labels | ids to A/B/C | label-name prior | small drop |
| reordered | options shuffled, ids kept | position bias | identical predictions |
| distractor | append `none_of_the_above` | mass movement | little mass moves when gold present |
| dropped_gold | remove gold option | mass movement | entropy rises, mass on `none` if present |

## 10. Base-model inference (`infer_gliclass.py`)

- Load with `classification_type='single-label'` for every family. Labels passed as `option.desc` strings (not ids); id-to-desc map kept in memory so results are keyed by id. Test both `labels=ids` and `labels=descs` on dev once, freeze the better in manifest as `label_mode`.
- `prompt = question`. The pipeline places prompt before text (`<<SEP>>{prompt}{text}`); end the prompt with a newline so the state is separated.
- Probabilities: take the pipeline's softmax scores over the supplied labels. Store the full vector per item.
- Score family: `pred_score = sum(p_level * level_value)`; also store argmax level.
- Batch size 16; max_length 512 (longer F8 states truncated; note truncation count).
- Latency harness: 1, 4, 16 questions against the same state at state lengths {64, 256, 512} tokens, 50 repeats, report median and p95 for both "one call per question" and "batched call". Run on the 4050 and on CPU. This establishes whether shared-state efficiency is even a problem before any architecture change.

## 11. Metrics (`metrics.py`)

Per family, per split, per variant:
- accuracy; macro-F1 (choice/boolean); MAE and Spearman for score.
- log loss = -mean log p(gold); Brier = mean sum over k of (p_k - 1[k=gold])^2.
- calibration: adaptive-bin ECE with 10 equal-mass bins; reliability diagram; stratified by number of options.
- risk-coverage: sort by max-prob descending; risk(c) = error rate among top-c fraction; report AURC and error at coverage 0.5 and 0.8.
- pair accuracy (section 8); invariance consistency = fraction of reordered items with same argmax and max |delta p| < 0.02.
- state contribution = acc(full) - acc(state_ablated).
- Uncertainty: 1000-resample bootstrap over items within a family, 95% CI on every number. No pooled cross-family CI; a descriptive mean across families may be printed, labeled "descriptive".

## 12. Freeze and versioning

`freeze.py` writes `manifest.json`: eval version, git commit, salt, per-file sha256, per-family counts by partition and gold option, source dataset ids + revisions + licenses, `base_benchmarked` flags, `label_mode`, template list. Tag `eval-v0.1.0`. Rule set in `DECISION_RULE.md` in the same commit.

Isolation rules: `report.py` refuses to run on `test` unless invoked with `--final --eval-version X`, and writes results to `reports/.../final/` which is git-tracked and append-only (a test result file may not be overwritten; a rerun creates a suffixed file). Model selection scripts (milestone 2) can only read `dev` and `cal`. Any change to test items produces `eval/v0.2.0`, old results kept, change log in manifest.

## 13. Pre-registered decision rules (fix numbers before freeze; proposals below)

Milestone-1 gate (is there a gap worth training for): proceed to training iff base macro-F1 (or MAE for F8) on **dev** is below 0.85 on at least 4 families, or pair accuracy below 0.70 on at least 4 families. If base is at or above 0.95 on all families, stop and report "base suffices"; ship an inference wrapper only.

Milestone-2 gate (transfer claim), evaluated on held-out families only, mean over 3 seeds:
- fine-tuned minus base macro-F1 >= +3 points in at least half of held-out families with bootstrap CI excluding 0;
- state_ablated accuracy <= chance + 5 points in every family;
- pair accuracy >= base pair accuracy in every held-out family (regression guard);
- ECE on held-out families <= base ECE + 0.02 after temperature fit on `cal` of training families only.
Failing any: publish as "specialized classifier for families X, Y" with the per-family table, not as a general Jev alternative.

## 14. Milestone 2 outline (only after the section 13 gate passes)

- Data: train pool t1 to t3 from training families, mixed 20% with `knowledgator/gliclass-v3-logic-dataset` rows as replay to limit forgetting. Format: `text = question + "\n" + state`, `all_labels = descs`, `true_labels = [gold desc]` (or `{desc: p}` soft targets if a permitted teacher exists). Confirm in `gliclass/data_processing.py` how `labels_desc_path` is consumed before choosing between ids and descs.
- Config: `problem_type single_label_classification`, encoder_lr 1e-5, others_lr 3e-5, batch 16 (fp16), epochs 3, max_length 512, warmup 0.05, EWC off in pilot.
- Folds (leave-families-out), 3 folds x 3 seeds = 9 runs: hold out {F4,F8,F9}, {F3,F10,F5}, {F1,F7,F6}; F2 always in train (base_benchmarked, keeps intent coverage).
- Pilot: 300 steps locally on the 4050, measure it/s, extrapolate. Expected: ~14k items x 3 epochs / 16 = ~2.6k steps, about 10 to 15 min on an A40. 9 runs is about 2.5 GPU-hours, about **$1.50 on A40**. Runpod is a convenience, not a requirement; the 4050 can run the whole grid overnight.
- Calibration: temperature fit on `cal` of training families; applied unchanged to held-out families.

## 15. Effort and order

1. Day 1: pyproject, schema, F3/F5/F6/F10 ingest (simplest), partition, freeze skeleton, base run on those 4, first numbers.
2. Day 2: F1/F2/F7 ingest + hand-written descriptions, templates for all, F4 with built-in pairs.
3. Day 3: F8, F9 generation, counterfactual generation for all families, queue CSVs out.
4. Day 4 to 5: human verification (~5 h), variants, full base report, latency harness, freeze `eval-v0.1.0`, write DECISION_RULE.md with final numbers.
Milestone-1 GPU spend: $0.

## 16. To verify before coding the affected module

- Licenses marked "verify" in section 4; drop or hash-only any NC/ND source.
- Whether `GLiClassDataset` accepts a prompt field or requires prompt-in-text (affects section 14 only).
- SummEval availability on HF with expert scores; fallback: use relevance instead of coherence, or defer F8 to milestone 2.
- TypeSafe API terms regarding distillation (blocks soft-target training only; nothing in milestone 1 depends on it).
