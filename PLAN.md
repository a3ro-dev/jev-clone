# Open Jev-style decision model: final implementation plan

Version 1.0 — 2026-09-20

## Objective

Build an open, question-conditioned decision model with typed choices, boolean probabilities, and rubric scores by adapting an existing pretrained model. Do not pretrain from scratch or claim to reproduce Jev's unpublished architecture or RLCD.

Milestone 1 delivers a versioned evaluation package and base-model report. Train only when development evidence identifies a useful, learnable gap. A specialized classifier is a valid release; broad transfer is a separate claim.

This file supersedes previous drafts. Implement the milestones in order; another planning round is not required.

## Budget

- Milestone 1 uses local hardware with no paid GPU or teacher calls. Probe the reported RTX 4050 Laptop's usable memory and throughput; CPU fallback is acceptable. Neither training fit nor runtime is assumed.
- Paid compute and storage combined have a hard cap of **$45**: $5 pilot, $15 initial training, $10 follow-up experiments, $5 final evaluation, $10 storage/contingency. Allocations are ceilings, not forecasts.
- Log actual spend and projected remaining cost before paid runs, including setup, idle time, evaluation, and storage. Preserve checkpoints before terminating temporary compute; persistent storage also costs money. Never automatically expand the cap.
- Existing TypeSafe credit is optional and separate. Distillation requires verification that applicable API terms permit the intended training and release. The project must work without it.
- Claude may assist with critique and candidate examples through a permitted workflow. Do not assume a subscription provides bulk API access or unrestricted output rights. Candidate generation does not establish gold labels.
- Engineering time and ongoing hosting are excluded from the compute budget.

## Starting point and verification

Candidate: `knowledgator/gliclass-instruct-base-v1.0`, reported as approximately 0.2B parameters with Apache-2.0 weights. Verify the exact revision, license, lineage, preprocessing, and probability behavior before use.

Primary references:

- Model: https://huggingface.co/knowledgator/gliclass-instruct-base-v1.0
- Implementation: https://github.com/Knowledgator/GLiClass
- Research: https://arxiv.org/abs/2508.07662
- Target interface: https://docs.typesafe.ai/primitives
- Calibration baseline: https://proceedings.mlr.press/v70/guo17a.html
- Teacher terms: https://typesafe.ai/legal/terms
- Compute prices: https://www.runpod.io/pricing

Use Context7 for current library/API documentation when available, following workspace instructions. Otherwise document its absence and consult official docs/source. Do not reuse unverified API signatures, training defaults, CUDA versions, or runtime estimates from earlier drafts.

Inspect installed runtimes; select a supported Python/PyTorch combination, lock dependencies, and record code, model, tokenizer, and dataset revisions. Verify CUDA with a small inference probe. Check training/inference preprocessing equivalence explicitly.

Exposure labels: `known_overlap`, `previously_benchmarked`, `overlap_unknown`. Exclude known training overlap from confirmatory evaluation. Similar task genres do not imply contamination; incomplete provenance does not establish absence.

## Milestone 1A: two-family development pipeline

Start with BoolQ passage questions and CLINC intent classification, subject to license and source verification. Use approximately 50 development examples per family to prove ingestion, grouped splitting, inference, probability metrics, paired tests, and reporting. These probe examples remain development-only.

Fix schema, truncation, preprocessing, and metric errors here before expanding data. No novel architecture or generation service is needed.

## Milestone 1B: baseline benchmark

| Family | Candidate source | Purpose |
|---|---|---|
| Passage question answering | BoolQ | State- and question-dependent boolean decisions |
| Intent classification | CLINC | Described choices and domain shift |
| Evidence verification | VitaminC | Three-way decisions and linked contrastive examples |
| Toxicity classification | Civil Comments | Distinct boolean task and annotator disagreement |

Target 200 development, 200 calibration, and 300 final-test independent source groups per family where feasible. These are sampling targets, not a power guarantee. Record shortfalls rather than duplicating examples.

Use original label distributions for primary evaluation. Maintain a separate balanced diagnostic subset within compatible question/template/candidate-set groups. Record toxicity thresholds and exclusions before freeze; conclusions apply to the selected population.

After classification works, optionally add SummEval coherence as a score family, subject to annotation and license verification. Retain individual ratings and their unrounded mean. Evaluate numeric predictions against the mean and categorical distributions against empirical rating distributions when available. Do not round the mean into a supposedly definitive class. Until validated, score support is experimental.

Verify use rights and redistribution rights separately. When use is permitted but redistribution is not, publish ingestion instructions, revisions, split identifiers, and hashes instead of restricted rows. Document replacements before freeze.

### Data contract

Each item contains:

- Stable item and source-group IDs, family, output type, template ID, and domain.
- State, question, options with unique IDs and descriptions.
- Gold option or rating targets; numeric rubric values where applicable.
- Source/revision/row provenance, exposure status, and license metadata.

Each pair contains both complete items, pair ID/kind, parent group, intended semantic change, verifier, status, and notes. Every scored semantic pair, including development pairs, must have both answers and its intended label flip verified.

### Partitioning

1. Normalize text and identify exact/near duplicates. Connect related passages, revisions, summaries, and contrastive pairs using provenance. Similarity flags require grouping/review; do not delete meaningful contrasts automatically.
2. Partition connected source groups, never individual row IDs. All paraphrases, edits, and counterfactual descendants follow their parent.
3. Reserve a named CLINC domain as a separate domain-shift test slice before ordinary splitting. Exclude it from training, development, and calibration; disclose that its task specification is known.
4. Deterministically assign remaining groups to train-eligible, development, calibration, and test pools with a recorded seed; sample quotas within each pool.
5. Attach semantically checked wording variants after assignment. Use distinct training, development/calibration, and test wording sets. A no-shared-n-gram rule is not evidence of semantic holdout.

Use predefined candidate sets within intent task groups. Candidate construction must not reveal the gold answer. Randomize option order deterministically in every example; never put gold systematically first. Document restricted candidate-set tasks and explicit rejection options.

### Inference

- Use verified preprocessing to provide state, question, and option semantics. Select label representation on development data only; freeze it and save full probability vectors mapped to option IDs.
- Validate finite normalized probabilities. Record input length, truncation, batch size, precision, and device. Reserve space for questions/options; reject or flag examples whose required evidence is truncated.
- Boolean outputs are per-question probabilities; choices are categorical distributions; scores are expectations over documented level values. Define confidence explicitly. Marginals do not identify the true joint distribution; multiplying them assumes independence.
- Begin with independent question batching. Do not claim shared-state efficiency. Profile warm model time and end-to-end request time separately, including synchronization, tokenization, and transfer as appropriate. Test multiple input lengths and 1/4/16 questions.

### Diagnostic views

| View | Interpretation |
|---|---|
| Empty state | Question/options-only baseline, compared with full input |
| Empty question | Incremental question use; descriptions may already specify the task |
| Reordered options | Prediction and probability stability after semantic remapping |
| Renamed IDs | Adapter integrity if hidden; semantic label sensitivity only if model-visible text changes |
| State-flip pairs | Same question, relevant evidence changes, gold changes |
| Question-flip pairs | Same state, different answer-bearing question where valid |
| Rejection subset | Define none-of-the-above, remove the substantive gold option, verify new gold |

Start with 20 development and 30 final-test semantic pairs per family where valid. Prefer existing or hand-authored candidates; generation is optional. Verify all scored pairs. Score both-correct accuracy and each half separately. Keep invariance transformations separate from label-flip pairs. Pairs passed by base remain regression tests.

Do not impose universal entropy increases or accuracy drops on ablations. Near-chance state-ablated performance requires appropriate conditional balance, not just globally equal class counts.

### Metrics

Report per family and relevant slice. Any cross-family average is descriptive.

- Accuracy, macro-F1, confusion counts, log loss, and multiclass Brier score.
- Equal-mass reliability bins with counts/uncertainty; ECE is a noisy diagnostic, not a sole gate. Stratify by option count and distinguish primary from balanced distributions.
- Risk–coverage curves, AURC, error at 50%/80% coverage. Document ranking confidence. Select operational thresholds on calibration data, then report achieved coverage and error on test.
- For scores: MAE against mean ratings, rank correlation where defined, rating-distribution metrics, annotator disagreement, and documented rubric spacing.
- Pair accuracy, invariance agreement/probability changes, and descriptive full-minus-ablated accuracy.
- 1,000 paired bootstrap resamples over independent source groups within each family; pairs travel with groups. Mark undefined metrics. Later report seed variability separately: item bootstraps do not capture training variability.

## Files, freeze, and test discipline

Keep implementation compact: schema/partitioning, ingestion, inference, metrics/reporting, and freeze utilities. Add modules when responsibilities warrant them, not one per hypothetical feature.

Artifacts:

```text
PLAN.md
pyproject.toml + dependency lock
src/jevc/
eval/v0.1.0/
  manifest.json
  DECISION_RULE.md
  splits.json
  families/<family>/{dev,cal,test,pairs_dev,pairs_test}.jsonl
reports/<run_id>/
  run.json
  predictions.jsonl
  metrics.json
  report.md
```

Use reconstruction manifests where rows cannot be distributed. Variant generation must be deterministic and versioned.

Freeze sequence:

1. Complete development debugging and manual test-data verification without inspecting test-model outputs.
2. Commit data/manifests, splits, templates, variants, metrics, and decision rules together; tag `eval-v0.1.0`.
3. Manifest hashes payload files, excluding itself. Each run records manifest hash, resolved evaluation commit, code commit, revisions, environment, hardware, and seeds. No self-referential hash or commit fields.
4. Verify hashes before inference. Require explicit final-test mode and version; write unique run directories and reject overwrites.
5. Test-data changes require a new evaluation version/change log. Metric fixes require a new evaluator version and disclosed reruns.

A commit provides provenance, not secrecy. After baseline test inspection, those items are fixed regression evidence, not untouched confirmation for subsequent development. Reserve fresh groups for later confirmation whenever inspected results influenced choices. Never select checkpoints on final-test results.

## Milestone-1 decision rule

Use development evidence for the training decision; final testing characterizes the frozen baseline.

- Provisional workflow adequacy targets: macro-F1 >= 0.90, semantic pair accuracy >= 0.85, and error <= 0.05 at >= 0.50 development coverage. Report uncertainty; these are project targets, not production guarantees. Define a separate score target before including a score workflow in the gate.
- If intended workflows meet targets, release the baseline wrapper and report limitations.
- If a workflow misses targets, inspect up to 30 development failures. Proceed to a training pilot only after documenting relevant, independently verified model errors and permitted training data covering that pattern without evaluation leakage. Aim for at least 10 verified errors; fewer support a tentative pilot only with explicit uncertainty.
- If failures stem from labels, candidates, preprocessing, or truncation, repair those before considering training.
- If evidence is ambiguous or training data unavailable, gather more local development evidence or stop with an insufficient-evidence report. Do not automatically rent a GPU.

Record any target adjustments and their rationale before final testing. Lower MAE is better; never apply classification thresholds to score errors.

## Milestone 2: capped specialization experiment

1. Freeze a pilot hypothesis and training recipe from development evidence. Start with supervised learning and the existing architecture; no new RL method.
2. Verify loss and target support in current source. Prefer cross-entropy for categorical labels or soft-target cross-entropy for permitted distributions; do not inherit focal-loss defaults without justification.
3. Measure local training memory and throughput. Choose length, precision, batching, and accumulation from measurements. If necessary, use the paid pilot allocation. Estimate full costs with overhead before launch.
4. Train one pilot. Continue only for useful development gains without material observed regression on predeclared companion tasks. Otherwise diagnose within the pilot cap; do not launch a grid automatically.
5. For a promising fixed recipe, run three seeds if budget permits; otherwise report actual seeds and limited evidence. Fit calibration on calibration data only, for both base and trained comparisons.
6. Before final evaluation, predeclare improvement and regression tolerances. Initial classification improvement target: +3 macro-F1 percentage points on the intended workflow, accompanied by probability-quality and paired-test results. Report all seeds and failures. Use fresh confirmation groups when prior test inspection influenced the recipe.
7. Release only supported specialization claims. Jev disagreement is not automatically error; teacher agreement is a separate diagnostic from independently established correctness.

Optional distillation requires applicable training/release rights, a small audited pilot batch, and provenance. Teacher-label metrics measure imitation, not calibration against real outcomes. Student calibration need not match teacher calibration.

## Milestone 3: optional transfer research

No automatic budget beyond the $45 cap. Add families only after specialization evidence warrants this research.

Separate task-family, domain, and wording holdouts. Two intent datasets are not two unrelated task families. Reserve development families and untouched confirmation families; exclude confirmation families from training, calibration, and recipe selection. Disclose prior inspection.

Compare base and trained models on identical held-out families and pairs. Report per-family deltas, calibration, risk–coverage, and seed variability. Few families support bounded claims about those families, not arbitrary tasks. Absent transfer does not diagnose memorization by itself. Failed transfer narrows claims without invalidating useful specialization.

## Release and immediate next step

Release code, reproducible environment, weights/adapter with required notices, recipe, evaluation protocol, permitted data or reconstruction instructions, results, and limitations. Document supported tasks, score validation status, truncation, confidence definition, latency hardware, data exposure uncertainty, and measured compute cost.

Do not claim no decision errors, Jev parity, deployment calibration, or architectural novelty without supporting evidence.

**Execution order:** verify environment and source rights; implement the two-family development pipeline; validate grouping and paired metrics; expand the benchmark; freeze; produce the base report; apply the training decision rule.
