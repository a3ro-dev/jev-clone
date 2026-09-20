# Plan (ultra, 2026-09-20)

Question milestone 1 answers: does `knowledgator/gliclass-instruct-base-v1.0` already do the job? Everything below exists only to answer that.

## Facts

| | |
|---|---|
| Base | `knowledgator/gliclass-instruct-base-v1.0`, 0.2B, Apache-2.0 |
| Its training data | commonsense_qa, gliclass-v3-logic-dataset (FineWeb, CommonsenseQA, **MultiNLI**), formal-logic-2k. Parent: MoritzLaurer synthetic. |
| Input string | `<<LABEL>>l1<<LABEL>>l2<<SEP>>{prompt}{text}`; single-label = softmax |
| Train JSON | `[{"text","all_labels","true_labels"}]`, soft `{label: p}` allowed |
| Local GPU | RTX 4050 6 GB. Milestone 1 costs $0. Fine-tuning 0.2B also fits. Runpod only for parallel seeds. |

## Files

```
eval.py        # ingest + split + run base + metrics. One file.
items.jsonl    # frozen eval, git-tagged eval-v0.1
pairs.csv      # hand-written counterfactuals: family,state_a,gold_a,state_b,gold_b
README.md      # decision rule + results table
```

## Families (4, not 10)

| family | type | source | why this one |
|---|---|---|---|
| boolq | boolean | `boolq` | passage-dependent yes/no |
| vitaminc | choice (3) | `tals/vitaminc` | verification; ships its own contrastive pairs, so counterfactuals are free |
| clinc | choice (5) | `clinc_oos` | intent; has domains for domain hold-out |
| civil | boolean | `civil_comments` (>=0.5 yes, <=0.1 no) | policy; CC0 |

Dropped: banking77, sms_spam, tweet_eval (same shapes as the above), paws, SummEval (availability unknown), constructed urgency (biggest labor item, zero evidence yet it's needed). Add a family when a result says the four don't cover a claim.

## Item

`{"family","template","state","question","options":["yes","no"],"gold":"yes","domain":null,"src":"boolq:train:12"}`. Options are description strings; that's what GLiClass consumes.

## Templates

Two question phrasings per family: `t1` for dev, `t2` for test. No shared 4-gram. Paraphrase hold-out is that.

## Split

`bucket = int(sha256(src).hexdigest(), 16) % 100`; `<70 train, 70..84 dev, else test`. Balance per `(family, template)` to the smallest gold class. 300 dev, 300 test per family. clinc: one domain appears in test only.

## Pairs

vitaminc: its own revision pairs, 50, checked by eye. boolq, clinc, civil: 20 hand-written each, in `pairs.csv`. No generator, no verification queue; writing 60 pairs is faster than building the pipeline to generate and verify them. Scored: pair correct iff both halves correct.

## Ablation

One: `state = ""`. Compare accuracy to full. Others when a result demands them.

## Metrics

accuracy, log loss, 10-bin ECE, pair accuracy, state-ablated accuracy. 1000-resample bootstrap 95% CI per family. Per-family rows only; no pooled number.

## Freeze

`git tag eval-v0.1` on the commit that adds `items.jsonl` and `pairs.csv`. `sha256sum` in README. Test rows are read only by `eval.py --test`. Change a test row -> `eval-v0.2`.

## Decision rule (in README before tagging)

Train iff base accuracy on dev is under 0.85 on at least 2 families, or pair accuracy under 0.70 on at least 2. Base at or above 0.95 everywhere: stop, ship a wrapper.

Milestone 2 gets written when this gate passes. Not before.

## Order

1. `eval.py` ingest for boolq + civil, split, base run. Half a day.
2. vitaminc + clinc, templates, pairs.csv. Half a day.
3. Run, fill README table, tag. Done.

## Verify before use

civil_comments and clinc_oos licenses at ingest; vitaminc license (redistribute rows only if permitted, else ship the ingest).
