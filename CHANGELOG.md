# NLPIdiomTranslator — Change Log

Tracks all improvements made after the initial baseline results.
Each entry records what changed, why, and what effect it had.

---

## Baseline (Pre-improvement)

**Detector results** (XLM-RoBERTa, `models/idiom_detector/`)

| Metric | Score |
|---|---|
| Accuracy | 93.50% |
| Macro F1 | 0.9104 |
| Literal F1 | 0.8635 |
| Idiomatic F1 | 0.9574 |

Confusion matrix: 177 literal→idiomatic false positives, 252 idiomatic→literal false negatives.

**Generator results** (mT5-small, `models/generator/`)

| Metric | Score |
|---|---|
| Test pairs | 269 |
| BLEU | 0.17 |
| BERTScore F1 | 0.8461 |
| Exact match | 0.0% (0/269) |

Known issues at baseline:
- Model hallucinating plausible-but-wrong idioms
- Mixed-script outputs (Malayalam script appearing in Telugu generations)
- Prompt has no grounding context — model must recall mappings from weights alone
- mT5-small capacity spread too thin across 3 languages with sparse data

---

## Changes

<!-- New entries go at the top of this section, most recent first -->

---

*No changes recorded yet. First entry will appear here.*

---

## Change Entry Template

Copy this block for each new change:

```
### [YYYY-MM-DD] — <short title>

**What changed:** <file(s) edited and what was modified>

**Why:** <the problem this solves or the hypothesis being tested>

**Expected effect:** <what we expect to improve and why>

**Result:** <fill in after re-running eval — metrics before/after>

---
```

---

## Metric Reference

All evals are run with:
```bash
python src/generator.py --mode eval
python src/pipeline.py --mode eval --target_lang hi --n_samples 50
python src/pipeline.py --mode eval --target_lang ml --n_samples 50
python src/pipeline.py --mode eval --target_lang te --n_samples 50
```

Report files land in `results/`.
