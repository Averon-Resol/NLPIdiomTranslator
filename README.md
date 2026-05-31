# NLP Idiom Translator

A hybrid NLP pipeline for detecting and translating idiomatic expressions. The system combines a binary idiom classifier with a Retrieval-Augmented Generation (RAG) approach powered by Llama-3.3-70B to produce culturally accurate, figurative translations.

---

## Overview

Idiomatic expressions are notoriously difficult for machine translation systems, which tend to produce literal (and often meaningless) renderings. This project tackles the problem with a two-stage pipeline:

1. **Idiom Detector** — a binary classifier that separates literal text from idiomatic text, acting as an efficient pre-filter before the expensive LLM call.
2. **Idiom Generator** — a RAG-augmented LLM (Llama-3.3-70B) that translates detected idioms into their culturally equivalent target-language expressions.

---

## Pipeline Architecture

```
Input Text
    │
    ▼
┌─────────────────────┐
│   Idiom Detector    │  ← Binary classifier (Literal / Idiomatic)
└────────┬────────────┘
         │ Idiomatic
         ▼
┌─────────────────────┐
│  Semantic Retriever │  ← Fetches relevant idiom pairs from knowledge base
└────────┬────────────┘
         │ Retrieved context
         ▼
┌─────────────────────┐
│  Llama-3.3-70B      │  ← Generates culturally accurate translation
└─────────────────────┘
         │
         ▼
    Translated Output
```

Literal sentences bypass the LLM entirely, keeping inference costs low.

---

## Results

### Idiom Detector

Evaluated on **6,605 samples** with binary classification (Literal vs. Idiomatic).

**Overall Accuracy: 93.50%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Literal | 0.8434 | 0.8846 | 0.8635 | 1,534 |
| Idiomatic | 0.9646 | 0.9503 | 0.9574 | 5,071 |
| *Macro Avg* | *0.9040* | *0.9175* | *0.9104* | *6,605* |
| *Weighted Avg* | *0.9364* | *0.9350* | *0.9356* | *6,605* |

**Confusion Matrix**

| Actual \ Predicted | Predicted Literal | Predicted Idiomatic |
|---|---|---|
| **Actual Literal** | 1,357 (TN) | 177 (FP) |
| **Actual Idiomatic** | 252 (FN) | 4,819 (TP) |

Key takeaways:
- **High idiomatic precision (96.46%):** When the model flags a sentence as an idiom, it is almost always correct — very few literal sentences are wastefully passed to the LLM.
- **Balanced errors:** False negatives (252) and false positives (177) are relatively symmetric, indicating no strong class bias in the classifier.

---

### Generator (Llama-3.3-70B via API)

Evaluated on **42 idiom pairs**.

| Metric | Score | Notes |
|---|---|---|
| **BERTScore F1** | **0.7599** | Strong semantic alignment even when exact wording differs. |
| **Exact Match** | **40.5%** (17/42) | Over 40% of outputs matched the target idiom string exactly. |
| **BLEU** | **13.87** | Expected for figurative translation; BLEU penalises creative-but-valid phrasing. |

The high BERTScore alongside a strong exact match rate confirms the RAG context is being effectively utilized — the model produces culturally accurate outputs rather than literal translations.

---

## Repository Structure

```
NLPIdiomTranslator/
├── data/               # Datasets and idiom knowledge base
├── detector/           # Idiom classifier (training, evaluation)
├── retriever/          # Semantic retrieval module (RAG)
├── generator/          # LLM inference pipeline
├── evaluation/         # Metrics scripts (BERTScore, BLEU, Exact Match)
├── notebooks/          # Exploratory analysis and experiments
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- Access to Llama-3.3-70B via API (e.g., Together AI, Groq, or a self-hosted endpoint)

### Installation

```bash
git clone https://github.com/Averon-Resol/NLPIdiomTranslator.git
cd NLPIdiomTranslator
pip install -r requirements.txt
```

### Configuration

Set your LLM API key as an environment variable:

```bash
export LLM_API_KEY="your_api_key_here"
```

---

## Usage

### Run the full pipeline

```bash
python main.py --input "It's raining cats and dogs."
```

### Evaluate the detector

```bash
python evaluation/evaluate_detector.py --data data/test.csv
```

### Evaluate the generator

```bash
python evaluation/evaluate_generator.py --pairs data/idiom_pairs_test.json
```

---

## Evaluation Metrics

- **BERTScore F1** — measures semantic similarity between generated and reference translations using contextual embeddings.
- **Exact Match** — percentage of outputs that exactly match the ground-truth target idiom string.
- **BLEU** — n-gram overlap score; included for completeness but expected to be low for figurative translation tasks.

---

## Acknowledgements

- [Llama 3.3 70B](https://ai.meta.com/blog/meta-llama-3/) by Meta AI
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
