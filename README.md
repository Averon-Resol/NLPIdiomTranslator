# Idiom-Aware Cross-Lingual Translation

An NLP pipeline for detecting idiomatic expressions and translating them into culturally meaningful equivalents across Indian languages. The project combines idiom detection, semantic retrieval, and fallback generation to avoid literal translations of figurative language.

## Contributors

| Person | Name | Responsibility |
| --- | --- | --- |
| 1 | Ihsal Riyas | Data processing and idiom detection |
| 2 | Ramanand Balaji | Semantic similarity and retrieval |
| 3 | Dhruv Nair | Generation, evaluation, and integration |

---

## Overview

Idioms often lose their meaning when translated word-for-word. This project addresses that problem through a three-stage pipeline:

1. **Detection**: Classify whether a sentence or phrase is idiomatic.
2. **Retrieval**: Search for a semantically similar target-language idiom.
3. **Generation**: Generate a fallback equivalent when retrieval is insufficient.

Supported target languages in the current pipeline:

- Hindi (`hi`)
- Malayalam (`ml`)
- Telugu (`te`)
- English (`en`) for explanation and reverse-direction training examples

---

## Architecture

```text
Input sentence
    |
    v
Idiom Detection
XLM-RoBERTa binary classifier
    |
    |-- Literal    -> Standard translation baseline
    |
    v
Semantic Retrieval
LaBSE embeddings + FAISS index
    |
    |-- High-confidence match -> Retrieved idiom equivalent
    |
    v
Fallback Generation
mT5-small sequence-to-sequence generator
    |
    v
Translated idiom or best available equivalent
```

---

## Objectives

- **Detect idiomatic usage** in English and multilingual inputs.
- **Retrieve culturally appropriate equivalents** instead of literal translations.
- **Support Indian languages** with curated Hindi, Malayalam, and Telugu idiom data.
- **Provide an end-to-end pipeline** that can be evaluated module-by-module or as a full system.

---

## Repository Structure

```text
NLPIdiomTranslator/
|-- data/
|   |-- raw/                     # Raw JSON/CSV idiom sources
|   `-- processed/               # Prepared train/test and retrieval datasets
|-- models/                      # Local model artifacts, ignored by Git
|-- notebooks/
|   `-- exploration.py           # Data sanity checks and exploratory analysis
|-- results/                     # Generated evaluation reports, ignored by Git
|-- src/
|   |-- data_pipeline.py         # Dataset loading and consolidation
|   |-- preprocess.py            # Cleaning and train/val/test split creation
|   |-- idiom_detector.py        # XLM-RoBERTa detector training and inference
|   |-- semantic_retriever.py    # LaBSE + FAISS retrieval module
|   |-- generator.py             # mT5 generator training and inference
|   `-- pipeline.py              # End-to-end translation pipeline
|-- requirements.txt
`-- README.md
```

---

## Data Files

Key processed files:

| File | Purpose |
| --- | --- |
| `data/processed/train.csv` | Detector training split |
| `data/processed/val.csv` | Detector validation split |
| `data/processed/test.csv` | Detector test split |
| `data/processed/cross_lingual.csv` | Cross-lingual idiom pairs for retrieval |
| `data/processed/generator_train.csv` | Generator training split |
| `data/processed/generator_val.csv` | Generator validation split |
| `data/processed/generator_test.csv` | Generator test split |

Detection split schema:

```text
text | label | idiom_string | source
```

Cross-lingual schema:

```text
source_idiom | source_lang | target_idiom | target_lang | label | idiom_string | split | source
```

Generator schema:

```text
input_text | target_text | source_lang | target_lang
```

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Build and preprocess data

```bash
python src/data_pipeline.py
python src/preprocess.py
```

Optional data inspection:

```bash
python notebooks/exploration.py
```

### 2. Train and evaluate the detector

```bash
python src/idiom_detector.py --mode train
python src/idiom_detector.py --mode eval
```

Single-sentence prediction:

```bash
python src/idiom_detector.py --mode predict --text "She spilled the beans about the surprise."
```

### 3. Build and query the semantic retriever

Build the FAISS index:

```bash
python src/semantic_retriever.py --mode build
```

Query target-language equivalents:

```bash
python src/semantic_retriever.py --mode query --text "spill the beans" --target_lang hi --top_k 5
```

Run detector plus retrieval:

```bash
python src/semantic_retriever.py --mode pipeline --text "He kicked the bucket last year." --target_lang hi --top_k 3
```

### 4. Train and evaluate the generator

```bash
python src/generator.py --mode train
python src/generator.py --mode eval
```

Generate a candidate idiom:

```bash
python src/generator.py --mode generate --text "spill the beans" --target_lang ml
```

### 5. Run the full pipeline

Translate a single input:

```bash
python src/pipeline.py --mode translate --text "She spilled the beans" --target_lang ml
```

Evaluate the end-to-end system:

```bash
python src/pipeline.py --mode eval --target_lang hi --n_samples 50
```

Check module availability:

```bash
python src/pipeline.py --mode status
```

---

## Model Components

| Component | Model/Method | Output |
| --- | --- | --- |
| Idiom detector | `xlm-roberta-base` | Literal vs idiomatic classification |
| Semantic retriever | LaBSE + FAISS | Ranked target-language idiom matches |
| Generator | `google/mt5-small` | Generated idiom equivalent or explanation |
| Baseline translator | Helsinki-NLP MarianMT | Standard non-idiom translation fallback |

Model artifacts are written to:

```text
models/idiom_detector/
models/semantic_retriever/
models/generator/
```

These directories are intentionally ignored by Git because model weights and checkpoints are large.

---

## Evaluation

The detector reports:

- Accuracy
- Macro F1
- Precision and recall per class
- Confusion matrix

The generator reports:

- BLEU
- BERTScore F1
- Exact match
- Sample generated outputs

Evaluation reports are written under `results/`. These are generated artifacts and are ignored by Git.

---

## Current Notes

- Retrieval is the strongest component for idiom translation because idioms often have fixed cultural equivalents.
- Generation is best treated as a fallback when retrieval confidence is low.
- Exact match is strict for idiom generation, so manual inspection of sample outputs is recommended.
- GPU is recommended for training detector and generator models. CPU can run inference and small checks, but training will be slow.

---

## Git Hygiene

The repository ignores local artifacts such as:

- `models/`
- `results/`
- `wandb/` and `src/wandb/`
- Python cache directories
- one-off append scripts and generated backup/checkpoint CSVs

Before pushing, verify the staged files:

```bash
git status
```

Only source code, final data files, documentation, and reproducible configuration should be committed.
