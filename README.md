# NLP Project — Idiom-Aware Cross-Lingual Translation
### Group Members
| Person | Name | Role |
|--------|------|------|
| 1 | Ihsal Riyas | Data & Detection |
| 2 | Ramanand Balaji | Semantic Similarity & Retrieval |
| 3 | Dhruv Nair | Generation, Evaluation & Integration |

---

## Project Overview
An idiom-aware cross-lingual translation system that detects idiomatic expressions in a source sentence and maps them to semantically equivalent phrases in the target language — instead of producing awkward literal translations.

### Pipeline
```
Input Sentence
     │
     ▼
[Stage 1] Idiom Detection       ← Person 1 (this repo)
     │  XLM-RoBERTa classifier
     │  Output: idiomatic / literal flag
     │
     ▼
[Stage 2] Semantic Similarity   ← Person 2
     │  LaBSE embeddings + FAISS retrieval
     │  Output: closest target-language equivalent
     │
     ▼
[Stage 3] Generation / Output   ← Person 3
          mT5 fallback generator + full pipeline
          Output: translated sentence with idiomatic equivalent
```

---

## Person 1 — Data & Detection (Ihsal Riyas)

### Datasets Used
| Dataset | Source | Purpose |
|---------|--------|---------|
| MAGPIE  | HuggingFace `hsseinmz/magpie` | Idiom detection (binary labels) |
| PIE-English | HuggingFace `hsseinmz/pie` | Idiom detection (token-level → sentence-level) |
| LIDIOMS | Zenodo | Cross-lingual idiom pairs (for Person 2 & 3) |

### Shared Data Format
All processed files use this schema:

```
source_idiom | source_lang | target_idiom | target_lang | label | idiom_string | split | source
```

- `label`: `1` = Idiomatic, `0` = Literal
- Files shared with the team: `train.csv`, `val.csv`, `test.csv`

---

## Project Structure
```
NLP-Project/
├── data/
│   ├── raw/              ← downloaded raw files
│   └── processed/        ← cleaned & split CSVs
│       ├── unified_idioms.csv
│       ├── detection_only.csv
│       ├── cross_lingual.csv   ← for Person 2 & 3
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   └── idiom_detector/   ← saved XLM-RoBERTa model
├── notebooks/
│   └── exploration.py    ← EDA & sanity checks
├── results/
│   └── detection_results.txt
├── src/
│   ├── data_pipeline.py  ← Step 1: download & merge datasets
│   ├── preprocess.py     ← Step 2: clean & split data
│   └── idiom_detector.py ← Step 3: train / eval / predict
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment
> ```bash
> python -m venv venv
> venv\Scripts\activate        # Windows
> source venv/bin/activate     # Mac/Linux
> ```

### 2. Run the full pipeline (in order)

**Step 1 — Download & consolidate datasets**
```bash
python src/data_pipeline.py
```

**Step 2 — Clean & create train/val/test splits**
```bash
python src/preprocess.py
```

**Step 3 — Sanity-check the data**
```bash
python notebooks/exploration.py
```

**Step 4 — Train the idiom detector**
```bash
python src/idiom_detector.py --mode train
```

**Step 5 — Evaluate on test set**
```bash
python src/idiom_detector.py --mode eval
```

**Step 6 — Predict on a single sentence**
```bash
python src/idiom_detector.py --mode predict --text "It's raining cats and dogs"
```

---

## Model Details

| Setting | Value |
|---------|-------|
| Base model | `xlm-roberta-base` |
| Task | Binary sequence classification |
| Labels | `0 = Literal`, `1 = Idiomatic` |
| Max token length | 128 |
| Learning rate | 2e-5 |
| Batch size | 16 (reduce to 8 if GPU runs out of memory) |
| Epochs | 5 (with early stopping, patience=2) |
| Mixed precision | Auto (fp16 on GPU) |

---

## Using the Detector in Another Module (Person 2 & 3)

```python
from src.idiom_detector import IdiomDetector

detector = IdiomDetector()   # loads saved model from models/idiom_detector/

result = detector.predict("She spilled the beans about the surprise party.")
# → {"label": "Idiomatic", "confidence": 0.96, "is_idiomatic": True}

# Batch prediction
results = detector.predict_batch([
    "The cat sat on the mat.",
    "He kicked the bucket last year.",
])
```

---

## Person 2 — Semantic Similarity & Retrieval (LaBSE + FAISS)

### Build retrieval index
```bash
python src/semantic_retriever.py --mode build
```

By default this consumes:
- `data/processed/cross_lingual.csv`
- idiom files in `data/raw/` (`hindi.json`, `malayalam.json`, `telugu.json`, `*_idioms.csv`)

Note: generic parallel corpora are excluded by default for retrieval quality.

Saved artifacts:
- `models/semantic_retriever/idiom_index.faiss`
- `models/semantic_retriever/metadata.csv`

### Query nearest idiom equivalents
```bash
python src/semantic_retriever.py --mode query \
  --text "He kicked the bucket" \
  --target_lang hi \
  --top_k 5
```

### Run detector + retrieval pipeline
```bash
python src/semantic_retriever.py --mode pipeline \
  --text "He kicked the bucket last year." \
  --target_lang hi \
  --top_k 3
```

If detector predicts `Literal`, retrieval is skipped. If detector predicts `Idiomatic`,
the module returns top semantic matches from the FAISS index.

---

## Compute Notes
- **Google Colab (T4)** — Recommended for training. Upload repo, install requirements, run scripts.
- **Kaggle Notebooks** — 30hrs/week free GPU, good for longer runs.
- **CPU** — Works but training will be slow (~1hr per epoch). Use for testing only.

### Colab Quick-Start
```python
# In a Colab cell:
!git clone https://github.com/YOUR_ORG/NLP-Project.git
%cd NLP-Project
!pip install -r requirements.txt
!python src/data_pipeline.py
!python src/preprocess.py
!python src/idiom_detector.py --mode train
```

---

## Evaluation Metrics
The detector reports:
- **Accuracy**
- **Macro F1** ← primary metric
- **Precision / Recall** per class
- **Confusion Matrix**

Results are saved to `results/detection_results.txt` after each eval run.

---

## GitHub Workflow
```
main          ← stable, working code only
├── person1   ← Ihsal (this branch)
├── person2   ← Ramanand
└── person3   ← Dhruv
```

- Push to your own branch, open a PR into `main` when your module is ready
- The shared data files (`train.csv`, `val.csv`, `test.csv`) live on `main` once Person 1 is done
