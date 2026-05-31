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
2026-05-31

Added

Modern Web Interface (`app.py` & `templates/index.html`):
Built a responsive, single-page web UI using Tailwind CSS and Lucide Icons. The frontend now features interactive language selection, loading states, and clipboard copying, served directly via Flask.

Environment Management: 
Integrated `python-dotenv` utilizing `find_dotenv()` for robust, location-agnostic loading of the `GROQ_API_KEY` to secure the cloud integration.

Changed

Hybrid RAG Architecture (`generator.py`):
Shifted from purely local generation to a hybrid architecture. Replaced the local Gemma-2-2B model with the Groq API, utilizing Meta's `llama-3.3-70b-versatile` model. This bypasses all local VRAM constraints and provides near-instantaneous, high-quality translations using LPUs.

Retriever Language Filtering (`pipeline.py`):
Fixed a bug where semantic retrieval context was bleeding across languages. Added a language map to correctly convert full language names (e.g., "telugu") to the 2-letter codes (e.g., "te") expected by the FAISS index.

Removed

Heavy Local Dependencies:
Stripped `accelerate`, `peft`, `trl`, and `bitsandbytes` from `requirements.txt`. The web app is now incredibly lightweight, relying only on `flask`, `groq`, and the core detector/retriever libraries.

Local Generator Weights:
Safely deleted the `models/generator/` directory, freeing up gigabytes of local storage since generation is now handled entirely in the cloud.

2026-05-29 (Update 2)

Changed

Model Architecture Upgrade (mT5 to Gemma-2):

Abandoned google/mt5-small full fine-tuning due to poor cross-lingual semantic reasoning and hallucination outputs.

Pivoted to parameter-efficient fine-tuning (QLoRA) using google/gemma-2-2b-it (2 Billion parameters) in 4-bit precision.

Note: Attempted to load google/gemma-2-9b-it but hit a hard 8GB VRAM limit during the 32-bit vocabulary projection phase.

Training Optimizations (generator.py):

Upgraded trl library initialization from TrainingArguments to the modern SFTConfig.

Fixed SFTTrainer initialization to use processing_class instead of the deprecated tokenizer argument.

Replaced standard attention with PyTorch's native sdpa (Scaled Dot Product Attention) to bypass flash-attn C++ compilation errors while retaining near-identical speed boosts.

Rebalanced memory constraints: Increased per_device_train_batch_size to 4, decreased gradient_accumulation_steps to 2, and added dataloader_num_workers=4 to accelerate data preprocessing and fully utilize the RTX 3070 Ti.

2026-05-29

Added

Bidirectional Training Data: Updated data_pipeline.py to generate Native $\rightarrow$ English and English $\rightarrow$ Native pairs for all Kaggle JSON sources (Malayalam, Telugu, Hindi). This was critical to fix target language bleed in the mT5 generator.

RAG Preparation (Retriever Index): Rebuilt the FAISS index (python src/semantic_retriever.py --mode build) to include the newly added idioms, allowing them to be used as context for the fallback generator.

Changed

mT5 Training Stability (generator.py):

Replaced standard FP16 mixed-precision (fp16=torch.cuda.is_available()) with BFloat16 (bf16=True) to prevent instant model collapse and nan losses caused by mT5's precision instability.

Lowered the learning rate from 5e-4 to 3e-4 to further stabilize mT5-small's fine-tuning.

Kaggle JSON Loading: Ensured data_pipeline.py correctly reads from the newly updated/combined .json files in data/raw/ instead of hardcoded limited sets. Total unified idiom count increased from 53,996 to 54,567 rows.

Fixed

CUDA Out of Memory (OOM) during Evaluation: Added torch.cuda.empty_cache() immediately prior to the BERTScore calculation in generator.py to clear the VRAM occupied by the training/generation phases, preventing roberta-large from crashing the 8GB RTX 3070 Ti.


## Metric Reference

All evals are run with:
```bash
python src/generator.py --mode eval
python src/pipeline.py --mode eval --target_lang hi --n_samples 50
python src/pipeline.py --mode eval --target_lang ml --n_samples 50
python src/pipeline.py --mode eval --target_lang te --n_samples 50
```

Report files land in `results/`.
