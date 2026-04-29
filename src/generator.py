"""
generator.py
============
Person 3 — Idiom Equivalent Generator (mT5-small Fine-tuning)
Author   : Dhruv Nair

Fine-tunes mT5-small as a seq2seq model to generate culturally equivalent
idiom phrases in a target language when FAISS retrieval finds no match.

The model learns to map:
  Input  → "translate idiom to <lang>: <source idiom or figurative meaning>"
  Output → "<target language idiomatic equivalent>"

Training data is built from the raw Indian-language idiom JSONs
(Malayalam, Telugu, Hindi), which contain figurative_meaning (English)
paired with the native-script idiom — giving us real cross-lingual pairs.

Inputs : data/raw/malayalam.json, telugu.json, hindi.json
         data/processed/cross_lingual.csv
Outputs: models/generator/          (saved mT5 model + tokenizer)
         results/generation_results.txt

Usage:
  # Prepare training data, fine-tune, then run a quick sanity check
  python src/generator.py --mode train

  # Evaluate the saved model on held-out pairs (BLEU + BERTScore)
  python src/generator.py --mode eval

  # Generate an equivalent for a single idiom
  python src/generator.py --mode generate \\
      --text "spill the beans" \\
      --target_lang ml

  # List supported target languages
  python src/generator.py --mode langs
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT / "data" / "raw"
DATA_PROC     = ROOT / "data" / "processed"
MODEL_DIR     = ROOT / "models" / "generator"
RESULTS       = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ─── Language map ─────────────────────────────────────────────────────────────
LANG_NAMES = {
    "ml": "malayalam",
    "te": "telugu",
    "hi": "hindi",
    "en": "english",
}

# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    """Hyperparameters for mT5-small fine-tuning."""
    model_name        : str   = "google/mt5-small"
    max_input_length  : int   = 128
    max_target_length : int   = 64
    train_batch_size  : int   = 8       # reduce to 4 on low VRAM
    eval_batch_size   : int   = 16
    learning_rate     : float = 5e-4
    num_epochs        : int   = 10
    warmup_steps      : int   = 100
    weight_decay      : float = 0.01
    seed              : int   = 42
    val_split         : float = 0.15    # fraction of data used for validation
    test_split        : float = 0.15    # fraction used for final eval
    score_threshold   : float = 0.0     # min FAISS score to trust retrieval
    beam_size         : int   = 4       # beams for generation
    early_stopping    : int   = 3       # patience epochs


CFG = GeneratorConfig()


# ─── Data Preparation ─────────────────────────────────────────────────────────

def _load_raw_pairs() -> pd.DataFrame:
    """
    Build training pairs from the raw JSON files.

    Each JSON entry looks like:
      {
        "idiom": "<native script phrase>",
        "figurative_meaning": "<English explanation>",
        ...
      }

    We create two complementary training examples per entry:
      1. English figurative meaning  →  native script idiom
         (teaches: given an English idiom concept, produce the native equivalent)
      2. Native script idiom  →  English figurative meaning
         (teaches: given a native idiom, explain it in English)

    This bidirectional approach doubles our training data and makes the
    model more robust when the source phrase is in either language.
    """
    rows = []

    lang_files = {
        "ml": DATA_RAW / "malayalam.json",
        "te": DATA_RAW / "telugu.json",
        "hi": DATA_RAW / "hindi.json",
    }

    for lang_code, json_path in lang_files.items():
        if not json_path.exists():
            print(f"  [WARN] {json_path.name} not found — skipping.")
            continue

        with open(json_path, encoding="utf-8") as f:
            entries = json.load(f)

        lang_name = LANG_NAMES[lang_code]
        added = 0

        for entry in entries:
            native_idiom   = str(entry.get("idiom", "")).strip()
            figurative_en  = str(entry.get("figurative_meaning", "")).strip()
            literal_en     = str(entry.get("literal_meaning", "")).strip()

            if not native_idiom or not figurative_en:
                continue

            # Pair 1: English meaning → native idiom
            rows.append({
                "input_text"  : f"translate idiom to {lang_name}: {figurative_en}",
                "target_text" : native_idiom,
                "source_lang" : "en",
                "target_lang" : lang_code,
            })

            # Pair 2: native idiom → English figurative meaning
            rows.append({
                "input_text"  : f"translate idiom to english: {native_idiom}",
                "target_text" : figurative_en,
                "source_lang" : lang_code,
                "target_lang" : "en",
            })

            # Pair 3 (bonus): literal → figurative (same language paraphrase)
            if literal_en and literal_en != figurative_en:
                rows.append({
                    "input_text"  : f"translate idiom to english: {literal_en}",
                    "target_text" : figurative_en,
                    "source_lang" : "en",
                    "target_lang" : "en",
                })

            added += 1

        print(f"  [{lang_code}] {added} raw idioms → {added * 2}+ training pairs")

    return pd.DataFrame(rows)


def _load_cross_lingual_pairs() -> pd.DataFrame:
    """
    Pull additional pairs from cross_lingual.csv.
    These are already in source_idiom / target_idiom format.
    """
    path = DATA_PROC / "cross_lingual.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[
        df["source_idiom"].notna() & df["target_idiom"].notna() &
        (df["source_idiom"].str.strip() != "") &
        (df["target_idiom"].str.strip() != "")
    ].copy()

    rows = []
    for _, row in df.iterrows():
        src_lang  = str(row["source_lang"]).strip()
        tgt_lang  = str(row["target_lang"]).strip()
        src_name  = LANG_NAMES.get(src_lang, src_lang)
        tgt_name  = LANG_NAMES.get(tgt_lang, tgt_lang)
        rows.append({
            "input_text"  : f"translate idiom to {tgt_name}: {row['source_idiom']}",
            "target_text" : str(row["target_idiom"]),
            "source_lang" : src_lang,
            "target_lang" : tgt_lang,
        })

    return pd.DataFrame(rows)


def prepare_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combines raw + cross-lingual pairs, deduplicates, and splits into
    train / val / test DataFrames.

    Returns:
        (train_df, val_df, test_df)
    """
    print("\n[Generator] Preparing training data...")

    raw_df   = _load_raw_pairs()
    cross_df = _load_cross_lingual_pairs()

    all_pairs = pd.concat([raw_df, cross_df], ignore_index=True)
    all_pairs.drop_duplicates(subset=["input_text", "target_text"], inplace=True)
    all_pairs.reset_index(drop=True, inplace=True)

    print(f"  Total pairs after dedup: {len(all_pairs)}")
    print(f"  Target lang distribution:\n{all_pairs['target_lang'].value_counts().to_string()}")

    # Shuffle and split
    all_pairs = all_pairs.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    n         = len(all_pairs)
    n_test    = max(1, int(n * CFG.test_split))
    n_val     = max(1, int(n * CFG.val_split))
    n_train   = n - n_val - n_test

    train_df  = all_pairs.iloc[:n_train].reset_index(drop=True)
    val_df    = all_pairs.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df   = all_pairs.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"  Split → train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Persist for later eval runs
    train_df.to_csv(DATA_PROC / "generator_train.csv", index=False, encoding="utf-8")
    val_df.to_csv(DATA_PROC / "generator_val.csv",   index=False, encoding="utf-8")
    test_df.to_csv(DATA_PROC / "generator_test.csv", index=False, encoding="utf-8")
    print("  Saved generator_train/val/test.csv to data/processed/")

    return train_df, val_df, test_df


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

def _make_dataset(df: pd.DataFrame, tokenizer):
    """
    Tokenises a DataFrame of (input_text, target_text) pairs for mT5.
    Returns a HuggingFace Dataset object.
    """
    from datasets import Dataset

    hf_ds = Dataset.from_pandas(df[["input_text", "target_text"]])

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length  = CFG.max_input_length,
            truncation  = True,
            padding     = "max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length  = CFG.max_target_length,
                truncation  = True,
                padding     = "max_length",
            )
        # Replace padding token id with -100 so loss ignores padding
        label_ids = [
            [(l if l != tokenizer.pad_token_id else -100) for l in lbl]
            for lbl in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return hf_ds.map(tokenize, batched=True, remove_columns=hf_ds.column_names)


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    """
    Fine-tunes mT5-small on idiom translation pairs.

    Steps:
      1. Prepare data (raw JSON + cross_lingual.csv)
      2. Tokenise with mT5 tokenizer
      3. Fine-tune with HuggingFace Seq2SeqTrainer
      4. Save model + tokenizer to models/generator/
    """
    import torch
    from transformers import (
        MT5ForConditionalGeneration,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        set_seed,
    )

    set_seed(CFG.seed)

    torch_version = torch.__version__.split("+", 1)[0]
    torch_parts = tuple(int(part) for part in torch_version.split(".")[:2])
    if torch_parts < (2, 6):
        raise RuntimeError(
            "torch>=2.6.0 is required before loading mT5 checkpoints with the "
            "current Transformers safety checks. Upgrade the environment with: "
            "python -m pip install --upgrade torch "
            "--index-url https://download.pytorch.org/whl/cu121"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  Idiom Generator — Training (mT5-small)")
    print(f"  Device : {device.upper()}")
    print(f"  Model  : {CFG.model_name}")
    print("=" * 60)

    train_df, val_df, _ = prepare_training_data()

    print(f"\nLoading tokenizer: {CFG.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, fix_mistral_regex=False)

    print("Tokenising datasets ...")
    train_ds = _make_dataset(train_df, tokenizer)
    val_ds   = _make_dataset(val_df,   tokenizer)

    print(f"Loading model: {CFG.model_name} ...")
    model = MT5ForConditionalGeneration.from_pretrained(CFG.model_name)

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model   = model,
        padding = True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(MODEL_DIR / "checkpoints"),
        num_train_epochs            = CFG.num_epochs,
        per_device_train_batch_size = CFG.train_batch_size,
        per_device_eval_batch_size  = CFG.eval_batch_size,
        learning_rate               = CFG.learning_rate,
        warmup_steps                = CFG.warmup_steps,
        weight_decay                = CFG.weight_decay,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        predict_with_generate       = True,
        generation_max_length       = CFG.max_target_length,
        logging_dir                 = str(RESULTS / "generator_logs"),
        logging_steps               = 20,
        seed                        = CFG.seed,
        report_to                   = "none",
        fp16                        = torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        data_collator   = collator,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping)],
    )

    print("\nStarting fine-tuning...\n")
    trainer.train()

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Save config for reference
    config_out = MODEL_DIR / "generator_config.json"
    config_out.write_text(
        json.dumps(CFG.__dict__, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n✅ Model saved to: {MODEL_DIR}")
    print("\nRunning evaluation on test set...")
    evaluate()


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model_path: Optional[str] = None):
    """
    Loads the saved generator and evaluates on the held-out test set.
    Reports BLEU and BERTScore, and writes results to
    results/generation_results.txt.
    """
    import torch
    from transformers import MT5ForConditionalGeneration, AutoTokenizer

    mp = model_path or str(MODEL_DIR)

    if not (Path(mp) / "config.json").exists():
        print(f"[ERROR] No saved model found at {mp}. Run --mode train first.")
        return

    print("\n[Generator] Loading saved model for evaluation ...")
    tokenizer = AutoTokenizer.from_pretrained(mp, fix_mistral_regex=False)
    model     = MT5ForConditionalGeneration.from_pretrained(mp)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load test set
    test_path = DATA_PROC / "generator_test.csv"
    if not test_path.exists():
        print("[INFO] generator_test.csv not found — re-preparing data...")
        _, _, test_df = prepare_training_data()
    else:
        test_df = pd.read_csv(test_path)

    print(f"  Test set: {len(test_df)} pairs")

    # Generate predictions
    hypotheses = []
    references = []

    for _, row in test_df.iterrows():
        input_text = str(row["input_text"])
        ref        = str(row["target_text"])

        enc = tokenizer(
            input_text,
            return_tensors = "pt",
            max_length     = CFG.max_input_length,
            truncation     = True,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens = CFG.max_target_length,
                num_beams      = CFG.beam_size,
                early_stopping = True,
            )

        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        hypotheses.append(pred)
        references.append(ref)

    # ── BLEU ──
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        bleu_score = bleu.score
    except Exception as e:
        bleu_score = -1.0
        print(f"  [WARN] BLEU failed: {e}")

    # ── BERTScore ──
    try:
        from bert_score import score as bscore
        _, _, bert_f1 = bscore(
            hypotheses, references,
            lang          = "en",      # BERTScore uses en model for multilingual scoring
            rescale_with_baseline = False,
            verbose       = False,
        )
        bert_mean = float(bert_f1.mean())
    except Exception as e:
        bert_mean = -1.0
        print(f"  [WARN] BERTScore failed: {e}")

    # ── Exact match (phrase-level accuracy) ──
    exact = sum(h.strip().lower() == r.strip().lower() for h, r in zip(hypotheses, references))
    exact_acc = exact / len(hypotheses) if hypotheses else 0.0

    # ── Sample outputs ──
    sample_indices = random.sample(range(len(test_df)), min(5, len(test_df)))
    samples = []
    for i in sample_indices:
        samples.append({
            "input"    : test_df.iloc[i]["input_text"],
            "reference": references[i],
            "generated": hypotheses[i],
        })

    # ── Write results ──
    report_lines = [
        "=" * 60,
        "  Idiom Generator — Test Set Results",
        "=" * 60,
        "",
        f"  Test pairs      : {len(test_df)}",
        f"  BLEU score      : {bleu_score:.2f}",
        f"  BERTScore F1    : {bert_mean:.4f}",
        f"  Exact match     : {exact_acc*100:.1f}%  ({exact}/{len(hypotheses)})",
        "",
        "-" * 60,
        "  Sample Outputs (5 random examples)",
        "-" * 60,
    ]
    for i, s in enumerate(samples, 1):
        report_lines += [
            f"\n  [{i}] Input     : {s['input']}",
            f"      Reference : {s['reference']}",
            f"      Generated : {s['generated']}",
        ]

    report = "\n".join(report_lines)
    out_path = RESULTS / "generation_results.txt"
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nResults saved to {out_path}")

    return {
        "bleu"       : bleu_score,
        "bert_score" : bert_mean,
        "exact_match": exact_acc,
    }


# ─── Inference Wrapper ────────────────────────────────────────────────────────

class IdiomGenerator:
    """
    Lightweight inference wrapper for use in pipeline.py (Person 3)
    and by anyone who imports this module.

    Example:
        gen = IdiomGenerator()

        # Generate a Malayalam equivalent for an English idiom
        result = gen.generate("spill the beans", target_lang="ml")
        # → {
        #     "input"        : "spill the beans",
        #     "target_lang"  : "ml",
        #     "generated"    : "രഹസ്യം വെളിപ്പെടുത്തുക",
        #     "input_prompt" : "translate idiom to malayalam: spill the beans",
        #   }

        # Generate top-3 beam candidates
        results = gen.generate_top_k("kick the bucket", target_lang="hi", top_k=3)
    """

    def __init__(self, model_path: Optional[str] = None):
        import torch
        from transformers import MT5ForConditionalGeneration, AutoTokenizer

        mp = model_path or str(MODEL_DIR)

        if not (Path(mp) / "config.json").exists():
            raise FileNotFoundError(
                f"No saved generator model found at '{mp}'. "
                "Run: python src/generator.py --mode train"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(mp, fix_mistral_regex=False)
        self.model     = MT5ForConditionalGeneration.from_pretrained(mp)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"[IdiomGenerator] Loaded from '{mp}' | Device: {self.device}")

    def _build_prompt(self, text: str, target_lang: str) -> str:
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        return f"translate idiom to {lang_name}: {text.strip()}"

    def generate(
        self,
        text       : str,
        target_lang: str = "ml",
        max_length : Optional[int] = None,
        num_beams  : Optional[int] = None,
    ) -> dict:
        """
        Generate a single idiomatic equivalent.

        Args:
            text        : Source idiom or figurative description (English or native).
            target_lang : ISO code of the target language ('ml', 'te', 'hi', 'en').
            max_length  : Override default max_target_length.
            num_beams   : Override default beam size.

        Returns:
            dict with keys: input, target_lang, generated, input_prompt
        """
        import torch

        prompt = self._build_prompt(text, target_lang)

        enc = self.tokenizer(
            prompt,
            return_tensors = "pt",
            max_length     = CFG.max_input_length,
            truncation     = True,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens = max_length or CFG.max_target_length,
                num_beams      = num_beams  or CFG.beam_size,
                early_stopping = True,
            )

        generated = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        return {
            "input"       : text,
            "target_lang" : target_lang,
            "generated"   : generated,
            "input_prompt": prompt,
        }

    def generate_top_k(
        self,
        text       : str,
        target_lang: str = "ml",
        top_k      : int = 3,
    ) -> list[dict]:
        """
        Generate top-k beam candidates for the given input.

        Args:
            text        : Source idiom.
            target_lang : Target language ISO code.
            top_k       : Number of distinct candidates to return.

        Returns:
            List of dicts, each with keys: rank, generated, score (beam score)
        """
        import torch

        prompt = self._build_prompt(text, target_lang)

        enc = self.tokenizer(
            prompt,
            return_tensors = "pt",
            max_length     = CFG.max_input_length,
            truncation     = True,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens        = CFG.max_target_length,
                num_beams             = max(top_k, CFG.beam_size),
                num_return_sequences  = top_k,
                early_stopping        = True,
                output_scores         = True,
                return_dict_in_generate = True,
            )

        sequences = out.sequences
        scores    = out.sequences_scores.tolist() if hasattr(out, "sequences_scores") else [0.0] * top_k

        results = []
        for rank, (seq, score) in enumerate(zip(sequences, scores), 1):
            text_out = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
            results.append({
                "rank"     : rank,
                "generated": text_out,
                "score"    : round(float(score), 4),
            })

        return results

    @staticmethod
    def supported_languages() -> dict:
        """Returns the supported target languages."""
        return LANG_NAMES


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Idiom Generator — mT5-small (Person 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "generate", "langs"],
        default="train",
        help="train | eval | generate | langs",
    )
    parser.add_argument("--text",         type=str, default=None, help="Source idiom text")
    parser.add_argument("--target_lang",  type=str, default="ml", help="Target language code (ml/te/hi/en)")
    parser.add_argument("--top_k",        type=int, default=1,    help="Number of candidates (generate mode)")
    parser.add_argument("--model_path",   type=str, default=None, help="Override model directory")
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "eval":
        evaluate(model_path=args.model_path)

    elif args.mode == "generate":
        if not args.text:
            parser.error("--text is required with --mode generate")
        gen = IdiomGenerator(model_path=args.model_path)

        if args.top_k > 1:
            results = gen.generate_top_k(args.text, target_lang=args.target_lang, top_k=args.top_k)
            print(f"\nInput       : {args.text}")
            print(f"Target lang : {args.target_lang} ({LANG_NAMES.get(args.target_lang, '?')})")
            print(f"\nTop-{args.top_k} candidates:")
            for r in results:
                print(f"  [{r['rank']}] {r['generated']}  (score: {r['score']})")
        else:
            result = gen.generate(args.text, target_lang=args.target_lang)
            print(f"\nInput       : {result['input']}")
            print(f"Target lang : {result['target_lang']} ({LANG_NAMES.get(result['target_lang'], '?')})")
            print(f"Generated   : {result['generated']}")
            print(f"Prompt used : {result['input_prompt']}")

    elif args.mode == "langs":
        print("\nSupported target languages:")
        for code, name in LANG_NAMES.items():
            print(f"  {code}  →  {name}")


if __name__ == "__main__":
    main()
