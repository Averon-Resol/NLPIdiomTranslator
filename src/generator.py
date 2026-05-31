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
from dataclasses import dataclass
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
    learning_rate     : float = 3e-4
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
    
    # Clean up empty rows
    df = df[
        df["source_idiom"].notna() & df["target_idiom"].notna() &
        (df["source_idiom"].str.strip() != "") &
        (df["target_idiom"].str.strip() != "")
    ].copy()

    # ── CRITICAL DATA FIX ──────────────────────────────────────────────────
    # Exclude rows from the generic 'kunchukuttan' parallel corpus dataset
    if "source" in df.columns:
        df = df[df["source"].fillna("").astype(str).str.lower() != "kunchukuttan"].copy()
    # ───────────────────────────────────────────────────────────────────────

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
    if all_pairs.empty:
        raise ValueError(
            "No generator training pairs found. Add raw idiom JSON files under "
            "data/raw or create data/processed/cross_lingual.csv."
        )
    all_pairs.drop_duplicates(subset=["input_text", "target_text"], inplace=True)
    all_pairs.reset_index(drop=True, inplace=True)

    print(f"  Total pairs after dedup: {len(all_pairs)}")
    print(f"  Target lang distribution:\n{all_pairs['target_lang'].value_counts().to_string()}")

    # Shuffle and split
    all_pairs = all_pairs.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    n         = len(all_pairs)
    if n < 3:
        raise ValueError("Need at least 3 generator training pairs to create train/val/test splits.")
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
    Fine-tunes Gemma-2-9B using QLoRA for idiom translation.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    # 1. Update Config (You can change this at the top of your file too)
    model_id = "google/gemma-2-2b-it" 
    
    print("=" * 60)
    print(f"  Idiom Generator — LoRA Training ({model_id})")
    print("=" * 60)

    train_df, val_df, _ = prepare_training_data()

    # 2. Format Data for Causal LM (Prompt -> Output)
    def format_instruction(row):
        # Using Gemma's chat template format
        return f"<bos><start_of_turn>user\n{row['input_text']}<end_of_turn>\n<start_of_turn>model\n{row['target_text']}<end_of_turn>"
    
    train_df['text'] = train_df.apply(format_instruction, axis=1)
    val_df['text'] = val_df.apply(format_instruction, axis=1)
    
    from datasets import Dataset
    train_ds = Dataset.from_pandas(train_df[['text']])
    val_ds = Dataset.from_pandas(val_df[['text']])

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Configure 4-bit Quantization (Crucial for 8GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 5. Load the Base Model
    print(f"Loading Base Model ({model_id}) in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"":0},
        attn_implementation="sdpa"
    )
    #model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 6. Configure LoRA (The "Adapter")
    lora_config = LoraConfig(
        r=16, # Rank of the adapter
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 7. Training Arguments
    training_args = SFTConfig(
        output_dir=str(MODEL_DIR / "checkpoints"),
        per_device_train_batch_size=1, # Keep low for 8GB VRAM
        gradient_accumulation_steps=8, # Simulates a batch size of 8
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=20,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=3, # LoRA learns much faster than full fine-tuning
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        max_length=128,
        dataset_text_field="text",
        dataloader_num_workers=4,
    )

    # 8. Train using SFTTrainer (Supervised Fine-Tuning)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("\nStarting LoRA fine-tuning...\n")
    trainer.train()

    # 9. Save the LoRA Adapter
    trainer.model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\n✅ LoRA Adapter saved to: {MODEL_DIR}")


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model_path: Optional[str] = None):
    """Evaluates the LoRA adapted model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from tqdm import tqdm

    mp = model_path or str(MODEL_DIR)
    model_id = "google/gemma-2-2b-it"

    print("\n[Generator] Loading base model and LoRA adapter for evaluation ...")
    
    tokenizer = AutoTokenizer.from_pretrained(mp)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Left padding is required for batched causal generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 1. Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    # 2. Attach LoRA Adapter
    model = PeftModel.from_pretrained(base_model, mp)
    model.eval()

    test_path = DATA_PROC / "generator_test.csv"
    test_df = pd.read_csv(test_path) if test_path.exists() else prepare_training_data()[2]

    print(f"  Test set: {len(test_df)} pairs")

    hypotheses = []
    references = test_df["target_text"].astype(str).tolist()
    
    # Format inputs for Gemma
    def build_prompt(text):
        return f"<bos><start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    
    input_texts = [build_prompt(text) for text in test_df["input_text"].astype(str).tolist()]
    
    batch_size = 8 # Adjust if VRAM fills up

    for i in tqdm(range(0, len(input_texts), batch_size), desc="Evaluating"):
        batch_inputs = input_texts[i : i + batch_size]
        
        enc = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=CFG.max_target_length,
                do_sample=False, # Greedy decoding for exact match
                pad_token_id=tokenizer.eos_token_id
            )
        
        # We only want the *newly generated* text, not the prompt
        for j, out_tokens in enumerate(outputs):
            input_len = enc['input_ids'][j].shape[0]
            new_tokens = out_tokens[input_len:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            hypotheses.append(pred)

    # Clean up VRAM before BERTScore
    del model
    del base_model
    torch.cuda.empty_cache()

    # ── BERTScore ──
    try:
        from bert_score import score as bscore
        _, _, bert_f1 = bscore(
            hypotheses, references,
            lang="other",      
            model_type="bert-base-multilingual-cased", # Lighter model to prevent OOM
            rescale_with_baseline=False,
            verbose=False,
        )
        bert_mean = float(bert_f1.mean())
    except Exception as e:
        bert_mean = -1.0
        print(f"  [WARN] BERTScore failed: {e}")

    # ── Exact match (phrase-level accuracy) ──
    exact = sum(h.strip().lower() == r.strip().lower() for h, r in zip(hypotheses, references))
    exact_acc = exact / len(hypotheses) if hypotheses else 0.0

    # ── Sample outputs ──
    import random
    sample_indices = random.sample(range(len(test_df)), min(5, len(test_df)))
    samples = [{"input": test_df.iloc[i]["input_text"], "reference": references[i], "generated": hypotheses[i]} for i in sample_indices]

    # Print Report
    print("\n" + "=" * 60)
    print("  Idiom Generator — Test Set Results (LoRA)")
    print("=" * 60)
    print(f"  BERTScore F1    : {bert_mean:.4f}")
    print(f"  Exact match     : {exact_acc*100:.1f}%  ({exact}/{len(hypotheses)})")
    print("-" * 60)
    for i, s in enumerate(samples, 1):
        print(f"\n  [{i}] Input     : {s['input']}")
        print(f"      Reference : {s['reference']}")
        print(f"      Generated : {s['generated']}")


# ─── Inference Wrapper ────────────────────────────────────────────────────────

class IdiomGenerator:
    """Lightweight inference wrapper for pipeline.py"""
    def __init__(self, model_path: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        mp = model_path or str(MODEL_DIR)
        model_id = "google/gemma-2-2b-it"

        self.tokenizer = AutoTokenizer.from_pretrained(mp)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        
        base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
        self.model = PeftModel.from_pretrained(base_model, mp)
        self.model.eval()

    def generate(self, text: str, target_lang: str = "ml") -> dict:
        import torch
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        prompt_text = f"translate idiom to {lang_name}: {text.strip()}"
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"

        enc = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **enc, 
                max_new_tokens=CFG.max_target_length, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_len = enc['input_ids'].shape[1]
        generated = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        return {
            "input": text,
            "input_prompt": prompt_text,
            "target_lang": target_lang,
            "generated": generated,
        }

    def generate_top_k(self, text: str, target_lang: str = "ml", top_k: int = 3) -> list[dict]:
        """Generate several candidates for the CLI's --top_k option."""
        import torch

        lang_name = LANG_NAMES.get(target_lang, target_lang)
        prompt_text = f"translate idiom to {lang_name}: {text.strip()}"
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
        enc = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=CFG.max_target_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        results = []
        for rank, tokens in enumerate(out, 1):
            generated = self.tokenizer.decode(tokens[input_len:], skip_special_tokens=True).strip()
            results.append({"rank": rank, "generated": generated, "score": None})
        return results

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
