"""
data_pipeline.py
================
Person 1 — Data Collection & Consolidation
Author   : Ihsal Riyas

Downloads and merges the following datasets into a unified idiom pair CSV:
  - PIE-English        (idiom detection, binary labels)
  - MAGPIE             (idiom detection, binary labels)
  - LIDIOMS            (cross-lingual idiom pairs — used by Person 2)
  - Kunchukuttan et al. IIT Bombay English-Hindi corpus (Indian language coverage)

Output schema (shared with the whole team):
  source_idiom | source_lang | target_idiom | target_lang | label | split

Run:
    python src/data_pipeline.py
"""

import os
import json
import csv
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ─── MAGPIE ───────────────────────────────────────────────────────────────────
# MAGPIE is a large-scale idiom dataset with binary idiomatic/literal labels.
# Switched from hsseinmz/magpie (private/deleted) to gsarti/magpie (public mirror)
MAGPIE_HF_ID = "gsarti/magpie"

def download_magpie() -> pd.DataFrame:
    """
    Loads MAGPIE from HuggingFace Datasets.
    Returns a DataFrame with columns: text, label (1=idiomatic, 0=literal), idiom.
    """
    print("\n[MAGPIE] Loading from HuggingFace Datasets...")
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset(MAGPIE_HF_ID, trust_remote_code=True)
        except Exception:
            ds = load_dataset(MAGPIE_HF_ID)

        rows = []
        for split_name, split_data in ds.items():
            for example in tqdm(split_data, desc=f"  MAGPIE {split_name}"):
                rows.append({
                    "source_idiom" : example.get("sentence", example.get("text", "")),
                    "source_lang"  : "en",
                    "target_idiom" : "",          # detection only — no target needed
                    "target_lang"  : "",
                    # MAGPIE uses 'idiomatic' or 'literal' string labels
                    "label"        : 1 if str(example.get("label", "")).lower() == "idiomatic" else 0,
                    "idiom_string" : example.get("idiom", ""),
                    "split"        : split_name,
                    "source"       : "MAGPIE",
                })
        df = pd.DataFrame(rows)
        print(f"  → {len(df)} examples loaded from MAGPIE.")
        return df

    except Exception as e:
        print(f"  [WARN] Could not load MAGPIE via HuggingFace: {e}")
        print("  Falling back to manual CSV if available...")
        fallback = DATA_RAW / "magpie.csv"
        if fallback.exists():
            return pd.read_csv(fallback)
        return pd.DataFrame()


# ─── PIE-English ──────────────────────────────────────────────────────────────
# PIE-English: token-level idiom detection corpus.
# We load it as sentence-level binary classification (idiomatic sentence = 1).
# HuggingFace Hub ID: hsseinmz/pie
PIE_HF_ID = "hsseinmz/pie"

def download_pie() -> pd.DataFrame:
    """
    Loads PIE-English from HuggingFace Datasets.
    Returns a DataFrame aligned with the shared schema.
    """
    print("\n[PIE-English] Loading from HuggingFace Datasets...")
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset(PIE_HF_ID, trust_remote_code=True)
        except Exception:
            ds = load_dataset(PIE_HF_ID)

        rows = []
        for split_name, split_data in ds.items():
            for example in tqdm(split_data, desc=f"  PIE {split_name}"):
                # PIE stores token-level BIO tags; we derive sentence-level label
                tags = example.get("tags", example.get("ner_tags", []))
                is_idiomatic = int(any(t != 0 for t in tags)) if tags else 0

                # Reconstruct sentence from tokens
                tokens = example.get("tokens", example.get("words", []))
                sentence = " ".join(tokens) if tokens else example.get("sentence", "")

                rows.append({
                    "source_idiom" : sentence,
                    "source_lang"  : "en",
                    "target_idiom" : "",
                    "target_lang"  : "",
                    "label"        : is_idiomatic,
                    "idiom_string" : example.get("idiom", ""),
                    "split"        : split_name,
                    "source"       : "PIE-English",
                })
        df = pd.DataFrame(rows)
        print(f"  → {len(df)} examples loaded from PIE-English.")
        return df

    except Exception as e:
        print(f"  [WARN] Could not load PIE-English via HuggingFace: {e}")
        fallback = DATA_RAW / "pie.csv"
        if fallback.exists():
            return pd.read_csv(fallback)
        return pd.DataFrame()


# ─── LIDIOMS (Cross-lingual) ──────────────────────────────────────────────────
# LIDIOMS provides idiom translations across many languages.
# Useful for Person 2 (embedding alignment) and Person 3 (generation).
LIDIOMS_URL = "https://zenodo.org/record/5765708/files/lidioms.zip"

def download_lidioms() -> pd.DataFrame:
    """
    Downloads LIDIOMS from Zenodo and extracts cross-lingual idiom pairs.
    Falls back to an empty frame if unreachable.
    """
    print("\n[LIDIOMS] Downloading from Zenodo...")
    dest_dir = DATA_RAW / "lidioms"
    dest_dir.mkdir(exist_ok=True)

    try:
        resp = requests.get(LIDIOMS_URL, timeout=60, stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc="  Downloading") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                buf.write(chunk)
                pbar.update(len(chunk))

        with zipfile.ZipFile(buf) as zf:
            zf.extractall(dest_dir)
        print(f"  → Extracted to {dest_dir}")

    except Exception as e:
        print(f"  [WARN] Could not download LIDIOMS: {e}")
        print("  If you have lidioms files, place JSON files in data/raw/lidioms/")

    # Parse extracted JSON files
    rows = []
    for json_file in dest_dir.rglob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # LIDIOMS structure: list of idiom objects with language entries
            if isinstance(data, list):
                for entry in data:
                    languages = entry.get("languages", {})
                    lang_keys = list(languages.keys())
                    # Create all pairwise combinations
                    for i, src_lang in enumerate(lang_keys):
                        for tgt_lang in lang_keys[i+1:]:
                            src_expr = languages[src_lang].get("expression", "")
                            tgt_expr = languages[tgt_lang].get("expression", "")
                            if src_expr and tgt_expr:
                                rows.append({
                                    "source_idiom" : src_expr,
                                    "source_lang"  : src_lang,
                                    "target_idiom" : tgt_expr,
                                    "target_lang"  : tgt_lang,
                                    "label"        : 1,   # all LIDIOMS are idiomatic
                                    "idiom_string" : src_expr,
                                    "split"        : "unlabeled",
                                    "source"       : "LIDIOMS",
                                })
        except Exception as e:
            print(f"  [WARN] Failed to parse {json_file.name}: {e}")

    df = pd.DataFrame(rows)
    print(f"  → {len(df)} cross-lingual pairs loaded from LIDIOMS.")
    return df


# ─── Kunchukuttan (IIT Bombay English-Hindi) ──────────────────────────────────
# Paper 7 — Kunchukuttan et al.
# This is a large English↔Hindi parallel corpus.
# We use it to add Indian language (Hindi) sentence pairs to the cross-lingual set.
# HuggingFace Hub ID: cfilt/iitb-english-hindi
# We take a sample (10 000 pairs) to keep things manageable.
KUNCHUKUTTAN_HF_ID = "cfilt/iitb-english-hindi"
KUNCHUKUTTAN_SAMPLE = 10_000   # number of sentence pairs to use

def download_kunchukuttan() -> pd.DataFrame:
    """
    Loads the IIT Bombay English-Hindi parallel corpus from HuggingFace.

    What is this corpus?
    --------------------
    Imagine a giant table with two columns:
      Column A: an English sentence
      Column B: the same sentence in Hindi
    This gives us real English-Hindi translation pairs.
    We include it so that Person 2 (Ramanand) has Hindi sentence pairs
    to work with when building the cross-lingual embedding space.

    Returns a DataFrame aligned with the shared schema.
    All rows have label=1 (we treat parallel sentences as meaningful pairs,
    not literal/idiomatic — Person 2 will handle them separately).
    """
    print("\n[Kunchukuttan] Loading IIT Bombay English-Hindi corpus...")
    try:
        from datasets import load_dataset
        try:
            # Load only the 'train' split and take a random sample
            ds = load_dataset(KUNCHUKUTTAN_HF_ID, split="train", trust_remote_code=True)
        except Exception:
            ds = load_dataset(KUNCHUKUTTAN_HF_ID, split="train")

        # Take a manageable sample
        total   = len(ds)
        sample  = min(KUNCHUKUTTAN_SAMPLE, total)
        indices = list(range(0, total, max(1, total // sample)))[:sample]
        rows    = []

        for idx in tqdm(indices, desc="  Kunchukuttan sample"):
            example = ds[idx]
            # The dataset has a 'translation' field: {"en": "...", "hi": "..."}
            translation = example.get("translation", {})
            en_text     = translation.get("en", "").strip()
            hi_text     = translation.get("hi", "").strip()

            if en_text and hi_text:
                rows.append({
                    "source_idiom" : en_text,
                    "source_lang"  : "en",
                    "target_idiom" : hi_text,
                    "target_lang"  : "hi",
                    "label"        : -1,   # -1 = parallel pair, not detection label
                    "idiom_string" : "",
                    "split"        : "train",
                    "source"       : "Kunchukuttan",
                })

        df = pd.DataFrame(rows)
        print(f"  → {len(df)} English-Hindi pairs loaded from Kunchukuttan corpus.")
        return df

    except Exception as e:
        print(f"  [WARN] Could not load Kunchukuttan corpus: {e}")
        print("  Skipping — this dataset is optional (used for Hindi coverage).")
        return pd.DataFrame()


# ─── Kaggle Indian Idioms ─────────────────────────────────────────────────────────
# Kaggle JSON format: Contains "idiom", "literal_meaning", "figurative_meaning", etc.

def download_kaggle_json(filename: str, lang_code: str) -> pd.DataFrame:
    """Helper to parse the Kaggle JSON files and align them with the shared schema."""
    json_path = DATA_RAW / filename
    if not json_path.exists():
        print(f"  [WARN] {filename} not found in data/raw/")
        return pd.DataFrame()
    
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        rows = []
        for item in data:
            # 1. Native to English (For Detector and Retriever)
            rows.append({
                "source_idiom" : item.get("idiom", ""),
                "source_lang"  : lang_code,
                "target_idiom" : item.get("figurative_meaning", ""),
                "target_lang"  : "en",
                "label"        : 1,
                "idiom_string" : item.get("idiom", ""),
                "split"        : "train",
                "source"       : "Kaggle-Indian-Idioms",
            })
            
            # 2. English to Native (CRITICAL FOR GENERATOR)
            rows.append({
                "source_idiom" : item.get("figurative_meaning", ""),
                "source_lang"  : "en",
                "target_idiom" : item.get("idiom", ""),
                "target_lang"  : lang_code,
                "label"        : 1,
                "idiom_string" : item.get("idiom", ""),
                "split"        : "train",
                "source"       : "Kaggle-Indian-Idioms-Reverse",
            })
            
        df = pd.DataFrame(rows)
        print(f"  → {len(df)} examples (both directions) loaded from {filename}.")
        return df
    except Exception as e:
        print(f"  [ERROR] Failed to load {filename}: {e}")
        return pd.DataFrame()

def download_malayalam_idioms() -> pd.DataFrame:
    print("\n[Malayalam] Loading from Kaggle JSON dataset...")
    return download_kaggle_json("malayalam.json", "ml")

def download_telugu_idioms() -> pd.DataFrame:
    print("\n[Telugu] Loading from Kaggle JSON dataset...")
    return download_kaggle_json("telugu.json", "te")

def download_hindi_idioms() -> pd.DataFrame:
    print("\n[Hindi] Loading from Kaggle JSON dataset...")
    return download_kaggle_json("hindi.json", "hi")


# ─── Merge & Save ─────────────────────────────────────────────────────────────

def build_unified_dataset():
    """
    Downloads all datasets, merges them, and writes:
      data/processed/unified_idioms.csv   — full merged set
      data/processed/detection_only.csv   — en-only rows for Person 1's classifier
      data/processed/cross_lingual.csv    — rows with target_idiom for Person 2 & 3
      data/processed/hindi_pairs.csv      — English-Hindi pairs from Kunchukuttan
    """
    print("=" * 60)
    print("  NLP-Project — Data Pipeline")
    print("  Person 1: Ihsal Riyas")
    print("=" * 60)

    frames = []

    magpie_df = download_magpie()
    if not magpie_df.empty:
        frames.append(magpie_df)

    pie_df = download_pie()
    if not pie_df.empty:
        frames.append(pie_df)

    lidioms_df = download_lidioms()
    if not lidioms_df.empty:
        frames.append(lidioms_df)

    kunchukuttan_df = download_kunchukuttan()
    if not kunchukuttan_df.empty:
        frames.append(kunchukuttan_df)
        # Also save Hindi pairs as a separate file for Person 2 & 3
        hindi_out = DATA_PROC / "hindi_pairs.csv"
        kunchukuttan_df.to_csv(hindi_out, index=False, encoding="utf-8")
        print(f"[SAVED] hindi_pairs.csv     → {len(kunchukuttan_df)} rows (for Ramanand & Dhruv)")

    ml_df = download_malayalam_idioms()
    if not ml_df.empty:
        frames.append(ml_df)
        
    te_df = download_telugu_idioms()
    if not te_df.empty:
        frames.append(te_df)

    hi_df = download_hindi_idioms()
    if not hi_df.empty:
        frames.append(hi_df)

    if not frames:
        print("\n[ERROR] No data was loaded. Check your internet connection or raw files.")
        return

    unified = pd.concat(frames, ignore_index=True)
    unified.drop_duplicates(subset=["source_idiom", "source_lang", "label"], inplace=True)
    unified.reset_index(drop=True, inplace=True)

    # ── Write full unified set ──
    out_unified = DATA_PROC / "unified_idioms.csv"
    unified.to_csv(out_unified, index=False, encoding="utf-8")
    print(f"\n[SAVED] unified_idioms.csv  → {len(unified)} rows")

    # ── Detection-only subset (sentences with binary labels) ──
    # Now including ALL language idioms that have a valid 0 or 1 label
    detection_df = unified[
        (unified["label"].isin([0, 1]))
    ].copy()
    out_det = DATA_PROC / "detection_only.csv"
    detection_df.to_csv(out_det, index=False, encoding="utf-8")
    print(f"[SAVED] detection_only.csv  → {len(detection_df)} rows")

    # ── Cross-lingual subset for Person 2 & 3 ──
    cross_df = unified[unified["target_idiom"].notna() & (unified["target_idiom"].str.strip() != "")].copy()
    out_cross = DATA_PROC / "cross_lingual.csv"
    cross_df.to_csv(out_cross, index=False, encoding="utf-8")
    print(f"[SAVED] cross_lingual.csv   → {len(cross_df)} rows")

    print("\n✅ Data pipeline complete!")
    print(f"   Files written to: {DATA_PROC}")
    print("\nLabel distribution (detection set):")
    print(detection_df["label"].value_counts().rename({0: "Literal", 1: "Idiomatic"}).to_string())


if __name__ == "__main__":
    build_unified_dataset()
