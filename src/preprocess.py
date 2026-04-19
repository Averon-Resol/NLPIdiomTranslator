"""
preprocess.py
=============
Person 1 — Data Cleaning & Train/Val/Test Splitting
Author   : Ihsal Riyas

Takes data/processed/detection_only.csv (output of data_pipeline.py)
and produces clean, stratified splits ready for model training.

Output files (data/processed/):
  train.csv    ← 70%
  val.csv      ← 15%
  test.csv     ← 15%

Each file has columns:
  text | label | idiom_string | source

Run:
    python src/preprocess.py
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"


# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Applies basic normalisation to a raw sentence:
      - Strips leading/trailing whitespace
      - Collapses multiple whitespace into one
      - Removes non-printable characters
      - Lowercases (XLM-RoBERTa uses its own casing internally, but
        consistent lowercase helps reproducibility)
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    # Removed the aggressive non-latin regex filter to support Malayalam, Telugu, and Hindi natively
    return text.lower()


def remove_short_sentences(df: pd.DataFrame, min_tokens: int = 3) -> pd.DataFrame:
    """Drop rows where the sentence has fewer than `min_tokens` words."""
    mask = df["text"].str.split().str.len() >= min_tokens
    dropped = (~mask).sum()
    if dropped:
        print(f"  [clean] Dropped {dropped} sentences with < {min_tokens} tokens.")
    return df[mask].copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate (text, label) pairs."""
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"])
    print(f"  [clean] Removed {before - len(df)} duplicate rows.")
    return df.reset_index(drop=True)


# ─── Split ────────────────────────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70 / 15 / 15 split preserving class balance.

    Returns:
        (train_df, val_df, test_df)
    """
    test_frac = 1.0 - train_frac - val_frac

    train_df, temp_df = train_test_split(
        df,
        test_size=val_frac + test_frac,
        random_state=seed,
        stratify=df["label"],
    )
    relative_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_split_stats(train_df, val_df, test_df):
    """Print class distribution for each split."""
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = df["label"].value_counts().rename({0: "Literal", 1: "Idiomatic"})
        total  = len(df)
        print(f"\n  {name} ({total} examples):")
        for cls, cnt in counts.items():
            print(f"    {cls:10s}  {cnt:5d}  ({100*cnt/total:.1f}%)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_preprocessing():
    print("=" * 60)
    print("  NLP-Project — Preprocessing")
    print("  Person 1: Ihsal Riyas")
    print("=" * 60)

    input_path = DATA_PROC / "detection_only.csv"
    if not input_path.exists():
        print(f"\n[ERROR] {input_path} not found.")
        print("  Run `python src/data_pipeline.py` first.")
        return

    df = pd.read_csv(input_path)
    print(f"\n[LOAD] {len(df)} rows loaded from detection_only.csv")

    # ── Rename columns to match internal format ──
    df = df.rename(columns={"source_idiom": "text"})
    df = df[["text", "label", "idiom_string", "source"]].copy()

    # ── Clean ──
    print("\n[CLEAN]")
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip() != ""]     # drop empty strings after cleaning
    df = remove_short_sentences(df, min_tokens=3)
    df = remove_duplicates(df)

    # ── Balance check ──
    counts = df["label"].value_counts()
    print(f"\n  Class distribution after cleaning:")
    print(f"    Idiomatic : {counts.get(1, 0)}")
    print(f"    Literal   : {counts.get(0, 0)}")

    # Warn if heavily imbalanced
    minority = counts.min() if not counts.empty else 0
    majority = counts.max() if not counts.empty else 0
    ratio    = (minority / majority) if majority > 0 else 0
    
    if ratio < 0.4 and majority > 0:
        print(f"\n  [WARN] Class imbalance detected (ratio={ratio:.2f}).")
        print("  Consider using class_weight='balanced' in the trainer.")

    # ── Split ──
    if len(df) < 3:
        print("\n[ERROR] Not enough data to split (need at least 3 rows).")
        return

    # If we only have 1 class due to missing data (e.g., HuggingFace datasets failed to load)
    # we cannot stratify the split.
    if len(df["label"].unique()) < 2:
        print("\n[WARN] Only 1 class present. Falling back to unstratified split...")
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    else:
        print("\n[SPLIT] Stratified 70/15/15 split...")
        train_df, val_df, test_df = stratified_split(df)
        
    print_split_stats(train_df, val_df, test_df)

    # ── Save ──
    train_df.to_csv(DATA_PROC / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(DATA_PROC / "val.csv",   index=False, encoding="utf-8")
    test_df.to_csv(DATA_PROC / "test.csv",  index=False, encoding="utf-8")

    print(f"\n✅ Splits saved to {DATA_PROC}")
    print("   train.csv / val.csv / test.csv")
    print("\nShare these 3 files with Person 2 and Person 3 via GitHub.")


if __name__ == "__main__":
    run_preprocessing()
