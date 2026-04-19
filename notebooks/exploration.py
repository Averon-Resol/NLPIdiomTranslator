"""
exploration.py
==============
Person 1 — Data Exploration & Sanity Checks
Author   : Ihsal Riyas

Run this AFTER data_pipeline.py and preprocess.py to:
  - Inspect class balance
  - Check sentence length distributions
  - Print sample rows from each dataset source
  - Confirm splits are clean before training

Run:
    python notebooks/exploration.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"

SEP = "─" * 55


def load_and_check(filename: str) -> pd.DataFrame:
    path = DATA_PROC / filename
    if not path.exists():
        print(f"[MISSING] {filename} — run data_pipeline.py / preprocess.py first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"\n{SEP}")
    print(f"  {filename}  ({len(df)} rows)")
    print(SEP)
    return df


def class_balance(df: pd.DataFrame, name: str):
    if df.empty:
        return
    counts = df["label"].value_counts().rename({0: "Literal", 1: "Idiomatic"})
    total  = len(df)
    print(f"\n  Class balance — {name}:")
    for cls, cnt in counts.items():
        bar = "█" * int(30 * cnt / total)
        print(f"    {cls:10s} {cnt:6d}  ({100*cnt/total:5.1f}%)  {bar}")


def token_length_stats(df: pd.DataFrame, col: str = "text"):
    if df.empty or col not in df.columns:
        return
    lengths = df[col].dropna().str.split().str.len()
    print(f"\n  Token length — {col}:")
    print(f"    min={lengths.min()}  max={lengths.max()}  "
          f"mean={lengths.mean():.1f}  median={lengths.median():.0f}")
    buckets = [0, 5, 10, 20, 30, 50, 999]
    for lo, hi in zip(buckets, buckets[1:]):
        cnt = ((lengths >= lo) & (lengths < hi)).sum()
        print(f"    [{lo:3d}–{hi:3d}) tokens : {cnt:5d} rows")


def source_breakdown(df: pd.DataFrame):
    if df.empty or "source" not in df.columns:
        return
    print("\n  Source breakdown:")
    for src, cnt in df["source"].value_counts().items():
        print(f"    {src:20s} : {cnt}")


def sample_rows(df: pd.DataFrame, col: str = "text", n: int = 5):
    if df.empty:
        return
    print(f"\n  Sample rows (n={n}):")
    sample = df.sample(min(n, len(df)), random_state=42)
    for _, row in sample.iterrows():
        lbl  = "Idiomatic" if row.get("label", 0) == 1 else "Literal"
        text = str(row.get(col, "")).strip()[:90]
        print(f"    [{lbl:9s}] {text}")


def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Checks that no sentence appears in more than one split.
    This is critical — leakage inflates evaluation scores artificially.
    """
    if any(df.empty for df in [train_df, val_df, test_df]):
        return

    print(f"\n{SEP}")
    print("  Leakage Check")
    print(SEP)

    train_texts = set(train_df["text"])
    val_texts   = set(val_df["text"])
    test_texts  = set(test_df["text"])

    tv = train_texts & val_texts
    tt = train_texts & test_texts
    vt = val_texts   & test_texts

    print(f"  Train ∩ Val  : {len(tv)} overlap")
    print(f"  Train ∩ Test : {len(tt)} overlap")
    print(f"  Val   ∩ Test : {len(vt)} overlap")

    if tv or tt or vt:
        print("  ⚠️  Leakage detected! Review preprocess.py deduplication logic.")
    else:
        print("  ✅ No leakage — splits are clean.")


def main():
    print("=" * 55)
    print("  NLP-Project — Data Exploration")
    print("  Person 1: Ihsal Riyas")
    print("=" * 55)

    # ── Unified dataset ──
    unified = load_and_check("unified_idioms.csv")
    if not unified.empty:
        source_breakdown(unified)
        print(f"\n  Languages in dataset:")
        for lang, cnt in unified["source_lang"].value_counts().items():
            print(f"    {lang:5s} : {cnt}")

    # ── Detection-only ──
    det = load_and_check("detection_only.csv")
    if not det.empty:
        class_balance(det, "detection_only")
        # detection_only uses "source_idiom" column
        col = "source_idiom" if "source_idiom" in det.columns else "text"
        token_length_stats(det, col=col)
        sample_rows(det, col=col)

    # ── Splits ──
    train_df = load_and_check("train.csv")
    val_df   = load_and_check("val.csv")
    test_df  = load_and_check("test.csv")

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        class_balance(df, name)
        if not df.empty:
            token_length_stats(df)

    check_leakage(train_df, val_df, test_df)

    # ── Cross-lingual ──
    cross = load_and_check("cross_lingual.csv")
    if not cross.empty:
        print("\n  Language pair coverage:")
        cross["pair"] = cross["source_lang"] + " → " + cross["target_lang"]
        for pair, cnt in cross["pair"].value_counts().head(15).items():
            print(f"    {pair:20s} : {cnt}")

    print(f"\n{'=' * 55}")
    print("  Exploration complete.")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
