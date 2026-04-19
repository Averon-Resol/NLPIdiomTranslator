"""
idiom_detector.py
=================
Person 1 — Idiom Detection Module (XLM-RoBERTa Fine-tuning)
Author   : Ihsal Riyas

Fine-tunes XLM-RoBERTa-base as a binary sequence classifier:
  Label 0 → Literal
  Label 1 → Idiomatic

Inputs : data/processed/train.csv, val.csv, test.csv
Outputs: models/idiom_detector/   (saved model + tokenizer)
         results/detection_results.txt

Usage:
  # Training
  python src/idiom_detector.py --mode train

  # Evaluate saved model
  python src/idiom_detector.py --mode eval

  # Predict on a single sentence
  python src/idiom_detector.py --mode predict --text "It's raining cats and dogs"
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PROC  = ROOT / "data" / "processed"
MODEL_DIR  = ROOT / "models" / "idiom_detector"
RESULTS    = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class DetectorConfig:
    """
    Central configuration for the detector.
    Adjust these hyperparameters when experimenting.
    """
    model_name        : str   = "xlm-roberta-base"
    max_length        : int   = 128       # max token length per sentence
    train_batch_size  : int   = 16        # reduce to 8 if GPU OOM
    eval_batch_size   : int   = 32
    learning_rate     : float = 2e-5
    num_epochs        : int   = 5
    warmup_steps      : int   = 200
    weight_decay      : float = 0.01
    seed              : int   = 42
    early_stopping    : int   = 2         # patience (epochs)
    id2label          : dict  = field(default_factory=lambda: {0: "Literal", 1: "Idiomatic"})
    label2id          : dict  = field(default_factory=lambda: {"Literal": 0, "Idiomatic": 1})


CFG = DetectorConfig()


# ─── Dataset Class ────────────────────────────────────────────────────────────

class IdiomDataset(Dataset):
    """
    PyTorch Dataset for idiom detection.
    Tokenises sentences with the XLM-RoBERTa tokenizer.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.labels     = df["label"].tolist()
        self.encodings  = tokenizer(
            df["text"].tolist(),
            truncation     = True,
            padding        = "max_length",
            max_length     = max_length,
            return_tensors = "pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Called by HuggingFace Trainer after each eval step.
    Returns accuracy and macro-F1.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall"   : recall_score(labels, preds, average="macro", zero_division=0),
    }


# ─── Training ──────────────────────────────────────────────────────────────────

def train():
    """
    Full training loop:
      1. Load CSV splits
      2. Tokenise with XLM-RoBERTa tokenizer
      3. Fine-tune with HuggingFace Trainer
      4. Save model to models/idiom_detector/
    """
    set_seed(CFG.seed)

    print("=" * 60)
    print("  Idiom Detector — Training")
    print(f"  Model : {CFG.model_name}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # ── Load data ──
    for split in ("train", "val", "test"):
        path = DATA_PROC / f"{split}.csv"
        if not path.exists():
            print(f"[ERROR] {path} not found. Run preprocess.py first.")
            return

    train_df = pd.read_csv(DATA_PROC / "train.csv")
    val_df   = pd.read_csv(DATA_PROC / "val.csv")

    print(f"\nLoaded  train={len(train_df)}  val={len(val_df)}")

    # ── Tokenizer ──
    print(f"\nLoading tokenizer: {CFG.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    train_dataset = IdiomDataset(train_df, tokenizer, CFG.max_length)
    val_dataset   = IdiomDataset(val_df,   tokenizer, CFG.max_length)

    # ── Model ──
    print(f"Loading model: {CFG.model_name} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name,
        num_labels = 2,
        id2label   = CFG.id2label,
        label2id   = CFG.label2id,
    )

    # ── Training Arguments ──
    training_args = TrainingArguments(
        output_dir              = str(MODEL_DIR / "checkpoints"),
        num_train_epochs        = CFG.num_epochs,
        per_device_train_batch_size = CFG.train_batch_size,
        per_device_eval_batch_size  = CFG.eval_batch_size,
        learning_rate           = CFG.learning_rate,
        warmup_steps            = CFG.warmup_steps,
        weight_decay            = CFG.weight_decay,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1",
        greater_is_better       = True,
        logging_dir             = str(RESULTS / "logs"),
        logging_steps           = 50,
        seed                    = CFG.seed,
        report_to               = "none",        # set "wandb" if you have wandb set up
        fp16                    = torch.cuda.is_available(),  # mixed precision on GPU
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping)],
    )

    # ── Train ──
    print("\nStarting training...\n")
    trainer.train()

    # ── Save best model ──
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\n✅ Best model saved to: {MODEL_DIR}")

    # Save config
    with open(MODEL_DIR / "detector_config.json", "w") as f:
        json.dump(
            {k: v for k, v in CFG.__dict__.items() if not callable(v)},
            f, indent=2,
        )

    print("\nRunning final evaluation on test set...")
    evaluate()


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model_path: Optional[str] = None):
    """
    Loads the saved model and evaluates on the test split.
    Writes a classification report to results/detection_results.txt.
    """
    mp          = model_path or str(MODEL_DIR)
    tokenizer   = AutoTokenizer.from_pretrained(mp)
    model       = AutoModelForSequenceClassification.from_pretrained(mp)
    model.eval()

    test_df     = pd.read_csv(DATA_PROC / "test.csv")
    test_dataset = IdiomDataset(test_df, tokenizer, CFG.max_length)

    trainer = Trainer(
        model           = model,
        compute_metrics = compute_metrics,
    )

    results = trainer.predict(test_dataset)
    preds   = np.argmax(results.predictions, axis=-1)
    labels  = results.label_ids

    report  = classification_report(
        labels, preds,
        target_names=["Literal", "Idiomatic"],
        digits=4,
    )
    cm = confusion_matrix(labels, preds)

    output = (
        "=" * 60 + "\n"
        "  Idiom Detector — Test Set Results\n"
        "=" * 60 + "\n\n"
        + report +
        "\nConfusion Matrix (rows=Actual, cols=Predicted):\n"
        "             Literal  Idiomatic\n"
        f"  Literal    {cm[0][0]:6d}   {cm[0][1]:6d}\n"
        f"  Idiomatic  {cm[1][0]:6d}   {cm[1][1]:6d}\n"
    )

    out_path = RESULTS / "detection_results.txt"
    out_path.write_text(output, encoding="utf-8")

    print(output)
    print(f"Results saved to {out_path}")
    return results.metrics


# ─── Inference ────────────────────────────────────────────────────────────────

class IdiomDetector:
    """
    Lightweight inference wrapper.
    Used by Person 3 to plug detection into the full pipeline.

    Example:
        detector = IdiomDetector()
        result   = detector.predict("It's raining cats and dogs.")
        # → {"label": "Idiomatic", "confidence": 0.97, "is_idiomatic": True}
    """

    def __init__(self, model_path: Optional[str] = None):
        mp              = model_path or str(MODEL_DIR)
        self.tokenizer  = AutoTokenizer.from_pretrained(mp)
        self.model      = AutoModelForSequenceClassification.from_pretrained(mp)
        self.model.eval()
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[IdiomDetector] Loaded from {mp} | Device: {self.device}")

    def predict(self, text: str) -> dict:
        """
        Predict whether a sentence is idiomatic or literal.

        Args:
            text: A single input sentence.

        Returns:
            dict with keys:
              - label        : "Idiomatic" or "Literal"
              - confidence   : float between 0-1
              - is_idiomatic : bool
        """
        inputs  = self.tokenizer(
            text,
            return_tensors = "pt",
            truncation     = True,
            max_length     = CFG.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1).squeeze()

        pred_id    = probs.argmax().item()
        confidence = probs[pred_id].item()
        label      = CFG.id2label[pred_id]

        return {
            "label"        : label,
            "confidence"   : round(confidence, 4),
            "is_idiomatic" : pred_id == 1,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run predict() on a list of sentences."""
        return [self.predict(t) for t in texts]


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Idiom Detector — XLM-RoBERTa")
    parser.add_argument(
        "--mode", choices=["train", "eval", "predict"], default="train",
        help="train | eval | predict",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Sentence to classify (used with --mode predict)",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to a saved model directory (optional override)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "eval":
        print("Evaluating saved model on test set...")
        evaluate(model_path=args.model_path)

    elif args.mode == "predict":
        if not args.text:
            parser.error("--text is required with --mode predict")
        detector = IdiomDetector(model_path=args.model_path)
        result   = detector.predict(args.text)
        print(f"\nInput      : {args.text}")
        print(f"Prediction : {result['label']}")
        print(f"Confidence : {result['confidence']*100:.1f}%")
        print(f"Idiomatic  : {result['is_idiomatic']}")


if __name__ == "__main__":
    main()
