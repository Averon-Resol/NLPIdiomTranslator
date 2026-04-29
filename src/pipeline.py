"""
pipeline.py
===========
Person 3 — End-to-End Idiom Translation Pipeline
Author   : Dhruv Nair

Ties together all three modules into a single unified pipeline:

  [Part 1] IdiomDetector      (XLM-RoBERTa)   — Ihsal Riyas
  [Part 2] CrossLingualRetriever (LaBSE+FAISS) — Ramanand Balaji
  [Part 3] IdiomGenerator     (mT5-small)      — Dhruv Nair (you)

Flow for each input sentence:
  1. IdiomDetector  → is the phrase idiomatic?
       • NO  → route to StandardNMT (mBART baseline) → output
       • YES → proceed to Step 2
  2. CrossLingualRetriever → is there a close match in the FAISS index?
       • score ≥ threshold → return retrieved equivalent → output
       • score < threshold → fall through to Step 3
  3. IdiomGenerator → generate equivalent with mT5-small → output
  4. Evaluate output against reference (BLEU + BERTScore + exact match)

The pipeline degrades gracefully:
  • If detector model is missing  → assume all input is idiomatic
  • If FAISS index is missing     → skip retrieval, go straight to generator
  • If generator model is missing → return retrieval result or [UNTRANSLATED]

Usage:
  # Translate a single sentence interactively
  python src/pipeline.py --mode translate \\
      --text "spill the beans" --target_lang ml

  # Run full evaluation against the test set and compare vs baseline
  python src/pipeline.py --mode eval --target_lang hi --n_samples 50

  # Show a summary of which modules are loaded
  python src/pipeline.py --mode status
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PROC  = ROOT / "data" / "processed"
MODEL_DETECTOR  = ROOT / "models" / "idiom_detector"
MODEL_RETRIEVER = ROOT / "models" / "semantic_retriever"
MODEL_GENERATOR = ROOT / "models" / "generator"
RESULTS    = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
RETRIEVAL_THRESHOLD = 0.60   # minimum cosine similarity to trust FAISS result
                              # lower this (e.g. 0.45) if retrieval is too sparse

LANG_NAMES = {
    "ml": "malayalam",
    "te": "telugu",
    "hi": "hindi",
    "en": "english",
}


# ─── Module Loader Helpers ────────────────────────────────────────────────────

def _load_detector():
    """
    Loads IdiomDetector from Part 1.
    Returns (detector, status_str).
    On failure returns (None, reason).
    """
    try:
        try:
            from src.idiom_detector import IdiomDetector
        except ModuleNotFoundError:
            from idiom_detector import IdiomDetector

        if not (MODEL_DETECTOR / "config.json").exists():
            return None, "not trained — run: python src/idiom_detector.py --mode train"

        det = IdiomDetector(model_path=str(MODEL_DETECTOR))
        return det, "loaded"
    except Exception as e:
        return None, f"error: {str(e)[:120]}"


def _load_retriever():
    """
    Loads CrossLingualRetriever from Part 2.
    Returns (retriever, status_str).
    """
    try:
        try:
            from src.semantic_retriever import CrossLingualRetriever
        except ModuleNotFoundError:
            from semantic_retriever import CrossLingualRetriever

        idx_path = MODEL_RETRIEVER / "idiom_index.faiss"
        if not idx_path.exists():
            return None, "index not built — run: python src/semantic_retriever.py --mode build"

        ret = CrossLingualRetriever()
        return ret, "loaded"
    except Exception as e:
        return None, f"error: {str(e)[:120]}"


def _load_generator():
    """
    Loads IdiomGenerator from Part 3 (this module's model).
    Returns (generator, status_str).
    """
    try:
        try:
            from src.generator import IdiomGenerator
        except ModuleNotFoundError:
            from generator import IdiomGenerator

        if not (MODEL_GENERATOR / "config.json").exists():
            return None, "not trained — run: python src/generator.py --mode train"

        gen = IdiomGenerator(model_path=str(MODEL_GENERATOR))
        return gen, "loaded"
    except Exception as e:
        return None, f"error: {str(e)[:120]}"


def _standard_nmt(text: str, target_lang: str) -> str:
    """
    Baseline translation using Helsinki-NLP opus-mt models (no fine-tuning).
    Used for non-idiomatic sentences and as a comparison baseline.

    Falls back to a plain string if the model can't be loaded.
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer

        # Helsinki-NLP model names per language
        model_map = {
            "hi": "Helsinki-NLP/opus-mt-en-hi",
            "ml": "Helsinki-NLP/opus-mt-en-ml",
            "te": "Helsinki-NLP/opus-mt-en-te",
        }

        model_name = model_map.get(target_lang)
        if not model_name:
            return f"[NO_BASELINE_FOR_{target_lang.upper()}] {text}"

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model     = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer([text], return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, num_beams=4, max_new_tokens=64)
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translated

    except Exception as e:
        return f"[BASELINE_FAILED: {str(e)[:80]}] {text}"


# ─── Core Pipeline ────────────────────────────────────────────────────────────

class IdiomTranslationPipeline:
    """
    End-to-end idiom-aware translation pipeline.

    Each module is optional — the pipeline degrades gracefully if
    a module isn't trained/built yet:

      detector   missing → assume all input is idiomatic (worst case: more retrieval calls)
      retriever  missing → skip to generator
      generator  missing → return best retrieval result or [UNTRANSLATED]

    Example:
        pipe = IdiomTranslationPipeline(target_lang="ml")
        result = pipe.translate("spill the beans")
        print(result["output"])
        print(result["route"])   # e.g. "detector→retrieval" or "detector→generator"
    """

    def __init__(
        self,
        target_lang          : str   = "ml",
        retrieval_threshold  : float = RETRIEVAL_THRESHOLD,
        verbose              : bool  = True,
    ):
        self.target_lang         = target_lang
        self.retrieval_threshold = retrieval_threshold
        self.verbose             = verbose

        if verbose:
            print("\n[Pipeline] Loading modules...")

        self.detector,  self._det_status  = _load_detector()
        self.retriever, self._ret_status  = _load_retriever()
        self.generator, self._gen_status  = _load_generator()

        if verbose:
            self._print_status()

    def _print_status(self):
        w = 42
        print("=" * w)
        print("  Idiom Translation Pipeline — Status")
        print("=" * w)
        icon = lambda s: "✅" if s == "loaded" else "⚠️ "
        print(f"  {icon(self._det_status)} Detector  : {self._det_status}")
        print(f"  {icon(self._ret_status)} Retriever : {self._ret_status}")
        print(f"  {icon(self._gen_status)} Generator : {self._gen_status}")
        print(f"  🌐 Target lang : {self.target_lang} ({LANG_NAMES.get(self.target_lang, '?')})")
        print(f"  📏 Retrieval threshold : {self.retrieval_threshold}")
        print("=" * w + "\n")

    def translate(self, text: str) -> dict:
        """
        Translate a single sentence through the full pipeline.

        Args:
            text : Source sentence (English or native script idiom).

        Returns a dict with:
            input         — original input
            output        — translated/equivalent phrase
            route         — which modules were used ('detector→retrieval', etc.)
            is_idiomatic  — True/False/None if detector is missing
            retrieval     — top retrieval results (may be empty)
            confidence    — detector confidence (None if unavailable)
            time_ms       — total time in milliseconds
        """
        t0 = time.time()
        result = {
            "input"        : text,
            "output"       : None,
            "route"        : [],
            "is_idiomatic" : None,
            "confidence"   : None,
            "retrieval"    : [],
            "time_ms"      : 0,
        }

        # ── Step 1: Idiom Detection ──────────────────────────────────────────
        if self.detector is not None:
            detection = self.detector.predict(text)
            result["is_idiomatic"] = detection["is_idiomatic"]
            result["confidence"]   = detection.get("confidence")
            result["route"].append("detector")

            if not detection["is_idiomatic"]:
                # Literal input → standard NMT baseline
                result["route"].append("standard_nmt")
                result["output"] = _standard_nmt(text, self.target_lang)
                result["time_ms"] = int((time.time() - t0) * 1000)
                return result
        else:
            # Detector unavailable — treat all input as idiomatic
            result["is_idiomatic"] = None
            result["route"].append("detector_skipped")

        # ── Step 2: FAISS Retrieval ──────────────────────────────────────────
        if self.retriever is not None:
            result["route"].append("retrieval")
            try:
                hits = self.retriever.retrieve(
                    query_text  = text,
                    top_k       = 5,
                    target_lang = self.target_lang,
                )
                result["retrieval"] = hits

                if hits and hits[0]["score"] >= self.retrieval_threshold:
                    result["output"] = hits[0]["target_idiom"]
                    result["route"].append("→ match_found")
                    result["time_ms"] = int((time.time() - t0) * 1000)
                    return result

            except Exception as e:
                result["route"].append(f"retrieval_error({str(e)[:60]})")
        else:
            result["route"].append("retrieval_skipped")

        # ── Step 3: mT5 Generator (fallback) ────────────────────────────────
        if self.generator is not None:
            result["route"].append("generator")
            try:
                gen_out = self.generator.generate(
                    text        = text,
                    target_lang = self.target_lang,
                )
                result["output"] = gen_out["generated"]
            except Exception as e:
                result["route"].append(f"generator_error({str(e)[:60]})")
                result["output"] = "[UNTRANSLATED]"
        else:
            # No generator — return best retrieval result if any, else fail
            if result["retrieval"]:
                result["output"] = result["retrieval"][0]["target_idiom"]
                result["route"].append("retrieval_best_effort")
            else:
                result["output"] = "[UNTRANSLATED — no generator or retrieval match]"

        result["time_ms"] = int((time.time() - t0) * 1000)
        return result

    def translate_batch(self, texts: list[str]) -> list[dict]:
        """Translate a list of sentences, returning a list of result dicts."""
        return [self.translate(t) for t in texts]


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_pipeline(
    target_lang : str = "ml",
    n_samples   : int = 50,
    threshold   : float = RETRIEVAL_THRESHOLD,
):
    """
    Runs the full pipeline against the generator test set and the
    standard NMT baseline side-by-side. Writes a comparison report to
    results/pipeline_eval_<lang>.txt.

    Metrics reported for both systems:
      • BLEU score
      • BERTScore F1
      • Exact match %
      • Route distribution (how many went through retrieval vs generator vs NMT)
    """
    print(f"\n[Pipeline Eval] target_lang={target_lang}, n_samples={n_samples}")

    # Load test set
    test_path = DATA_PROC / "generator_test.csv"
    if not test_path.exists():
        # fall back to cross_lingual
        test_path = DATA_PROC / "cross_lingual.csv"

    df = pd.read_csv(test_path)

    # Filter to target language and to EN source only
    lang_col = "target_lang" if "target_lang" in df.columns else "target_lang"
    src_col  = "source_lang" if "source_lang" in df.columns else "source_lang"

    df = df[
        (df[lang_col] == target_lang) &
        (df[src_col]  == "en")
    ].dropna(subset=["source_idiom" if "source_idiom" in df.columns else "input_text",
                     "target_idiom" if "target_idiom" in df.columns else "target_text"])

    if df.empty:
        print(f"[WARN] No test pairs found for target_lang={target_lang}. Exiting.")
        return

    # Normalise column names
    if "input_text" in df.columns:
        df = df.rename(columns={"input_text": "source_idiom", "target_text": "target_idiom"})

    df = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)
    print(f"  Evaluating on {len(df)} pairs...")

    pipe = IdiomTranslationPipeline(
        target_lang         = target_lang,
        retrieval_threshold = threshold,
        verbose             = True,
    )

    pipeline_outputs  = []
    baseline_outputs  = []
    references        = []
    route_counts      = {}

    for _, row in df.iterrows():
        src = str(row["source_idiom"]).strip()
        ref = str(row["target_idiom"]).strip()

        # Pipeline
        res = pipe.translate(src)
        hyp = res["output"] or ""
        pipeline_outputs.append(hyp)
        references.append(ref)

        route_key = " → ".join(res["route"])
        route_counts[route_key] = route_counts.get(route_key, 0) + 1

        # Baseline (standard NMT, no idiom awareness)
        baseline_outputs.append(_standard_nmt(src, target_lang))

    # ── BLEU ──
    try:
        import sacrebleu
        pipe_bleu = sacrebleu.corpus_bleu(pipeline_outputs, [references]).score
        base_bleu = sacrebleu.corpus_bleu(baseline_outputs,  [references]).score
    except Exception as e:
        pipe_bleu = base_bleu = -1.0
        print(f"  [WARN] BLEU failed: {e}")

    # ── BERTScore ──
    try:
        from bert_score import score as bscore
        _, _, pf1 = bscore(pipeline_outputs, references, lang="en", verbose=False)
        _, _, bf1 = bscore(baseline_outputs,  references, lang="en", verbose=False)
        pipe_bert = float(pf1.mean())
        base_bert = float(bf1.mean())
    except Exception as e:
        pipe_bert = base_bert = -1.0
        print(f"  [WARN] BERTScore failed: {e}")

    # ── Exact match ──
    pipe_exact = sum(h.strip().lower() == r.strip().lower()
                     for h, r in zip(pipeline_outputs, references)) / len(references)
    base_exact = sum(h.strip().lower() == r.strip().lower()
                     for h, r in zip(baseline_outputs, references)) / len(references)

    # ── Sample outputs ──
    sample_idx = random.sample(range(len(df)), min(5, len(df)))
    samples = []
    for i in sample_idx:
        samples.append({
            "source"   : df.iloc[i]["source_idiom"],
            "reference": references[i],
            "pipeline" : pipeline_outputs[i],
            "baseline" : baseline_outputs[i],
        })

    # ── Build report ──
    delta_bleu = pipe_bleu - base_bleu
    delta_bert = pipe_bert - base_bert

    lines = [
        "=" * 64,
        f"  Pipeline Evaluation Report  |  target_lang = {target_lang}",
        "=" * 64,
        "",
        f"  Test pairs   : {len(df)}",
        f"  Threshold    : {threshold}",
        "",
        f"  {'Metric':<20} {'Pipeline':>12} {'Baseline':>12} {'Δ':>8}",
        f"  {'-'*20} {'-'*12} {'-'*12} {'-'*8}",
        f"  {'BLEU':<20} {pipe_bleu:>12.2f} {base_bleu:>12.2f} {delta_bleu:>+8.2f}",
        f"  {'BERTScore F1':<20} {pipe_bert:>12.4f} {base_bert:>12.4f} {delta_bert:>+8.4f}",
        f"  {'Exact Match %':<20} {pipe_exact*100:>11.1f}% {base_exact*100:>11.1f}% {(pipe_exact-base_exact)*100:>+7.1f}%",
        "",
        "  Route distribution:",
        *[f"    {k:<45} {v:>4} samples" for k, v in sorted(route_counts.items(), key=lambda x: -x[1])],
        "",
        "-" * 64,
        "  Sample Outputs (5 random examples)",
        "-" * 64,
    ]
    for i, s in enumerate(samples, 1):
        lines += [
            f"\n  [{i}] Source    : {s['source']}",
            f"      Reference : {s['reference']}",
            f"      Pipeline  : {s['pipeline']}",
            f"      Baseline  : {s['baseline']}",
        ]

    report = "\n".join(lines)
    out_path = RESULTS / f"pipeline_eval_{target_lang}.txt"
    out_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\n✅ Report saved to {out_path}")

    return {
        "pipeline_bleu": pipe_bleu,
        "baseline_bleu": base_bleu,
        "pipeline_bert": pipe_bert,
        "baseline_bert": base_bert,
        "pipeline_exact": pipe_exact,
        "baseline_exact": base_exact,
    }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Idiom Translation Pipeline (Person 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["translate", "eval", "status"],
        default="translate",
        help="translate | eval | status",
    )
    parser.add_argument("--text",       type=str,   default=None,  help="Input sentence (translate mode)")
    parser.add_argument("--target_lang",type=str,   default="ml",  help="Target language: ml | te | hi")
    parser.add_argument("--n_samples",  type=int,   default=50,    help="Number of test pairs to evaluate")
    parser.add_argument("--threshold",  type=float, default=RETRIEVAL_THRESHOLD,
                        help=f"Retrieval similarity threshold (default {RETRIEVAL_THRESHOLD})")
    args = parser.parse_args()

    if args.mode == "status":
        pipe = IdiomTranslationPipeline(target_lang=args.target_lang, verbose=True)
        return

    if args.mode == "translate":
        if not args.text:
            parser.error("--text is required with --mode translate")
        pipe = IdiomTranslationPipeline(
            target_lang         = args.target_lang,
            retrieval_threshold = args.threshold,
            verbose             = True,
        )
        result = pipe.translate(args.text)
        print(f"\n  Input      : {result['input']}")
        print(f"  Output     : {result['output']}")
        print(f"  Route      : {' → '.join(result['route'])}")
        print(f"  Idiomatic  : {result['is_idiomatic']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Time (ms)  : {result['time_ms']}")
        if result["retrieval"]:
            print(f"\n  Top retrieval hits:")
            for i, h in enumerate(result["retrieval"][:3], 1):
                print(f"    [{i}] {h['target_idiom']}  (score: {h['score']})")
        return

    if args.mode == "eval":
        evaluate_pipeline(
            target_lang = args.target_lang,
            n_samples   = args.n_samples,
            threshold   = args.threshold,
        )


if __name__ == "__main__":
    main()
