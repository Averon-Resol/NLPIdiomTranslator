"""
semantic_retriever.py
=====================
Person 2 — Semantic Similarity & Retrieval Module
Author   : Ramanand Balaji

Builds a cross-lingual embedding index (LaBSE + FAISS) and retrieves
target-language equivalents for a source idiom/sentence.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
RETRIEVER_DIR = ROOT / "models" / "semantic_retriever"

RETRIEVER_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


class CrossLingualRetriever:
    """
    Cross-lingual idiom retrieval:
      1. Encode source phrases with LaBSE
      2. Build FAISS cosine-similarity index
      3. Retrieve nearest target-language equivalents
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        index_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir) if index_dir else RETRIEVER_DIR
        self.index_path = self.index_dir / "idiom_index.faiss"
        self.meta_path = self.index_dir / "metadata.csv"
        self.config_path = self.index_dir / "retriever_config.json"

        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.metadata: Optional[pd.DataFrame] = None

        if self.index_path.exists() and self.meta_path.exists():
            self.load()

    def _load_training_pairs(self, data_path: Optional[str]) -> pd.DataFrame:
        if data_path:
            frames = [pd.read_csv(data_path)]
        else:
            frames = []
            cross_lingual_path = DATA_PROC / "cross_lingual.csv"
            if cross_lingual_path.exists():
                frames.append(pd.read_csv(cross_lingual_path))

            for p in DATA_RAW.glob("*_idioms.csv"):
                frames.append(pd.read_csv(p))

            json_lang_map = {
                "hindi.json": "hi",
                "malayalam.json": "ml",
                "telugu.json": "te",
            }
            for filename, lang_code in json_lang_map.items():
                p = DATA_RAW / filename
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        items = json.load(f)
                    rows = []
                    for item in items if isinstance(items, list) else []:
                        source_text = str(item.get("figurative_meaning", "")).strip()
                        target_text = str(item.get("idiom", "")).strip()
                        if source_text and target_text:
                            rows.append(
                                {
                                    "source_idiom": source_text,
                                    "source_lang": "en",
                                    "target_idiom": target_text,
                                    "target_lang": lang_code,
                                    "source": f"Raw-{filename}",
                                }
                            )
                    if rows:
                        frames.append(pd.DataFrame(rows))
            if not frames:
                raise FileNotFoundError(
                    "No idiom pair data found. Provide --data_path or add idiom files under data/processed or data/raw."
                )

        df = pd.concat(frames, ignore_index=True)
        required_cols = ["source_idiom", "source_lang", "target_idiom", "target_lang"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["source_idiom"] = _normalize_text(df["source_idiom"])
        df["target_idiom"] = _normalize_text(df["target_idiom"])
        df["source_lang"] = _normalize_text(df["source_lang"])
        df["target_lang"] = _normalize_text(df["target_lang"])
        df = df[
            (df["source_idiom"] != "")
            & (df["target_idiom"] != "")
            & (df["source_lang"] != "")
            & (df["target_lang"] != "")
        ].copy()
        # Drop generic parallel corpora if present; keep idiom-centric sources only by default.
        if "source" in df.columns:
            source_col = df["source"].fillna("").astype(str).str.lower()
            df = df[~source_col.eq("kunchukuttan")].copy()
        df = df.drop_duplicates(
            subset=["source_idiom", "source_lang", "target_idiom", "target_lang"]
        ).reset_index(drop=True)
        return df

    def build_index(
        self,
        data_path: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> dict:
        df = self._load_training_pairs(data_path=data_path)

        if source_lang:
            df = df[df["source_lang"] == source_lang].copy()
        if target_lang:
            df = df[df["target_lang"] == target_lang].copy()

        if df.empty:
            raise ValueError("No rows left after language filtering; cannot build index.")

        source_texts = df["source_idiom"].tolist()
        embeddings = self.model.encode(
            source_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings)
        self.metadata = df.reset_index(drop=True)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata.to_csv(self.meta_path, index=False, encoding="utf-8")

        config = {
            "model_name": self.model_name,
            "rows_indexed": len(self.metadata),
            "embedding_dim": embedding_dim,
            "source_langs": sorted(self.metadata["source_lang"].unique().tolist()),
            "target_langs": sorted(self.metadata["target_lang"].unique().tolist()),
        }
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config

    def load(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                "Index files not found. Run with --mode build first."
            )
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_csv(self.meta_path)

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> list[dict]:
        if not query_text or not query_text.strip():
            raise ValueError("query_text must be non-empty.")
        if self.index is None or self.metadata is None:
            self.load()

        query_vec = self.model.encode(
            [query_text.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        search_k = min(max(top_k * 10, top_k), len(self.metadata))
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if idx < 0:
                continue
            row = self.metadata.iloc[idx]
            if source_lang and row["source_lang"] != source_lang:
                continue
            if target_lang and row["target_lang"] != target_lang:
                continue
            results.append(
                {
                    "source_idiom": row["source_idiom"],
                    "source_lang": row["source_lang"],
                    "target_idiom": row["target_idiom"],
                    "target_lang": row["target_lang"],
                    "score": float(round(score, 4)),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def retrieve_with_detector(
        self,
        sentence: str,
        top_k: int = 3,
        target_lang: Optional[str] = None,
        detector_model_path: Optional[str] = None,
    ) -> dict:
        try:
            from src.idiom_detector import IdiomDetector
        except ModuleNotFoundError:
            from idiom_detector import IdiomDetector

        try:
            detector = IdiomDetector(model_path=detector_model_path)
            detection = detector.predict(sentence)
        except (OSError, ValueError, RuntimeError) as exc:
            try:
                retrieval = self.retrieve(
                    query_text=sentence,
                    top_k=top_k,
                    target_lang=target_lang,
                )
            except FileNotFoundError:
                retrieval = []
            raw_detail = str(exc).splitlines()[0]
            if "Unrecognized model in" in raw_detail:
                detail = "invalid or incomplete detector model directory"
            else:
                detail = raw_detail[:240]
            return {
                "input": sentence,
                "detection": {
                    "label": "Unavailable",
                    "confidence": None,
                    "is_idiomatic": None,
                },
                "retrieval": retrieval,
                "message": (
                    "Detector model unavailable; retrieval executed directly. "
                    f"Details: {detail}. "
                    "If retrieval is empty, build index with --mode build first."
                ),
            }

        if not detection["is_idiomatic"]:
            return {
                "input": sentence,
                "detection": detection,
                "retrieval": [],
                "message": "Input predicted as literal; retrieval skipped.",
            }

        try:
            retrieval = self.retrieve(
                query_text=sentence,
                top_k=top_k,
                target_lang=target_lang,
            )
        except FileNotFoundError:
            return {
                "input": sentence,
                "detection": detection,
                "retrieval": [],
                "message": "Retrieval index not found. Run --mode build first.",
            }
        return {
            "input": sentence,
            "detection": detection,
            "retrieval": retrieval,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Person 2 — Semantic Retriever")
    parser.add_argument(
        "--mode",
        choices=["build", "query", "pipeline"],
        required=True,
        help="build index | query index | detector+retrieval pipeline",
    )
    parser.add_argument("--text", type=str, default=None, help="Input text to query")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional CSV path for index building",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default=None,
        help="Language filter for source side",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        help="Language filter for target side",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Top-K retrieval results")
    parser.add_argument(
        "--detector_model_path",
        type=str,
        default=None,
        help="Optional detector model path (for pipeline mode)",
    )
    args = parser.parse_args()

    retriever = CrossLingualRetriever()

    if args.mode == "build":
        config = retriever.build_index(
            data_path=args.data_path,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
        print(json.dumps(config, indent=2))
        print(f"✅ Index saved to: {RETRIEVER_DIR}")
        return

    if not args.text:
        parser.error("--text is required for query and pipeline modes.")

    if args.mode == "query":
        results = retriever.retrieve(
            query_text=args.text,
            top_k=args.top_k,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    output = retriever.retrieve_with_detector(
        sentence=args.text,
        top_k=args.top_k,
        target_lang=args.target_lang,
        detector_model_path=args.detector_model_path,
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
