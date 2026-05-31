import argparse
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PROC  = ROOT / "data" / "processed"
MODEL_DETECTOR  = ROOT / "models" / "idiom_detector"
MODEL_RETRIEVER = ROOT / "models" / "semantic_retriever"
MODEL_GENERATOR = ROOT / "models" / "generator" # Updated to new LoRA path
RESULTS    = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
# DUAL THRESHOLD SYSTEM
DIRECT_MATCH_THRESHOLD = 0.60   # Bypasses the LLM entirely ONLY for near-perfect database matches
RAG_CONTEXT_THRESHOLD = 0.60   # Passes the database row to Gemma as context if it's a reasonable semantic match

LANG_NAMES = {
    "ml": "malayalam",
    "te": "telugu",
    "hi": "hindi",
    "en": "english",
}

# <-- NEW: INDIC TRANSLATION MAPPING -->
INDIC_LANG_MAP = {
    "hi": "hin_Deva",
    "ml": "mal_Mlym",
    "te": "tel_Telu"
}

# ─── Module Loader Helpers ────────────────────────────────────────────────────

def _load_detector():
    try:
        from idiom_detector import IdiomDetector
        if not (MODEL_DETECTOR / "config.json").exists():
            return None, "not trained"
        return IdiomDetector(model_path=str(MODEL_DETECTOR)), "loaded"
    except Exception as e:
        return None, f"error: {str(e)[:120]}"

def _load_retriever():
    try:
        from semantic_retriever import CrossLingualRetriever
        idx_path = MODEL_RETRIEVER / "idiom_index.faiss"
        if not idx_path.exists():
            return None, "index not built"
        return CrossLingualRetriever(), "loaded"
    except Exception as e:
        return None, f"error: {str(e)[:120]}"

def _load_generator():
    """Loads the new Gemma-2-2B model in 4-bit with LoRA adapters."""
    try:
        adapter_config = MODEL_GENERATOR / "adapter_config.json"
        if not adapter_config.exists():
            return None, "not trained — run: python src/generator.py --mode train"
        from peft import PeftModel
        
        model_id = "google/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_GENERATOR))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": 0},
            quantization_config=bnb_config
        )
        
        print(f"Loading LoRA adapter from: {MODEL_GENERATOR}")
        model = PeftModel.from_pretrained(model, str(MODEL_GENERATOR))
        model.eval()
        
        return {"model": model, "tokenizer": tokenizer}, "loaded"
    except Exception as e:
        return None, f"error: {str(e)[:120]}"

def _load_indictrans():
    """Loads IndicTrans2-1B in FP16 precision for literal fallback translation."""
    try:
        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            from IndicTransToolkit.processor import IndicProcessor

        model_name = "ai4bharat/indictrans2-en-indic-1B"
        processor = IndicProcessor(inference=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load in half-precision to save VRAM alongside Gemma
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"": 0} if torch.cuda.is_available() else None,
        )
        model.eval()
        return {"model": model, "tokenizer": tokenizer, "processor": processor}, "loaded"
    except Exception as e:
        return None, f"error: {str(e)[:120]}"

# ─── Core Pipeline ────────────────────────────────────────────────────────────

class IdiomTranslationPipeline:
    def __init__(self, target_lang="ml", verbose=True):
        self.target_lang = target_lang
        self.verbose = verbose

        if verbose: print("\n[Pipeline] Loading modules...")
        self.detector,  self._det_status  = _load_detector()
        self.retriever, self._ret_status  = _load_retriever()
        
        gen_dict, self._gen_status = _load_generator()
        self.gen_model = gen_dict["model"] if gen_dict else None
        self.gen_tokenizer = gen_dict["tokenizer"] if gen_dict else None

        # <-- NEW: Load IndicTrans2 Fallback -->
        indic_dict, self._indic_status = _load_indictrans()
        if indic_dict:
            self.indic_model = indic_dict["model"]
            self.indic_tokenizer = indic_dict["tokenizer"]
            self.indic_processor = indic_dict["processor"]
        else:
            self.indic_model = None

        if verbose: self._print_status()

    def _print_status(self):
        w = 42
        print("=" * w)
        print("  Idiom Translation Pipeline — Status")
        print("=" * w)
        icon = lambda s: "✅" if s == "loaded" else "⚠️ "
        print(f"  {icon(self._det_status)} Detector  : {self._det_status}")
        print(f"  {icon(self._ret_status)} Retriever : {self._ret_status}")
        print(f"  {icon(self._gen_status)} Generator : {self._gen_status}")
        print(f"  {icon(self._indic_status)} Fallback  : {self._indic_status}") # <-- NEW LINE
        print(f"  🌐 Target lang : {self.target_lang}")
        print("=" * w + "\n")

    def _translate_literal(self, text: str) -> str:
        """Executes the IndicTrans2 fallback pipeline."""
        if not self.indic_model:
            return f"[NO_FALLBACK_MODEL_LOADED] {text}"
            
        tgt_lang_code = INDIC_LANG_MAP.get(self.target_lang, "hin_Deva")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Pre-process
        batch = self.indic_processor.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=tgt_lang_code)
        inputs = self.indic_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = self.indic_model.generate(**inputs, use_cache=False, max_length=256, num_beams=5)
            
        # Decode and Post-process
        decoded = self.indic_tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = self.indic_processor.postprocess_batch(decoded, lang=tgt_lang_code)
        
        return translations[0]

   # <-- TWEAK A: Add target_lang to the signature -->
    def translate(self, text: str, target_lang: str = None) -> dict:
        # Override the default language if the Flask route provided one
        if target_lang:
            self.target_lang = target_lang

        t0 = time.time()
        result = {
            "input": text, "output": None, "route": [],
            "is_idiomatic": None, "confidence": None, "retrieval": [], "time_ms": 0,
            "retrieved_context": "None" 
        }

        # ── Step 1: Idiom Detection ──
        if self.detector:
            detection = self.detector.predict(text)
            result["is_idiomatic"], result["confidence"] = detection["is_idiomatic"], detection.get("confidence")
            result["route"].append("detector")
            
            # <-- MODIFIED BLOCK -->
            if not detection["is_idiomatic"]:
                result["route"].append("indictrans2_fallback")
                
                result["output"] = self._translate_literal(text)
                
                result["time_ms"] = int((time.time() - t0) * 1000)
                result["generated_translation"] = result["output"]
                return result
            # <-- END MODIFIED BLOCK -->
        else:
            result["route"].append("detector_skipped")

        # ── Step 2: FAISS Retrieval Evaluation ──
        if self.retriever:
            result["route"].append("retrieval")
            try:
                hits = self.retriever.retrieve(query_text=text, top_k=3, target_lang=self.target_lang)
                result["retrieval"] = hits
                
                if hits:
                    top_hit = hits[0]
                    score = top_hit.get("score", 0)
                    
                    # GATE 1: Direct Exact Match
                    if score >= DIRECT_MATCH_THRESHOLD:
                        result["route"].append("→ match_found")
                        result["output"] = top_hit.get("target_idiom", "[No Translation]")
                        result["retrieved_context"] = f"Perfect Match (Score: {score:.2f})"
                        result["time_ms"] = int((time.time() - t0) * 1000)
                        result["generated_translation"] = result["output"]
                        return result
                    
            except Exception as e:
                result["route"].append("retrieval_error")
        else:
            result["route"].append("retrieval_skipped")

        # ── Step 3: Gemma Generative Fallback ──
        if self.gen_model:
            result["route"].append("gemma_solo")
            try:
                lang_name = LANG_NAMES.get(self.target_lang, self.target_lang).lower()
                
                # If it reached here, FAISS didn't find a high-confidence match
                result["retrieved_context"] = "Database Miss (Fallback to Generator)"
                
                # Match the Gemma chat format used during LoRA fine-tuning.
                # Change your current prompt_text to this:
                # NEW FEW-SHOT PROMPT
                prompt_text = (
                    f"Translate the figurative meaning of the English idiom into natural {lang_name}. Do not translate the words literally.\n\n"
                    f"Example 1:\nIdiom: 'Piece of cake'\nTranslation: വളരെ എളുപ്പമുള്ള കാര്യം\n\n"
                    f"Example 2:\nIdiom: 'Break a leg'\nTranslation: വിജയാശംസകൾ\n\n"
                    f"Now translate this:\nIdiom: '{text.strip()}'\nTranslation:"
                )
                if getattr(self.gen_tokenizer, "chat_template", None):
                    prompt = self.gen_tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    prompt = f"<bos><start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = self.gen_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                with torch.no_grad():
                    outputs = self.gen_model.generate(
                        **inputs, 
                        max_new_tokens=64, 
                        do_sample=False, 
                        pad_token_id=self.gen_tokenizer.eos_token_id
                    )
                
                # Token Slicing
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                decoded_text = self.gen_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                result["output"] = decoded_text
                
            except Exception as e:
                result["route"].append("generator_error")
                result["output"] = f"[UNTRANSLATED: {str(e)}]"
        else:
            result["output"] = "[UNTRANSLATED - No Generator]"

        result["time_ms"] = int((time.time() - t0) * 1000)
        result["generated_translation"] = result["output"] 
        return result

# ─── CLI Entry Point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="End-to-end Idiom Translation Pipeline (Updated)")
    parser.add_argument("--mode", choices=["translate", "status"], default="translate")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--target_lang", type=str, default="ml")
    args = parser.parse_args()

    if args.mode == "status":
        IdiomTranslationPipeline(target_lang=args.target_lang, verbose=True)
        return

    if args.mode == "translate":
        if not args.text: parser.error("--text is required")
        pipe = IdiomTranslationPipeline(target_lang=args.target_lang, verbose=True)
        res = pipe.translate(args.text)
        print(f"\n  Input      : {res['input']}")
        print(f"  Context    : {res['retrieved_context']}")
        print(f"  Output     : {res['output']}")
        print(f"  Route      : {' → '.join(res['route'])}")
        
        # Diagnostic print for FAISS scores
        if res["retrieval"]:
            print(f"\n  Top retrieval hits (FAISS Diagnostics):")
            for i, h in enumerate(res["retrieval"][:3], 1):
                print(f"    [{i}] {h.get('target_idiom')} (score: {h.get('score', 0):.4f})")

if __name__ == "__main__":
    main()
