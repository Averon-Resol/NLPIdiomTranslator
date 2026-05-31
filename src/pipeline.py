from src.idiom_detector import IdiomDetector
from src.semantic_retriever import CrossLingualRetriever  # FIX 1: Correct class name
from src.generator import GroqTranslator

class IdiomTranslationPipeline:
    def __init__(self):
        print("Initializing Idiom Detector (Local)...")
        self.detector = IdiomDetector()
        
        print("Initializing Semantic Retriever (Local)...")
        self.retriever = CrossLingualRetriever()  # FIX 1: Correct class name
        
        print("Initializing Groq API Translator (Cloud)...")
        self.generator = GroqTranslator()

    def process_text(self, text, target_language="malayalam"):
        if not text or not text.strip():
            return []

        detection_result = self.detector.predict(text)
        
        if not detection_result.get("is_idiomatic", False):
            return []
        
        # 1. Map the full language name to the 2-letter code your retriever uses
        lang_map = {"malayalam": "ml", "hindi": "hi", "telugu": "te"}
        target_lang_code = lang_map.get(target_language.lower())
        
        # 2. Pass target_lang to strictly filter the FAISS index
        raw_retrieved_data = self.retriever.retrieve(
            text, 
            top_k=5,
            target_lang=target_lang_code # <--- This fixes the language mixing!
        )
        
        context_string = ""
        if raw_retrieved_data:
            # Change to target_idiom so it displays the local language phrase, not English
            meanings = [f"'{res['target_idiom']}'" for res in raw_retrieved_data]
            context_string = "Related known idioms/meanings: " + ", ".join(meanings)
        
        translation = self.generator.translate_idiom(text, context_string, target_language)
        
        return [{
            "english_idiom": text,
            "detected_confidence": detection_result.get("confidence"),
            "retrieved_context": context_string,
            "target_language": target_language.capitalize(),
            "translation": translation
        }]