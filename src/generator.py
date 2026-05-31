import os
from groq import Groq

class GroqTranslator:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found! Please ensure your .env file contains GROQ_API_KEY.")
        
        self.client = Groq(api_key=self.api_key)
        self.model_id = "llama-3.3-70b-versatile"

    def translate_idiom(self, english_idiom, retrieved_context="", target_language="malayalam"):
        """
        Translates the English idiom into the specified target language using Groq.
        """
        # Ensure it's cleanly capitalized (e.g., 'hindi' -> 'Hindi')
        target_language = target_language.capitalize()

        system_instruction = (
            f"You are an expert bilingual linguist specializing in English and {target_language}. "
            f"Your task is to translate the figurative meaning of English idioms into natural {target_language}. "
            "1. Analyze the figurative meaning of the provided English idiom. "
            "2. Review the 'Related known idioms/meanings' provided in the context. "
            "3. Select the most culturally accurate equivalent from the context, or generate one if the context is insufficient. "
            f"Provide ONLY the final {target_language} translation in native script, nothing else. Do not add quotes or explanations."
        )

        prompt = f"English Idiom: '{english_idiom}'\n"
        if retrieved_context:
            prompt += f"Figurative Meaning/Context: {retrieved_context}\n"
        prompt += f"{target_language} Translation:"

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_id,
                temperature=0.2, 
            )
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            return f"Translation Error: Could not connect to API."