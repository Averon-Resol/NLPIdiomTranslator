import os
from dotenv import load_dotenv, find_dotenv

# 1. Actively hunt down the .env file and load it
env_path = find_dotenv()
load_dotenv(env_path)

if not os.environ.get("GROQ_API_KEY"):
    print("--- CRITICAL ERROR: .env file found, but GROQ_API_KEY is missing! ---")

from flask import Flask, request, jsonify, render_template_string
from src.pipeline import IdiomTranslationPipeline

app = Flask(__name__)

# Initialize your actual ML pipeline
print("Booting up the Translation Pipeline...")
pipeline = IdiomTranslationPipeline()
print("Pipeline is ready!")

# =====================================================================
# Frontend Template (Single-File Architecture)
# =====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Idiom Translator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        /* Custom scrollbar and animations */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1f2937; }
        ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #6b7280; }
        .fade-in { animation: fadeIn 0.3s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen font-sans selection:bg-indigo-500 selection:text-white">

    <div class="max-w-4xl mx-auto px-4 py-12">
        <header class="text-center mb-12">
            <div class="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-2xl mb-4 text-indigo-400">
                <i data-lucide="languages" class="w-8 h-8"></i>
            </div>
            <h1 class="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400 mb-4">
                Idiom Translator
            </h1>
            <p class="text-gray-400 text-lg">Cross-lingual semantic RAG pipeline powered by Llama-3.3-70B</p>
        </header>

        <div class="bg-gray-800 rounded-3xl shadow-xl border border-gray-700 overflow-hidden">
            <div class="p-6 md:p-8">
                
                <div class="space-y-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-2">Source Idiom</label>
                        <textarea id="idiomInput" rows="3" 
                            class="w-full bg-gray-900 border border-gray-700 rounded-xl p-4 text-gray-100 placeholder-gray-500 focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all resize-none"
                            placeholder="e.g., Barking dogs seldom bite..."></textarea>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-300 mb-2">Target Language</label>
                            <div class="relative">
                                <select id="targetLang" class="w-full bg-gray-900 border border-gray-700 rounded-xl p-3 text-gray-100 appearance-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent cursor-pointer">
                                    <option value="hi">Hindi (हिंदी)</option>
                                    <option value="te">Telugu (తెలుగు)</option>
                                    <option value="ml">Malayalam (മലയാളം)</option>
                                </select>
                                <div class="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none text-gray-400">
                                    <i data-lucide="chevron-down" class="w-5 h-5"></i>
                                </div>
                            </div>
                        </div>
                        
                        <div class="flex items-end">
                            <button id="translateBtn" onclick="translateIdiom()" 
                                class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-xl p-3 transition-colors flex items-center justify-center gap-2 group">
                                <span>Translate Idiom</span>
                                <i data-lucide="sparkles" class="w-4 h-4 group-hover:animate-pulse"></i>
                            </button>
                        </div>
                    </div>
                </div>

                <div id="loadingState" class="hidden py-12 flex flex-col items-center justify-center space-y-4">
                    <div class="w-10 h-10 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin"></div>
                    <p class="text-indigo-400 text-sm animate-pulse">Running semantic retrieval & inference...</p>
                </div>

                <div id="outputSection" class="hidden mt-8 pt-8 border-t border-gray-700 fade-in">
                    <div class="space-y-6">
                        <div class="bg-gray-900/50 rounded-xl p-4 border border-gray-700/50">
                            <div class="flex items-center gap-2 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                                <i data-lucide="database" class="w-4 h-4"></i>
                                FAISS Retriever Context
                            </div>
                            <p id="retrievedContext" class="text-gray-300 italic"></p>
                        </div>

                        <div>
                            <div class="flex items-center gap-2 text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-3">
                                <i data-lucide="bot" class="w-4 h-4"></i>
                                Generated Translation
                            </div>
                            <div class="bg-gray-900 rounded-xl p-6 border border-indigo-500/20 relative group">
                                <p id="generatedText" class="text-xl text-white font-medium leading-relaxed"></p>
                                <button onclick="copyToClipboard(event)" class="absolute top-4 right-4 text-gray-500 hover:text-indigo-400 opacity-0 group-hover:opacity-100 transition-opacity" title="Copy to clipboard">
                                    <i data-lucide="copy" class="w-5 h-5"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <script>
        // Initialize Lucide icons
        lucide.createIcons();

        async function translateIdiom() {
            const input = document.getElementById('idiomInput').value.trim();
            const targetLang = document.getElementById('targetLang').value;
            
            if (!input) return;

            // UI State updates
            document.getElementById('translateBtn').disabled = true;
            document.getElementById('translateBtn').classList.add('opacity-50');
            document.getElementById('outputSection').classList.add('hidden');
            document.getElementById('loadingState').classList.remove('hidden');
            document.getElementById('loadingState').classList.add('flex');

            try {
                const response = await fetch('/api/translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: input, target_lang: targetLang })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "Something went wrong.");
                }

                // Populate data
                document.getElementById('retrievedContext').textContent = data.retrieved_context;
                document.getElementById('generatedText').textContent = data.generated_translation;

                // Show results
                document.getElementById('loadingState').classList.add('hidden');
                document.getElementById('loadingState').classList.remove('flex');
                document.getElementById('outputSection').classList.remove('hidden');
                
            } catch (error) {
                alert('Error connecting to the translation server: ' + error.message);
                document.getElementById('loadingState').classList.add('hidden');
                document.getElementById('loadingState').classList.remove('flex');
            } finally {
                document.getElementById('translateBtn').disabled = false;
                document.getElementById('translateBtn').classList.remove('opacity-50');
            }
        }

        function copyToClipboard(event) {
            const text = document.getElementById('generatedText').textContent;
            navigator.clipboard.writeText(text).then(() => {
                const btn = event.currentTarget;
                btn.classList.add('text-green-400');
                setTimeout(() => btn.classList.remove('text-green-400'), 1000);
            });
        }
    </script>
</body>
</html>
"""

# =====================================================================
# API Routes
# =====================================================================

@app.route('/')
def home():
    """Serves the single-page application UI."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/translate', methods=['POST'])
def translate():
    """Handles the translation inference request using the real pipeline."""
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    
    # Get the 2-letter code from the frontend and map it to the full name
    target_lang_code = data.get('target_lang', 'ml')
    lang_map = {
        "hi": "hindi",
        "te": "telugu",
        "ml": "malayalam"
    }
    target_lang_full = lang_map.get(target_lang_code, "malayalam")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Run the actual ML pipeline
        translation_results = pipeline.process_text(text, target_language=target_lang_full)
        
        # If the detector flagged it as literal, the pipeline returns an empty list
        if not translation_results:
            return jsonify({
                "retrieved_context": "No idioms detected by XLM-RoBERTa.",
                "generated_translation": "Literal text. Translation bypassed."
            })
            
        # Extract the data from the first matched idiom
        result = translation_results[0]
        
        return jsonify({
            "retrieved_context": result.get("retrieved_context", "No context found in FAISS."),
            "generated_translation": result.get("translation", "Error generating translation.")
        })
        
    except Exception as e:
        print(f"Pipeline Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Idiom Translator Web Interface...")
    print("👉 Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, host='0.0.0.0', port=5000)