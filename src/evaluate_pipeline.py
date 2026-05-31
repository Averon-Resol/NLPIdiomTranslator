import os
from dotenv import load_dotenv, find_dotenv

# 1. Load the API key BEFORE importing the pipeline
load_dotenv(find_dotenv())

import time
import pandas as pd
import argparse
from src.pipeline import IdiomTranslationPipeline
import evaluate 

# ... (the rest of the script stays exactly the same)

def run_evaluation(target_lang, n_samples=50):
    print(f"--- Starting Pipeline Evaluation for {target_lang.upper()} ---")
    
    # 1. Map the full language name to the 2-letter code used in the CSV
    lang_map = {"hindi": "hi", "telugu": "te", "malayalam": "ml"}
    csv_lang_code = lang_map.get(target_lang.lower())
    
    if not csv_lang_code:
        print(f"Error: '{target_lang}' is not supported. Use hindi, telugu, or malayalam.")
        return

    # 2. Load your test data
    try:
        df = pd.read_csv("data/processed/generator_test.csv")
    except FileNotFoundError:
        print("Error: Could not find data/processed/generator_test.csv")
        return
        
    # 3. Filter using the 2-letter code (csv_lang_code) instead of target_lang
    df_lang = df[df['target_lang'] == csv_lang_code].head(n_samples)
    
    if df_lang.empty:
        print(f"No test data found for language code: {csv_lang_code}")
        return

    pipeline = IdiomTranslationPipeline()
    
    # ... (the rest of the script stays exactly the same!)
    
    predictions = []
    references = []
    exact_matches = 0
    
    print(f"Testing {len(df_lang)} idioms via Groq (Llama 3.3)...")
    
    # 2. Run inference
    for index, row in df_lang.iterrows():
        input_text = row['input_text']
        gold_translation = row['target_text']
        
        # Hit the API
        results = pipeline.process_text(input_text, target_language=target_lang)
        
        if not results:
            pred_text = input_text
        else:
            pred_text = results[0].get('translation', '')
            
        predictions.append(pred_text)
        references.append([gold_translation]) # evaluate expects a list of lists for references
        
        # Calculate Exact Match
        if pred_text.strip() == gold_translation.strip():
            exact_matches += 1
            
        # Sleep for 1 second to respect Groq free tier rate limits
        time.sleep(1)
        print(f"Processed {index + 1}/{len(df_lang)}")

    # 3. Calculate Metrics
    print("\nCalculating metrics (this may take a moment for BERTScore)...")
    bleu_metric = evaluate.load("sacrebleu")
    bertscore_metric = evaluate.load("bertscore")
    
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    
    # BERTScore requires specifying a model for multilingual evaluation
    bert_results = bertscore_metric.compute(
        predictions=predictions, 
        references=[ref[0] for ref in references], 
        lang=target_lang[:2] # e.g., 'hi', 'te', 'ml'
    )
    
    avg_bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
    exact_match_pct = (exact_matches / len(df_lang)) * 100
    
    # 4. Print the final report
    print("\n" + "="*40)
    print(f" Generator Results (Llama-3.3-70B)")
    print("="*40)
    print(f"Test pairs  | {len(df_lang)}")
    print(f"BLEU        | {bleu_results['score']:.2f}")
    print(f"BERTScore F1| {avg_bert_f1:.4f}")
    print(f"Exact match | {exact_match_pct:.1f}% ({exact_matches}/{len(df_lang)})")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_lang", type=str, required=True, help="hindi, telugu, or malayalam")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to test")
    args = parser.parse_args()
    
    run_evaluation(args.target_lang, args.n_samples)