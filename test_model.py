import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "google/gemma-2-2b-it"
adapter_path = "/home/fatweeb/college/NLP/models/generator"

print("Loading base model (offline mode)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    device_map="auto",
    local_files_only=True
)

print("Applying your trained adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# --- THIS IS THE NEW PART ---
# Create a proper chat message
messages = [
    {"role": "user", "content": "Explain the idiom: 'Spill the beans'."}
]

# Let the tokenizer apply Gemma's specific chat format
formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print(f"\nFormatted Prompt:\n{formatted_prompt}\n")

inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=150,
        temperature=0.7,      # Adds a little bit of creativity
        do_sample=True        # Required when using temperature
    )

# Slice the output so it only prints the model's new generated text, not the prompt
input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]
print("\n--- MODEL OUTPUT ---")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))