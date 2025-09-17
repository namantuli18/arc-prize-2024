import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Define model information and repository
model_id = "akhil-dua/baseline-test" 
# Replace with your Hugging Face username and a new repository name
new_repo_id = "akhil-dua/baseline-test-4bit"

# 2. Configure 4-bit quantization
# This configuration defines how the model will be quantized
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading and quantizing model: {model_id}...")
# 3. Load the model with the quantization configuration
# The model is automatically quantized to 4-bit on the fly
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Model loaded and quantized successfully.")

# 4. Upload the quantized model to the Hugging Face Hub
# The push_to_hub() method handles the upload of the quantized weights
print(f"Uploading quantized model to the Hub as {new_repo_id}...")
model.push_to_hub(new_repo_id)
tokenizer.push_to_hub(new_repo_id)

print("Upload complete! ✅ The 4-bit quantized model is now on your profile.")