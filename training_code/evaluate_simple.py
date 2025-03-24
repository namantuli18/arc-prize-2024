import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from arc_loader import ArcDataset

# Paths
model_path = os.path.join('pretrained_models', "DDP-LLama-ReArc-4GPU-Full_trial_dataset-merged")
eval_challenges_path = os.path.join('input/arc-data/ARC-Data/eval/arc-prize-2024', 'arc-agi_evaluation_challenges_sample.json')
eval_solutions_path = os.path.join('input/arc-data/ARC-Data/eval/arc-prize-2024', 'arc-agi_evaluation_solutions_sample.json')

# Output paths
output_path = 'output_evaluation_simple'
submission_file = os.path.join(output_path, 'submission.json')

def load_model_and_tokenizer(model_path):
    """Load the trained model and tokenizer."""
    print(f"Loading model and tokenizer from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def format_prompt(train_examples, test_example):
    """Format the prompt in a way compatible with the training format."""
    prompt = ""
    
    # Use the standard format as described in the example scripts
    alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz'
    
    # Add training examples
    for i, example in enumerate(train_examples):
        if i < len(alphabet):
            prompt += f"{alphabet[i]} {example['input']}\n{alphabet[i]} {example['output']}\n"
    
    # Add test example
    prompt += f"I {test_example['input']}\n+/-=O"
    
    return prompt

def generate_solution(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a solution using the standard HuggingFace generation API."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part (after +/-=O)
    if "+/-=O" in generated_text:
        response = generated_text.split("+/-=O", 1)[1].strip()
    else:
        response = generated_text[len(prompt):].strip()
    
    return response

def process_dataset(model, tokenizer, eval_set):
    """Process all examples in the evaluation dataset."""
    results = {}
    
    for task_idx, task in enumerate(tqdm(eval_set.tasks, desc="Processing tasks")):
        task_id = task.id
        
        # Format prompt with training examples and test example
        prompt = format_prompt(
            train_examples=[{"input": ex.input, "output": ex.output} for ex in task.train],
            test_example={"input": task.test.input}
        )
        
        # Generate solution
        response = generate_solution(model, tokenizer, prompt)
        
        # Store result
        results[task_id] = response
        
        # Log progress occasionally
        if task_idx % 10 == 0:
            print(f"Task {task_idx+1}/{len(eval_set.tasks)}: Generated response for {task_id}")
    
    return results

def main():
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    arc_eval_set = ArcDataset.load_from_json(eval_challenges_path)
    arc_eval_set = arc_eval_set.load_solutions(eval_solutions_path)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Process dataset
    print("Processing evaluation dataset...")
    results = process_dataset(model, tokenizer, arc_eval_set)
    
    # Save results
    print("Saving results...")
    with open(submission_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Validate submission if possible
    try:
        with open(submission_file, 'r') as f:
            submission = json.load(f)
            score = arc_eval_set.validate_submission(submission)
            print(f"Score: {score}")
    except Exception as e:
        print(f"Could not validate submission: {e}")
    
    print(f"Evaluation complete! Results saved to {submission_file}")

if __name__ == "__main__":
    main() 