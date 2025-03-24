import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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
            # Handle nested structure - example is an object with 'input' field
            input_str = json.dumps(example["input"])
            output_str = json.dumps(example["output"])
            prompt += f"{alphabet[i]} {input_str}\n{alphabet[i]} {output_str}\n"
    
    # Add test example - test_example is a list with one item
    test_item = test_example[0]
    input_str = json.dumps(test_item["input"])
    prompt += f"I {input_str}\n+/-=O"
    
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

def load_dataset(challenges_path, solutions_path=None):
    """Load the dataset directly from JSON files."""
    # Load challenges
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    
    # Load solutions if provided
    solutions = None
    if solutions_path:
        try:
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load solutions: {e}")
    
    return challenges, solutions

def process_dataset(model, tokenizer, challenges):
    """Process all examples in the evaluation dataset."""
    results = {}
    
    # Iterate through the challenges dictionary
    total_tasks = len(challenges)
    print(f"Processing {total_tasks} tasks")
    
    for idx, (task_id, task_data) in enumerate(tqdm(challenges.items(), desc="Processing tasks")):
        try:
            # Extract train and test examples
            train_examples = task_data["train"]
            test_example = task_data["test"]
            
            # Debug info
            print(f"\nProcessing task {task_id}")
            print(f"Train examples: {len(train_examples)}")
            print(f"Test example: {len(test_example)}")
            
            # Format prompt
            prompt = format_prompt(train_examples, test_example)
            
            # Generate solution
            response = generate_solution(model, tokenizer, prompt)
            
            # Store result
            results[task_id] = response
            
            # Log progress occasionally
            if idx % 5 == 0 or idx < 5:
                print(f"Task {idx+1}/{total_tasks}: Generated response for {task_id}")
                print(f"Prompt excerpt: {prompt[:200]}...")
                print(f"Response: {response[:100]}...")
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            print(f"Task data: {json.dumps(task_data, indent=2)[:500]}...")
            continue
    
    return results

def validate_submission(submission, solutions):
    """Simple validation function to check if predictions match solutions."""
    if not solutions:
        return "No solutions available for validation"
    
    correct = 0
    for task_id, solution in solutions.items():
        if task_id in submission and submission[task_id] == solution:
            correct += 1
    
    score = correct / len(solutions) if solutions else 0
    return f"Score: {correct}/{len(solutions)} = {score:.3f}"

def main():
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load evaluation dataset directly from JSON
    print("Loading evaluation dataset...")
    challenges, solutions = load_dataset(eval_challenges_path, eval_solutions_path)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Process dataset
    print("Processing evaluation dataset...")
    results = process_dataset(model, tokenizer, challenges)
    
    # Save results
    print("Saving results...")
    with open(submission_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Validate submission if possible
    if solutions:
        score = validate_submission(results, solutions)
        print(score)
    
    print(f"Evaluation complete! Results saved to {submission_file}")

if __name__ == "__main__":
    main() 