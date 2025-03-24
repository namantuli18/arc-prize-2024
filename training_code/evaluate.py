import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Paths
model_path = os.path.join('pretrained_models', "DDP-LLama-ReArc-4GPU-Full_trial_dataset-merged")
eval_challenges_path = os.path.join('input/arc-data/ARC-Data/eval/arc-prize-2024', 'arc-agi_evaluation_challenges_sample.json')
eval_solutions_path = os.path.join('input/arc-data/ARC-Data/eval/arc-prize-2024', 'arc-agi_evaluation_solutions_sample.json')

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
    
    return model, tokenizer

def format_prompt(train_examples, test_example):
    """Format the prompt in the same way as training."""
    prompt = ""
    
    # Add training examples
    for i, example in enumerate(train_examples):
        prompt += f"{chr(65+i)} {example['input']}\n{chr(65+i)} {example['output']}\n"
    
    # Add test example
    prompt += f"I {test_example['input']}\n+/-=O"
    
    return prompt

def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.7, top_p=0.9):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part (after +/-=O)
    response = response.split("+/-=O")[-1].strip()
    
    return response

def evaluate_model(model, tokenizer, challenges_path, solutions_path):
    """Evaluate the model on the evaluation dataset."""
    # Load evaluation data
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)
    
    results = []
    
    # Process each challenge
    for challenge in tqdm(challenges, desc="Evaluating challenges"):
        task_id = challenge['task_id']
        train_examples = challenge['train_examples']
        test_example = challenge['test_example']
        
        # Format prompt
        prompt = format_prompt(train_examples, test_example)
        
        # Generate response
        response = generate_response(model, tokenizer, prompt)
        
        # Store result
        results.append({
            'task_id': task_id,
            'predicted': response,
            'actual': solutions[task_id] if task_id in solutions else None
        })
    
    return results

def save_results(results, output_path):
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Run evaluation
    results = evaluate_model(model, tokenizer, eval_challenges_path, eval_solutions_path)
    
    # Save results
    output_path = os.path.join('results', 'evaluation_results.json')
    save_results(results, output_path)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total challenges evaluated: {len(results)}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 