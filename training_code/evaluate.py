import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from diskcache import Cache
from arc_loader import ArcDataset
from inference_tools import inference_run
from selection import EvalTool

# Paths
model_path = os.path.join('pretrained_models', "DDP-LLama-ReArc-4GPU-Full_trial_dataset-merged")
eval_challenges_path = os.path.join('input/arc-data/ARC-Data/eval/arc-prize-2024', 'arc-agi_evaluation_challenges_sample.json')
eval_solutions_path = os.path.join('input/arc-data/ARC-Data/eval/arc-prize-2024', 'arc-agi_evaluation_solutions_sample.json')

# Output paths
output_path = 'output_evaluation'
inference_cache = os.path.join(output_path, 'inference_cache')
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
    
    return model, tokenizer

def main():
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    arc_eval_set = ArcDataset.load_from_json(eval_challenges_path)
    arc_eval_set = arc_eval_set.load_solutions(eval_solutions_path)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Set formatting options
    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n',
        max_tokens=128000,
    )
    
    # Set up inference options
    infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
    infer_dataset = arc_eval_set.repeat(2).augment(**infer_aug_opts)
    
    # Set up caching
    model_cache = Cache(inference_cache).memoize(typed=True, ignore=set(['model_tok', 'guess']))
    
    # Set up evaluation tool
    eval_tool = EvalTool(n_guesses=2)
    
    # Run inference
    print("Running inference...")
    inference_results = inference_run(
        model_tok=(model, tokenizer),
        fmt_opts=fmt_opts,
        dataset=infer_dataset,
        min_prob=0.1,
        aug_score_opts=infer_aug_opts,
        callback=eval_tool.process_result,
        cache=model_cache,
    )
    
    # Write submission
    print("Writing submission file...")
    with open(submission_file, 'w') as f:
        json.dump(arc_eval_set.get_submission(inference_results), f)
    
    # Validate submission
    with open(submission_file, 'r') as f:
        score = arc_eval_set.validate_submission(json.load(f))
        print(f"Score for '{submission_file}':", score)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 