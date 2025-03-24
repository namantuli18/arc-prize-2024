
import os
import json
from unsloth import FastLanguageModel
from diskcache import Cache

from arc_loader import ArcDataset
from model_tools import load_unsloth_4bit
from inference_tools import inference_run
from selection import EvalTool

# Paths
base_model = os.path.join('pretrained_models', "DDP-LLama-ReArc-4GPU-Full_trial_dataset-merged")
#eval_challenges_path = os.path.join('input/arc-data/ARC-Data/eval/arc-prize-2024', 'arc-agi_evaluation_challenges_sample.json')
#eval_solutions_path = os.path.join('input/arc-data/ARC-Data/eval/arc-prize-2024', 'arc-agi_evaluation_solutions_sample.json')
arc_data_path = os.path.join('input/arc-data/ARC-Data/eval', 'arc-prize-2024')

# Output paths
output_path = 'output_evaluation'
inference_cache = os.path.join(output_path, 'inference_cache')
submission_file = os.path.join(output_path, 'submission.json')
# load evaluation dataset
arc_eval_set = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi_evaluation_challenges.json'))
arc_eval_set = arc_eval_set.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))


# # input paths
# base_model = 'da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit'  # auto-downloaded from huggingface.co
# arc_data_path = os.path.join('input', 'arc-prize-2024')  # as on kaggle arc prize 2024

# output paths
# output_path = 'output_evaluation_Llama-rearc_without_ttt'
# inference_cache = os.path.join(output_path, 'inference_cache')
# submission_file = os.path.join(output_path, 'submission.json')

# load evaluation dataset
# arc_eval_set = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi_evaluation_challenges.json'))
# arc_eval_set = arc_eval_set.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))

# load model
model, tokenizer = load_unsloth_4bit(base_model)

# set formatting options
fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=128000,
)

# run inference
FastLanguageModel.for_inference(model)
infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
infer_dataset = arc_eval_set.repeat(2).augment(**infer_aug_opts)
model_cache = Cache(inference_cache).memoize(typed=True, ignore=set(['model_tok', 'guess']))
eval_tool = EvalTool(n_guesses=2)
inference_results = inference_run(
    model_tok=(model, tokenizer),
    fmt_opts=fmt_opts,
    dataset=infer_dataset,
    min_prob=0.1,
    aug_score_opts=infer_aug_opts,
    callback=eval_tool.process_result,
    cache=model_cache,
)

# write submission
with open(submission_file, 'w') as f:
    json.dump(arc_eval_set.get_submission(inference_results), f)
with open(submission_file, 'r') as f:
    print(f"Score for '{submission_file}':", arc_eval_set.validate_submission(json.load(f)))
