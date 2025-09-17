# from config.config import Config
import torch
from models.load_model import Model
from huggingface_hub import login
login(token="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")
# fix the seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "EleutherAI/pythia-2.8b"
# model_name = "openai-community/gpt2-large"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "meta-llama/Llama-3.1-8B"


import argparse

parser = argparse.ArgumentParser(description='Natural Log Model Parameters')
parser.add_argument('--transform', type=str, default="PCA", help='Transformation method')
parser.add_argument('--Tdim', type=int, default=1, help='Transform dimension')
parser.add_argument('--k', type=int, default=30, help='Number of samples')
parser.add_argument('--num_examples', type=int, default=3, help='Number of examples')
parser.add_argument('--upper_bound', type=int, default=100, help='Upper bound for random numbers')
parser.add_argument('--context', type=str, default='same', help='Context type')
parser.add_argument('--data', type=str, default='numerics', help='Data type')
parser.add_argument('--groups', nargs='+', type=int, default=[1,2,3,4], help='Group sizes')
parser.add_argument('--save', action='store_true', default=True, help='Save results')
parser.add_argument('--plot', action='store_true', default=True, help='Plot results')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B", help='Model name')
parser.add_argument('--device', type=str, default="2", help='Device to run on')

args = parser.parse_args()

interval_function = lambda i: range(10**i - 2 * 10, 10**i + 2 * 10) if i >1 else range(1, 40) 

model_name = args.model_name
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
model = Model(model_name, device)

# Load the prompts data
from utils.data_utils import generate_prompts_numerals,generate_prompts_symbols
if args.data.lower() == 'numerics':
    prompts = generate_prompts_numerals(args.k, args.num_examples, args.upper_bound, args.groups,interval_function , context=args.context)
else:
    prompts = generate_prompts_symbols(args.k, args.num_examples, args.groups , context=args.context)

from utils.data_utils import strings_to_numbers 
# Initialize a dictionary to save all layers' hidden states raw imported from models layers
results = {} # the shape of this dictionary, will be hacing for every layer two dict hidden_states and answers. and inside each of these they woll be groups
# Iterate through each layer
for idx, layer in enumerate(list(range(0, model.model.config.num_hidden_layers, 1))):
    hidden_states = {}
    answers = {} # assuming the model answer the last number
    # Collect hidden states for each prompt in each group
    for key, prompt_list in prompts.items():
        hidden_states[key] = []
        answers[key] = []
        for prompt in prompt_list:
            hidden_state = model.get_hidden_state(prompt, layer_index=layer)
            hidden_states[key].append(hidden_state)
            last_number = prompt[prompt.rfind(',')+1:prompt.rfind('=')]
            # Assert that model can generate the correct answer for this prompt
            # generated = model.predict(prompt)
            # assert last_number in generated, f"Model failed to generate correct answer {last_number} in output {generated}"
            answers[key].append(last_number)
        if args.data.lower() == 'symbols':
            answers[key] = strings_to_numbers(answers[key])
    # Save the hidden states and answers for the current layer
    results[layer] = {'hidden_states': hidden_states, 'answers': answers}


# Apply T(x) using PLS or PCA
from utils.compute_utils import transform_hidden_states,analyze_transformed_hidden_states
results_T = transform_hidden_states(results,args.transform,args.Tdim)

# Coompute the metris for sublineraity and monotonicity:results_T
results_analysis = analyze_transformed_hidden_states(results_T)

from utils.visual_utils import plot_pca_projections
import os
# save_path = os.path.join('checkpoints', f'{model_name.replace("/", "_")}_{args.data}_{args.transfom}_{args.Tdim}_{args.num_examples}_{args.k}')
# plot_pca_projections(results_analysis, save_path)


save_path = os.path.join('checkpoints', f'{model_name.replace("/", "_")}_{args.data}_{args.transform}_{args.Tdim}_{args.num_examples}_{args.k}_R4.pth')
torch.save(results_analysis, save_path)
