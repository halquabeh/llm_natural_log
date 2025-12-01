import numbers
import torch
from huggingface_hub import login
import os
import pickle
from utils.compute_utils import transform_hidden_states, analyze_transformed_hidden_states
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import random
login(token="")
# fix the seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class analyzer:
    def __init__(self,args):
        self.save_dir = "ICLR_results"
        self.data = args.data
        self.num_examples = args.num_examples
        self.k = args.k
        self.groups = args.groups
        self.context = args.context
        self.upper_bound = args.upper_bound
        self.Tdim = args.Tdim
        self.runs = args.runs
        self.log = []
        self.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name).to(self.device)
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = f"{args.model_name.replace('/', '_')}_{args.data}_k{args.k}_n{args.num_examples}_context_{args.context}_{args.transform.upper()}.pkl"
        self.save_path = os.path.join(self.save_dir, self.filename)

        #     self.save_to_file()

    def save_to_file(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.results, f)
        print(f"results saved to folder to {self.save_path}")
    def string_to_number(self,s):
        """Converts a string to a base-26 number, where 'a' = 0, 'b' = 1, ..., 'z' = 25."""
        num = 0
        for char in s:
            num = num * 26 + (ord(char) - ord('a')) #get_numerical_value(char, mapping)#
        return num

    def collect_states(self):
        # Load the prompts:
        # 1- Interval function return a range of 40 around 10*i given i.
        interval_function = lambda i: range(10**i - 2 * 10, 10**i + 2 * 10) if i >1 else range(1, 40)
        # 2- Load the prompts data based on the interval formula: it return list of pormpts in the format:
        # [w=w,y=y,z=z,x=] * k where k is the number of examples in all groups
        from utils.data_utils import generate_prompts_numerals,generate_prompts_symbols
        if self.data.lower() == 'numerics':
            prompts = generate_prompts_numerals(self.k, self.num_examples, self.upper_bound, self.groups,interval_function , context=self.context)
        else:
            prompts = generate_prompts_symbols(self.k, self.num_examples, self.groups , context=self.context)
        # Load the model and its tokenizer:
        model_num_layer = self.model.config.num_hidden_layers
        # if not found then run the for loop again
        # Initialize results: for each layer, store a dict with 'hidden_states' and 'answers'
        print("Starting new experiment")
        results = {}
        for i in range(model_num_layer):
            results[i] = {'hidden_states': {}, 'answers': {}}
        with torch.no_grad():
            for group, prompt_list in prompts.items():
                group_hidden_states = {i: [] for i in range(model_num_layer)}
                group_answers = []
                for prompt in prompt_list:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    last_number = prompt[prompt.rfind(',')+1:prompt.rfind('=')]
                    last_token_rep_all_layers = [outputs.hidden_states[lyr][:, -1, :].detach().cpu() for lyr in range(model_num_layer)]
                    for i in range(model_num_layer):
                        group_hidden_states[i].append(last_token_rep_all_layers[i])
                    if self.data.lower() != 'numerics':
                        last_number = self.string_to_number(last_number)
                    group_answers.append(last_number)
                for i in range(model_num_layer):
                    results[i]['hidden_states'][group] = group_hidden_states[i]
                    results[i]['answers'][group] = group_answers
        
        return results
    def analyze(self,transform='PCA'):
        # print("Computing transformed hidden states and analysis...")
        results_T = transform_hidden_states(self.states,transform, self.Tdim)
        results_analysis = analyze_transformed_hidden_states(results_T)
        return results_analysis
    def run_multiple(self):
        assert self.runs > 1, "args.runs must be greater than 1 to run multiple experiments."
        total_res_PCA = []
        total_res_PLS = []
        for _ in range(self.runs):
            self.states = self.collect_states()
            total_res_PCA.append(self.analyze(transform='PCA'))
            total_res_PLS.append(self.analyze(transform='PLS'))
        res_PCA = {}
        res_PLS = {}
        for layer in range(1, self.model.config.num_hidden_layers):
            # accumlate the PCA
            res_PCA[layer] = {}
            res_PCA[layer]['EV'] = np.array([total_res_PCA[i][layer]['Explained_variance'] for i in range(self.runs)]).mean()
            res_PCA[layer]['rho'] = abs(np.array([total_res_PCA[i][layer]['monotonicity_metric'] for i in range(self.runs)]).mean())
            res_PCA[layer]['beta'] = np.array([total_res_PCA[i][layer]['sublinearity_metric'] for i in range(self.runs)]).mean()
            res_PCA[layer]['EV_std'] = np.array([total_res_PCA[i][layer]['Explained_variance'] for i in range(self.runs)]).std()
            res_PCA[layer]['rho_std'] = np.array([total_res_PCA[i][layer]['monotonicity_metric'] for i in range(self.runs)]).std()
            res_PCA[layer]['beta_std'] = np.array([total_res_PCA[i][layer]['sublinearity_metric'] for i in range(self.runs)]).std()
            # accumlate the PLS
            res_PLS[layer] = {}
            res_PLS[layer]['EV'] = np.array([total_res_PLS[i][layer]['Explained_variance'] for i in range(self.runs)]).mean()
            res_PLS[layer]['rho'] = abs(np.array([total_res_PLS[i][layer]['monotonicity_metric'] for i in range(self.runs)]).mean())
            res_PLS[layer]['beta'] = np.array([total_res_PLS[i][layer]['sublinearity_metric'] for i in range(self.runs)]).mean()
            res_PLS[layer]['EV_std'] = np.array([total_res_PLS[i][layer]['Explained_variance'] for i in range(self.runs)]).std()
            res_PLS[layer]['rho_std'] = np.array([total_res_PLS[i][layer]['monotonicity_metric'] for i in range(self.runs)]).std()
            res_PLS[layer]['beta_std'] = np.array([total_res_PLS[i][layer]['sublinearity_metric'] for i in range(self.runs)]).std()
        return res_PCA,res_PLS
        
    def which_is_larger(self,num_dem):
        '''
        in this function we aim to see how the accuracy of answering 
        the question which is larger a or b, answer with a or b only.
        The accuracy should be dispoptioanlywith group number
        '''
        N = 100 # from each group make N pairs
        interval_function = lambda i: range(int(10**i - 2 * 10), int(10**i + 2 * 10)) if i >1 else range(1, 40)
        res = []
        for i in self.groups:
            pairs = [(random.choice(interval_function(i)),random.choice(interval_function(i))) for _ in range(N)]
            correct = 0
            for pair in pairs:
                # Provide model with in-context examples to improve understanding
                # Construct the prompt based on the value of domenstrations (dem)
                demonstrations = [
                    "Which is larger 10 or 7 ? 10, ",
                    "Which is larger 290 or 305 ? 305, ",
                    "Which is larger 1232 or 1124 ? 1232, "
                ]
                if num_dem == 0:#zero shot question
                    prompt = f"Which is larger {pair[0]} or {pair[1]} ? Answer with one number only>"
                else:
                    prompt = "".join(demonstrations[:num_dem]) + f"Which is larger {pair[0]} or {pair[1]} ?>"
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=10,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.model.config.eos_token_id
                )
                answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
                try:
                    # Split to get model's actual answer text after prompt
                    response_text = answer.split('>')[-1].strip()
                    # convert both numbers to strings to check containment
                    self.log.append({'pair': pair, 'answer': answer, 'response_text': response_text})
                    # check if correct answer appears in plain text, prioritize exact number match
                    correct_answer = str(max(pair[0], pair[1]))
                    if correct_answer in response_text:
                        correct += 1
                except:
                    pass
            print(f'Accuracy for group {i} is: {correct / N:.2f}')
            res.append(f'Accuracy for group {i} is: {correct / N:.2f}')
        return res
    def generate(self,prompt,max_new_tokens=10):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.model.config.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    def collect_state_real(self,data):
        model_num_layer = self.model.config.num_hidden_layers
        # if not found then run the for loop again
        # Initialize results: for each layer, store a dict with 'hidden_states' and 'answers'
        print("Starting new experiment")
        prompts,answers = data.prompts.tolist(),data.value.tolist()
        results = {}
        for i in range(model_num_layer):
            results[i] = {'hidden_states': {}, 'answers': {}}
        with torch.no_grad():
            for group, prompt_list in prompts.items():
                group_hidden_states = {i: [] for i in range(model_num_layer)}
                group_answers = []
                for prompt in prompt_list:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    last_number = prompt[prompt.rfind(',')+1:prompt.rfind('=')]
                    last_token_rep_all_layers = [outputs.hidden_states[lyr][:, -1, :].detach().cpu() for lyr in range(model_num_layer)]
                    for i in range(model_num_layer):
                        group_hidden_states[i].append(last_token_rep_all_layers[i])
                    if self.data.lower() != 'numerics':
                        last_number = self.string_to_number(last_number)
                    group_answers.append(last_number)
                for i in range(model_num_layer):
                    results[i]['hidden_states'][group] = group_hidden_states[i]
                    results[i]['answers'][group] = group_answers
        
        return results
