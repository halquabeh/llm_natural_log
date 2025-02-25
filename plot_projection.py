import torch
import os
import numpy as np

# Define models to test
MODELS = [
    "EleutherAI/pythia-2.8b", 
    "openai-community/gpt2-large",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
]

# Define settings
SETTINGS = [
            # "numerics_PCA", 
            # "numerics_PLS", 
            "symbols_PCA", 
            # "symbols_PLS"
            ]

# Number of runs
RUNS = 1

# Directory containing the .pth files
DATA_DIR = "checkpoints"  # Update this with the correct path

# Dictionary to store loaded models
loaded_models = {}

for model in MODELS:
    model_name = model.replace("/", "_")  # Extract model identifier
    loaded_models[model_name] = {}

    for setting in SETTINGS:
        loaded_models[model_name][setting] = []

        for run in range(1, RUNS + 1):
            filename = f"{model_name}_{setting}_1_4_40"
            if run > 1:
                filename += f"_R{run}"
            filename += ".pth"
            file_path = os.path.join(DATA_DIR, filename)

            if os.path.exists(file_path):
                loaded_models[model_name][setting].append(torch.load(file_path, weights_only=False))
            else:
                print(f"Warning: {file_path} not found.")

# Dictionary to store processed results
processed_models = {}

for model_name, settings in loaded_models.items():
    processed_models[model_name] = {}

    for setting, runs in settings.items():
        processed_models[model_name][setting] = {}

        for layer in runs[0]:  # Assume all runs have the same structure
            processed_models[model_name][setting][layer] = {}

            # Preserve "answers"
            processed_models[model_name][setting][layer]["answers"] = runs[0][layer]["answers"]

            # Process hidden states
            processed_models[model_name][setting][layer]["hidden_states"] = {}
            for group in runs[0][layer]["hidden_states"].keys():
                all_runs = np.array([run[layer]["hidden_states"][group] for run in runs])
                processed_models[model_name][setting][layer]["hidden_states"][group] = {
                    "mean": np.mean(all_runs, axis=0),
                    "std": np.std(all_runs, axis=0)
                }

            # Process metrics
            processed_models[model_name][setting][layer]["metrics"] = {}
            for metric in ["Explained_variance", "monotonicity_metric", "sublinearity_metric"]:
                metric_values = np.abs(np.array([run[layer][metric] for run in runs]))
                processed_models[model_name][setting][layer]["metrics"][metric] = {
                    "mean": np.mean(metric_values),
                    "std": np.std(metric_values)
                }

print("Processed models successfully!")

import numpy as np
import matplotlib.pyplot as plt

# Set up figure and axes for 2x2 grid
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), constrained_layout=True)
axs = axs.flatten()  # Flatten for easier indexing

# Iterate over models and plot the highest explained variance layer
for idx, (model_name, settings) in enumerate(processed_models.items()):
    for setting, layers in settings.items():
        max_var_layer = None
        max_explained_variance = -float("inf")

        # Find the layer with max explained variance, skipping layer 0
        for layer, data in layers.items():
            if int(layer) == 0:  # Ensure layer 0 is skipped
                continue
            explained_var_mean = data["metrics"]["Explained_variance"]["mean"]
            if explained_var_mean > max_explained_variance:
                max_explained_variance = explained_var_mean
                max_var_layer = layer

        # Print the metrics of the selected layer
        if max_var_layer:
            print(f"Model: {model_name}, Setting: {setting}")
            print(f"Layer with max explained variance: {max_var_layer}")

            # Prepare data for plotting
            all_answers = []
            projections = []
            new_array = []

            data = layers[max_var_layer]
            for group_id, group_data in data['answers'].items():
                all_answers.extend(group_data)
                projections.extend(data['hidden_states'][group_id]['mean'])  # Use mean hidden state
                new_array.extend([int(group_id)] * len(group_data))

            all_answers = np.array(all_answers, dtype=float)
            projections = np.array(projections).squeeze()
            print(all_answers.shape,projections.shape)
            # Plot for the current model
            sc = axs[idx].scatter(
                np.log10(all_answers), projections, c=new_array, cmap='viridis', alpha=0.7, s=100  # Increased marker size
            )
            
            model_name_short = model_name.replace("openai-community_gpt2-large", "GPT2-L").replace("meta-llama_Llama-2-7b-hf", "Llama-2.7B").replace("EleutherAI_pythia-2.8b", "Pythia-2.8B").replace("mistralai_Mistral-7B-v0.1", "Mistral-7B")
            axs[idx].set_title(f'{model_name_short}\n $\\rho$={data["metrics"]["monotonicity_metric"]["mean"]:.2f}, $\\beta$={data["metrics"]["sublinearity_metric"]["mean"]:.2f}', fontsize=26)
            axs[idx].set_xlabel('$\log_{10}$(x)', fontsize=26)
            axs[idx].set_ylabel('T(x)', fontsize=26)
            axs[idx].tick_params(axis='both', labelsize=20)
            axs[idx].grid(True, linestyle="--", alpha=0.5)
plt.savefig('plot_PCAs.pdf', format='pdf')