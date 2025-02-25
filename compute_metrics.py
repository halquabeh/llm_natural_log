import torch
import os
import numpy as np

# Define models to test
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "EleutherAI/pythia-2.8b", 
    "openai-community/gpt2-large",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-3.1-8B"
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


# Find the layer with the highest explained variance
for model_name, settings in processed_models.items():
    for setting, layers in settings.items():
        max_var_layer = None
        max_explained_variance = -float("inf")

        # Find the layer with max explained variance
        for layer, data in layers.items():
            if layer == 0:continue
            explained_var_mean = data["metrics"]["Explained_variance"]["mean"]
            if explained_var_mean > max_explained_variance:
                max_explained_variance = explained_var_mean
                max_var_layer = layer

        # Print the metrics of the selected layer
        if max_var_layer:
            print(f"Model: {model_name}, Setting: {setting}")
            print(f"Layer with max explained variance: {max_var_layer}")
            print("Metrics:")
            for metric, values in layers[max_var_layer]["metrics"].items():
                print(f"  {metric}: Mean = {values['mean']:.4f}, Std = {values['std']:.4f}")
            print("=" * 50)
