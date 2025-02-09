#!/bin/bash

# List of models to run
declare -a models=(
    "EleutherAI/pythia-2.8b"
)

# Dataset name
dataset="numerics"

# Create checkpoints directory if not exists

# Loop over models and run them in the background using GPUs 1, 2, 3 in parallel
gpu_ids=(1 2 3)
num_gpus=${#gpu_ids[@]}

for i in "${!models[@]}"; do
    model="${models[i]}"
    gpu="${gpu_ids[i % num_gpus]}"
    log_file="checkpoints/${dataset}_$(echo $model | tr '/' '_').txt"
    echo "Running model: $model on GPU: $gpu"
    nohup python3 main.py --data "$dataset" --model_name "$model" --device "cuda:$gpu" > "$log_file" 2>&1 &
done

echo "All models are running in the background."
