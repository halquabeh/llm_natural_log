#!/bin/bash

# Define configurations
MODELS=(
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "EleutherAI/pythia-2.8b" 
    # "openai-community/gpt2-large"
    "meta-llama/Llama-2-7b-hf"
    # "mistralai/Mistral-7B-v0.1"
    # "meta-llama/Llama-3.1-8B"
)
TRANSFORMS=("PCA")
K_VALUES=(40)
NUM_EXAMPLES=(4)
DATA_TYPES=("numerics")

# Available devices
DEVICES=(2 3)

# Clean and create logs directory
# rm -rf logs/*
# mkdir -p logs

# Create command file
cmd_file="job_commands.txt"
rm -f $cmd_file

# Generate all commands
for model in "${MODELS[@]}"; do
    for transform in "${TRANSFORMS[@]}"; do
        for k in "${K_VALUES[@]}"; do
            for num_ex in "${NUM_EXAMPLES[@]}"; do
                for data in "${DATA_TYPES[@]}"; do
                    # Create output filename based on parameters
                    output_file="logs/${model//\//_}_${transform}_k${k}_ex${num_ex}_${data}_R4.txt"
                    
                    # Append command to file
                    echo "python3 main.py \
                        --model_name \"$model\" \
                        --transform $transform \
                        --k $k \
                        --num_examples $num_ex \
                        --data $data \
                        --device 3 \
                        > $output_file 2>&1" >> $cmd_file
                done
            done
        done
    done
done

# Check if GNU Parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel is not installed. Installing now..."
    sudo apt-get update && sudo apt-get install -y parallel
fi

# Run all commands in parallel using GNU Parallel
nohup parallel --joblog parallel_log.txt --results parallel_results -j ${#DEVICES[@]} < $cmd_file

echo "All jobs have been executed in parallel across ${#DEVICES[@]} devices."
