#!/bin/bash
# Train a text-guided Graph Language Model (REBEL dataset)
# NB: Doesn't work hand-run within an salloc job, only via sbatch

# Job name
#SBATCH --job-name=graph

#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --cpus-per-task=2            # Number of CPU cores per task
#SBATCH --mem=32G                    # Total memory requested
#SBATCH --partition=students

# Output and error logs
#SBATCH --output="graph_out.txt"
#SBATCH --error="graph_err.txt"

# Email notifications
#SBATCH --mail-user="ivakhnenko@cl.uni-heidelberg.de"
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

source ~/.bashrc 2>/dev/null

pyenv activate .venv || { echo "Error: Could not activate the .venv pyenv environment."; exit 1; }

cd baselines/GraphLanguageModels || { echo "Error: Could not change to GraphLanguageModels directory."; exit 1; }

MODELSIZE=t5-small  # options: t5-small, t5-base, t5-large

python3 -m experiments.encoder.text_guided_relation_prediction.train_LM \
    --graph_representation gGLM \
    --reset_params False \
    --modelsize $MODELSIZE \
    --use_text FullyConnected \
    --use_graph True \
    --params_to_train all \
    --seed 0 \
    --entailed_triplets_only False \
    --device cuda \
    --logging_level INFO \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --num_epochs 1 \
    --run_eval True \
    --max_seq_len 596 \
    --save_model_dir baselines/model_4
