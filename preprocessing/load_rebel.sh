#!/bin/bash
# Train a text-guided Graph Language Model (REBEL dataset)

# Job name
#SBATCH --job-name=rebel

#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --cpus-per-task=2            # Number of CPU cores per task
#SBATCH --mem=100G                    # Total memory requested
#SBATCH --partition=students

# Output and error logs
#SBATCH --output="rebel_out.txt"
#SBATCH --error="rebel_err.txt"

# Email notifications
#SBATCH --mail-user="ivakhnenko@cl.uni-heidelberg.de"
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

source ~/.bashrc 2>/dev/null

python3 preprocessing/load_rebel.py