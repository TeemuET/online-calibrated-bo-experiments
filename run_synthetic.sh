#!/bin/bash

#SBATCH --job-name=unibo_synth   # Job name
#SBATCH --output=slurm_logs/synth_%A_%a.out # Standard output and error log (%A: job ID, %a: task ID)
#SBATCH --error=slurm_logs/synth_%A_%a.err
#SBATCH --array=0-0           # Array of 2560 jobs (128 problems * 20 seeds)
#SBATCH --time=01:00:00          # Time limit per job (e.g., 1 hour). Adjust after a test run.
#SBATCH --mem=4G                 # Memory limit per job (e.g., 4 GB). Adjust as needed.

# --- Job Setup ---
echo "Starting job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID"

# Activate your virtual environment
source venv/bin/activate

# --- Parameter Calculation ---
# With the flat JSON file, this simple logic is all we need.
NUM_SEEDS=1
PROBLEM_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

echo "Running with Problem Index: $PROBLEM_IDX, Seed: $SEED"

# --- Experiment Execution ---
# We no longer need to pass 'd'. The python script figures it out.
ARGS="problem_idx=${PROBLEM_IDX}|seed=${SEED}|data_name=benchmark|n_evals=100|surrogate=GP|acquisition=EI|bo=True|extensive_metrics=True"

# Run the experiment
python3 main.py "$ARGS"

echo "Job finished."