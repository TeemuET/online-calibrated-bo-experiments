#!/bin/bash

#SBATCH --job-name=unibo_synth   # Job name
#SBATCH --output=slurm_logs/synth_%A_%a.out # Standard output and error log (%A: job ID, %a: task ID)
#SBATCH --error=slurm_logs/synth_%A_%a.err
#SBATCH --array=0-19           # Array of 2560 jobs (128 problems * 20 seeds)
#SBATCH --time=02:00:00          # Time limit per job (e.g., 1 hour). Adjust after a test run.
#SBATCH --mem=4G                 # Memory limit per job (e.g., 4 GB). Adjust as needed.

# --- Job Setup ---
echo "Starting job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID"

module load mamba

# Activate your virtual environment
source activate unibo-env

# --- Parameter Calculation ---
# With the flat JSON file, this simple logic is all we need.
NUM_SEEDS=20
#PROBLEM_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
PROBLEM_IDX=127
SEED=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

echo "Running with Problem Index: $PROBLEM_IDX, Seed: $SEED"

# --- Experiment Execution ---
# We no longer need to pass 'd'. The python script figures it out.
ARGS="problem_idx=${PROBLEM_IDX}|seed=${SEED}|data_name=benchmark|experiment=GP-EI-BENCHMARKS-notFixed"
ARGS+="|n_initial=10|n_evals=90|n_seeds_per_job=20|n_test=5000|n_pool=5000"
ARGS+="|surrogate=GP|acquisition=EI|bo=True|extensive_metrics=True|recalibrate=True|test=False"
ARGS+="|noisify=True|snr=100.0|fix_surrogate_logic=False"

# Run the experiment     
python3 main.py "$ARGS"

echo "Job finished."