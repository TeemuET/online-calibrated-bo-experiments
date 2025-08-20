#!/bin/bash

#SBATCH --job-name=unibo_experiments     # Job name
#SBATCH --output=slurm_logs/%x_%A_%a.out # Standard output and error log
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-119
#SBATCH --time=04:00:00                # Time limit per job. Adjust as needed.
#SBATCH --mem=4G                       # Memory limit per job. Adjust as needed.

# --- 1. USER CONFIGURATION: EDIT THIS SECTION FOR EACH BATCH ---

# a) Define the problems you want to run in this batch
PROBLEM_NAMES=("Forrester" "Ackley" "SixHumpCamel" "Sargan" "Ackley")
DIMS=(         1           2        2       4     10)

NUM_SEEDS_PER_PROBLEM=10

SURROGATE="GP"
ACQUISITION="UCB"
N_INITIAL=10
QUANTILE_LEVEL=0.95
NOISIFY=False

METHODS=("None" "v1" "v2")

NUM_PROBLEMS=${#PROBLEM_NAMES[@]}
NUM_METHODS=${#METHODS[@]}
TOTAL_JOBS=$((NUM_PROBLEMS * NUM_SEEDS_PER_PROBLEM * NUM_METHODS))
echo "This script requires an --array directive of: #SBATCH --array=0-$((TOTAL_JOBS - 1))"

# --- Decode SLURM_ARRAY_TASK_ID to specific parameters ---
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))
TEMP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_METHODS))
SEED=$((TEMP_IDX % NUM_SEEDS_PER_PROBLEM))
PROBLEM_ARRAY_IDX=$((TEMP_IDX / NUM_SEEDS_PER_PROBLEM))

PROBLEM_NAME=${PROBLEM_NAMES[$PROBLEM_ARRAY_IDX]}
PROBLEM_DIM=${DIMS[$PROBLEM_ARRAY_IDX]}
METHOD=${METHODS[$METHOD_IDX]}

if (( PROBLEM_DIM > 5 )); then
    N_EVALS=100
else
    N_EVALS=50
fi

if [ "$METHOD" == "None" ]; then
    RECAL_ARGS="recalibrate=False"
else
    RECAL_ARGS="recalibrate=True|recalibrator_type=${METHOD}"
fi

EXPERIMENT_NAME="${SURROGATE}-${ACQUISITION}-${N_INITIAL}init-recal_${METHOD}-quantile_${QUANTILE_LEVEL}-noisify_${NOISIFY}"

ARGS="problem=${PROBLEM_NAME}|d=${PROBLEM_DIM}|seed=${SEED}|experiment=${EXPERIMENT_NAME}"
ARGS+="|acquisition=${ACQUISITION}|n_initial=${N_INITIAL}|quantile_level=${QUANTILE_LEVEL}"
ARGS+="|n_evals=${N_EVALS}|n_seeds_per_job=1"
ARGS+="|data_name=benchmark|surrogate=GP|bo=True|test=False"
ARGS+="|noisify=${NOISIFY}|snr=100.0|${RECAL_ARGS}"

echo "--- Running Experiment ---"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Problem: ${PROBLEM_NAME} (dim ${PROBLEM_DIM})"
echo "Seed: ${SEED}"
echo "Method: ${METHOD}"
echo "Python ARGS: ${ARGS}"
echo "--------------------------"

exit 0

# --- Execute the Python Script ---
module load mamba
source activate unibo-env # Activate your conda environment
python3 main.py "$ARGS"

echo "Job finished successfully."