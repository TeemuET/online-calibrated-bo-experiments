#!/bin/bash

#SBATCH --job-name=unibo_experiments     # Job name
#SBATCH --output=slurm_logs/%x_%A_%a.out # Standard output and error log
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-999
#SBATCH --time=00:45:00                # Time limit per job. Adjust as needed.
#SBATCH --mem=1G                       # Memory limit per job. Adjust as needed.

# --- 1. USER CONFIGURATION: EDIT THIS SECTION FOR EACH BATCH ---

# a) Define the problems you want to run in this batch
PROBLEM_NAMES=("Forrester" "Ackley" "SixHumpCamel" "Sargan" "Ackley")
DIMS=(         1           2        2       6     10)

NUM_SEEDS_PER_PROBLEM=10

SURROGATE="GP"

N_INITIALS=(5 10)
ACQ_FUNCS=("UCB" "EI")
QUANTILE_LEVELS=(0.9 0.95)
NOISIFY_FLAGS=(True False)

METHOD_NAMES=("None" "UNIBOv1" "UNIBOv2" "ONLINEv1" "ONLINEv2" "ONLINEv1" "ONLINEv2")
METHOD_ETAS=( 0.0 0.0 0.0 0.05 0.05 0.5 0.5)
FOLDER_NAMES=("None"  "UNIBOv1" "UNIBOv2" "OnlineV1_e005" "OnlineV2_e005" "OnlineV1_e05" "OnlineV2_e05")

NUM_PROBLEMS=${#PROBLEM_NAMES[@]}
NUM_METHOD_CONFIGS=${#METHOD_NAMES[@]}
NUM_N_INITIALS=${#N_INITIALS[@]}
NUM_ACQS=${#ACQ_FUNCS[@]}
NUM_QUANTILES=${#QUANTILE_LEVELS[@]}
NUM_NOISIFY=${#NOISIFY_FLAGS[@]}

TOTAL_JOBS=$((NUM_PROBLEMS * NUM_SEEDS_PER_PROBLEM * NUM_METHOD_CONFIGS * NUM_N_INITIALS * NUM_ACQS * NUM_QUANTILES * NUM_NOISIFY))
echo "This script requires an --array directive of: #SBATCH --array=0-$((TOTAL_JOBS - 1))"

# --- Decode SLURM_ARRAY_TASK_ID to specific parameters ---
NOISIFY_IDX=$((SLURM_ARRAY_TASK_ID % NUM_NOISIFY)); TEMP_IDX=$((SLURM_ARRAY_TASK_ID / NUM_NOISIFY))
QUANTILE_IDX=$((TEMP_IDX % NUM_QUANTILES)); TEMP_IDX=$((TEMP_IDX / NUM_QUANTILES))
ACQ_IDX=$((TEMP_IDX % NUM_ACQS)); TEMP_IDX=$((TEMP_IDX / NUM_ACQS))
N_INIT_IDX=$((TEMP_IDX % NUM_N_INITIALS)); TEMP_IDX=$((TEMP_IDX / NUM_N_INITIALS))
METHOD_IDX=$((TEMP_IDX % NUM_METHOD_CONFIGS)); TEMP_IDX=$((TEMP_IDX / NUM_METHOD_CONFIGS))
SEED=$((TEMP_IDX % NUM_SEEDS_PER_PROBLEM)); PROBLEM_ARRAY_IDX=$((TEMP_IDX / NUM_SEEDS_PER_PROBLEM))

PROBLEM_NAME=${PROBLEM_NAMES[$PROBLEM_ARRAY_IDX]}; PROBLEM_DIM=${DIMS[$PROBLEM_ARRAY_IDX]}
N_INITIAL=${N_INITIALS[$N_INIT_IDX]}; ACQUISITION=${ACQ_FUNCS[$ACQ_IDX]}
QUANTILE_LEVEL=${QUANTILE_LEVELS[$QUANTILE_IDX]}; NOISIFY=${NOISIFY_FLAGS[$NOISIFY_IDX]}

METHOD_NAME=${METHOD_NAMES[$METHOD_IDX]}
METHOD_ETA=${METHOD_ETAS[$METHOD_IDX]}
FOLDER_NAME_SUFFIX=${FOLDER_NAMES[$METHOD_IDX]}

if (( PROBLEM_DIM > 5 )); then
    N_EVALS=100
else
    N_EVALS=50
fi

if [ "$METHOD_NAME" == "None" ]; then
    RECAL_ARGS="recalibrate=False"
else
    RECAL_ARGS="recalibrate=True|recalibrator_type=${METHOD_NAME}"
fi
# Add eta to arguments ONLY if it's an online method
if (( $(echo "$METHOD_ETA > 0" | bc -l) )); then
    RECAL_ARGS+="|eta=${METHOD_ETA}"
fi

EXPERIMENT_NAME="${SURROGATE}-${ACQUISITION}_q${QUANTILE_LEVEL}-${N_INITIAL}init-recal_${FOLDER_NAME_SUFFIX}-noisify_${NOISIFY}"

ARGS="problem=${PROBLEM_NAME}|d=${PROBLEM_DIM}|seed=${SEED}|experiment=${EXPERIMENT_NAME}"
ARGS+="|surrogate=${SURROGATE}|acquisition=${ACQUISITION}|n_initial=${N_INITIAL}|quantile_level=${QUANTILE_LEVEL}"
ARGS+="|n_evals=${N_EVALS}|n_seeds_per_job=1|noisify=${NOISIFY}|snr=100.0"
ARGS+="|data_name=benchmark|bo=True|test=False|${RECAL_ARGS}"

echo "--- Running Experiment ---"
echo "Task ID: $SLURM_ARRAY_TASK_ID -> Parsed to:"
echo "Experiment Group: ${EXPERIMENT_NAME}"
echo "Problem: ${PROBLEM_NAME} (dim ${PROBLEM_DIM})"
echo "Method: ${METHOD_NAME} (Eta: ${METHOD_ETA})"
echo "Seed: ${SEED}"; echo "Noisify: ${NOISIFY}"
echo "--------------------------"

exit 0

# --- Execute the Python Script ---
module load mamba
source activate unibo-env
python3 main.py "$ARGS"

echo "Job finished successfully."