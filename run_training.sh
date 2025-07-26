#!/bin/bash

# --- SLURM Configuration for Lab-SB ---
#SBATCH --job-name=sentiment-analysis-mx
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.log  # This path is now correct because we run from the root

# --- Best Practices ---
set -e

# --- Command-line Arguments ---
MODEL_PATH=${1:-"models/BETO_local"}
RUN_NAME=${2:-"BETO_baseline_final"}
EPOCHS=${3:-3}
BATCH_SIZE=${4:-32}
MAX_LENGTH=${5:-256}

# --- Environment and Experiment Setup ---
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)" # Should be the project root
echo "---"
echo "Model Path: $MODEL_PATH"
echo "Run Name: $RUN_NAME"
echo "========================================================"

# --- Conda Environment Activation & Job Execution ---
# We use the robust 'conda run' method. Since we are in the project root,
# all paths passed to the python script are now correct.
export PATH="/opt/anaconda_python311/bin:$PATH"
echo "Starting Python training script using 'conda run'..."

conda run -n llms-mx-env python src/train.py \
    --model_name "$MODEL_PATH" \
    --run_name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH"

echo "========================================================"
echo "Training script finished."
echo "Job finished."
echo "========================================================"












