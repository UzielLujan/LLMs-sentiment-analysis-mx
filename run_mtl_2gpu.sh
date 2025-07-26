#!/bin/bash

# --- SLURM Configuration for Lab-SB (Multi-GPU with DDP) ---
#SBATCH --job-name=sentiment-analysis-mx
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/LLMs-sentiment-analysis-mx
#SBATCH --output=logs/%x-%j.log

# --- Best Practices ---
set -e

# --- Command-line Arguments ---
MODEL_PATH=${1:-"models/MarIA_local"}
RUN_NAME=${2:-"MarIA_2gpu_DDP"}
EPOCHS=${3:-3}
BATCH_SIZE=${4:-32} # Con DDP, 32 por GPU es un buen punto de partida
MAX_LENGTH=${5:-256}

# --- Create logs directory just in case ---
mkdir -p logs

# --- Environment and Experiment Setup ---
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "---"
echo "Model Path: $MODEL_PATH"
echo "Run Name: $RUN_NAME"
echo "========================================================"

# --- Conda Environment Activation & Job Execution ---
export PATH="/opt/anaconda_python311/bin:$PATH"
echo "Starting Python training script using 'torchrun' for Distributed Data Parallel (DDP)..."

# --- THE MAGIC LINE ---
# Use torchrun to launch 2 processes, one for each GPU in the node.
# The Trainer will detect this and automatically activate the DDP backend.
conda run -n llms-mx-env torchrun --nproc_per_node=2 src/train_mtl.py \
    --model_name "$MODEL_PATH" \
    --run_name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH"

echo "========================================================"
echo "Training script finished."
echo "Job finished."
echo "========================================================"



