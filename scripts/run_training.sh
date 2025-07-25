#!/bin/bash

# --- SLURM Configuration ---
#SBATCH --job-name=sentiment-analysis-mx   # Nombre del trabajo
#SBATCH --partition=GPU                    # Partición (cola) a usar. Para fine-tuning necesitamos GPU
#SBATCH --nodes=1                          # Pedimos un único nodo
#SBATCH --ntasks=1                         # Una única tarea en ese nodo
#SBATCH --cpus-per-task=8                  # CPUs para la tarea (útil para data loading)
#SBATCH --gres=gpu:1                       # Pedimos 1 GPU
#SBATCH --mem=0                            # Usar toda la memoria RAM del nodo
#SBATCH --time=08:00:00                    # Tiempo máximo de ejecución (HH:MM:SS)
#SBATCH --output=../logs/%x-%j.log         # Archivo de log para stdout y stderr

# --- Best Practices ---
# Exit script immediately if any command fails
set -e

# --- Command-line Arguments ---
# Recibimos los parámetros desde la línea de comandos para máxima flexibilidad
MODEL_NAME=${1:-"dccuchile/bert-base-spanish-wwm-cased"}
RUN_NAME=${2:-"BETO_baseline_cluster"}
EPOCHS=${3:-3}
BATCH_SIZE=${4:-32}
MAX_LENGTH=${5:-256}

# --- Environment and Experiment Setup ---
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing on partition: $SLURM_JOB_PARTITION"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "---"
echo "Model Name: $MODEL_NAME"
echo "Run Name: $RUN_NAME"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Length: $MAX_LENGTH"
echo "========================================================"

# --- Conda Environment Activation ---
# En muchos clústeres, es necesario cargar un módulo antes de usar conda
echo "Loading Conda module..."
module load anaconda3

echo "Activating Conda environment..."
source activate llms-mx-env

# --- Job Execution ---
echo "Starting Python training script..."
python ../src/train.py \
    --model_name "$MODEL_NAME" \
    --run_name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH"

echo "========================================================"
echo "Training script finished."
echo "Job finished."
echo "========================================================"
