#!/bin/bash

# --- SLURM Configuration ---
#SBATCH --job-name=sentiment-analysis-mx   # Nombre del trabajo
#SBATCH --partition=GPU                    # Partición (cola) a usar. [cite_start]Para fine-tuning necesitamos GPU [cite: 802]
#SBATCH --nodes=1                          # Pedimos un único nodo
#SBATCH --ntasks=1                         # Una única tarea en ese nodo
#SBATCH --cpus-per-task=8                  # CPUs para la tarea (útil para data loading)
#SBATCH --gres=gpu:1                       # Pedimos 1 GPU
[cite_start]#SBATCH --mem=0                            # Usar toda la memoria RAM del nodo [cite: 851]
#SBATCH --time=08:00:00                    # Tiempo máximo de ejecución (HH:MM:SS). 8 horas es un buen comienzo.
#SBATCH --output=logs/%x-%j.log            # Archivo de log para stdout y stderr. %x=job-name, %j=job-id

# --- Environment and Experiment Setup ---
# Variables para configurar el experimento fácilmente
export MODEL_NAME="BSC-TeMU/roberta-base-bne"
export RUN_NAME="MarIA_base_fine_tuned"
export EPOCHS=3
export BATCH_SIZE=32 # Un batch size mayor es posible con las Titan RTX de 24GB
export LEARNING_RATE=2e-5
export USE_WANDB_FLAG="--use_wandb" # Comenta esta línea si no quieres usar W&B

# --- Job Execution ---
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# Cargar el módulo de conda (esto puede variar ligeramente en el clúster)
# A menudo es 'module load anaconda3' o similar. Consulta la documentación del clúster si falla.
echo "Activando entorno Conda..."
# Asumiendo que has configurado conda para inicializarse con bash
source ~/anaconda3/etc/profile.d/conda.sh # Ajusta esta ruta si es necesario
conda activate restmex_env

# Verificar que el entorno se activó
if [ $? -ne 0 ]; then
    echo "Error: No se pudo activar el entorno Conda 'restmex_env'."
    exit 1
fi
echo "Entorno activado. Versión de Python:"
python --version

# Ejecutar el script de entrenamiento
echo "Iniciando script de Python..."
python src/train.py \
    --model_name "$MODEL_NAME" \
    --run_name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    $USE_WANDB_FLAG

echo "========================================================"
echo "El script de entrenamiento ha finalizado."
echo "========================================================"