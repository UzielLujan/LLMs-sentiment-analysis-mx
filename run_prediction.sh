#!/bin/bash

# --- SLURM Configuration for Lab-SB (Inference on a Single GPU) ---
#SBATCH --job-name=sentiment-inference-mx
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1 # La inferencia es menos intensiva, 1 GPU es suficiente
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00 # 1 hora debería ser más que suficiente
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/LLMs-sentiment-analysis-mx
#SBATCH --output=logs/%x-%j.log

# --- Best Practices ---
set -e

# --- Command-line Arguments ---
# El modelo campeón
MODEL_PATH=${1:-"models/BETO_MTL"} 
# El nombre que le daremos a la carpeta de resultados
RUN_NAME=${2:-"BETO_MTL_final_submission"} 

# --- Create logs and submissions directories ---
mkdir -p logs
mkdir -p submissions/$RUN_NAME

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
echo "Starting Python inference script..."

# --- THE MAGIC LINE ---
# Ejecutamos el script de inferencia con los parámetros adecuados
conda run -n llms-mx-env python src/inference.py \
    --model_path "$MODEL_PATH" \
    --test_file "data/Rest-Mex_2025_test.xlsx" \
    --output_dir "submissions/$RUN_NAME" \
    --max_length 256 \
    --batch_size 64 # Podemos usar un batch size mayor en inferencia

echo "========================================================"
echo "Inference script finished."
echo "Submission file generated in: submissions/$RUN_NAME/CorpusChristi_Run.txt"
echo "Job finished."
echo "========================================================"
