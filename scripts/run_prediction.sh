#!/bin/bash

# --- SLURM Configuration for Lab-SB (Inference on a Single GPU) ---
#SBATCH --job-name=sentiment-inference-mx
#SBATCH --partition=GPU # La forma correcta de pedir un nodo con GPU en este clúster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00 # 1 hora debería ser más que suficiente
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/LLMs-sentiment-analysis-mx
#SBATCH --output=logs/%x-%j.log

# --- Best Practices ---
set -e

# --- Command-line Arguments ---
MODEL_PATH=${1:?"Error: Debes especificar la ruta al modelo."} 
RUN_NAME=${2:?"Error: Debes especificar un nombre para la ejecución."}
TOKENIZER_PATH=${3:?"Error: Debes especificar la ruta al tokenizer."}

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
echo "Tokenizer Path: $TOKENIZER_PATH"
echo "========================================================"

# --- Conda Environment Activation & Job Execution ---
export PATH="/opt/anaconda_python311/bin:$PATH"
echo "Starting Python inference script..."

# --- THE MAGIC LINE ---
# El script de python usará automáticamente la primera GPU que encuentre (cuda:0)
conda run -n llms-mx-env python src/inference.py \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --test_file "data/Rest-Mex_2025_test.xlsx" \
    --output_dir "submissions/$RUN_NAME" \
    --max_length 256 \
    --batch_size 64

echo "========================================================"
echo "Inference script finished."
echo "Submission file generated in: submissions/$RUN_NAME/CorpusChristi_Run.txt"
echo "Job finished."
echo "========================================================"
