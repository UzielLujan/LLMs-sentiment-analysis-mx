# **LLMs-sentiment-analysis-mx**

An advanced sentiment analysis project on Mexican tourism reviews, evolving from a baseline model to a high-performance, multi-task learning (MTL) pipeline. This project leverages modern Transformer architectures (BETO, MarIA) and is optimized for execution on High-Performance Computing (HPC) environments like SLURM clusters.

The primary goal was to significantly improve the sentiment classification (Polarity) by training the model on auxiliary tasks simultaneously: predicting the type of establishment (Type) and the location (Town).

## **Project Structure**

LLMs-sentiment-analysis-mx/  
├── data/                 \# Raw and processed data  
├── notebooks/            \# Jupyter notebooks for EDA and experimentation  
├── src/                  \# Source code for data loading, training, evaluation  
├── models/               \# Saved model checkpoints (ignored by Git)  
├── results/              \# Metrics and plots from model evaluation (ignored by Git)  
├── submissions/          \# Submission files for competitions  
├── .gitignore            \# Specifies intentionally untracked files to ignore  
├── requirements.txt      \# Project dependencies  
├── README.md             \# Project overview  
└── run\_\*.sh              \# SLURM submission scripts (located in root)

**Note on Submission Scripts:** The run\_\*.sh scripts are intentionally placed in the project root. Our extensive debugging on the target SLURM cluster revealed that executing submission scripts from the root directory is the most robust method to prevent complex relative path issues, ensuring a stable and reproducible workflow.

## **Key Features & Methodology**

This project follows a structured, multi-phase approach to model development and optimization:

1. **Reproducible Pipeline:** All code is modular (data\_loader, train, evaluation) and fully parameterized via command-line arguments, ensuring experiments are repeatable.  
2. **HPC Optimization:** The workflow was systematically debugged and optimized for the CIMAT Lab-SB SLURM cluster. This included solving environment inconsistencies, library conflicts, and network issues on compute nodes.  
3. **Distributed Training:** Implemented multi-GPU training using torchrun and Distributed Data Parallel (DDP), achieving a **\~3x speedup** (from \~1.5 hours to \~30 minutes per run) compared to a single-GPU setup.  
4. **Multi-Task Learning (MTL):** The core of the project. A custom MultiTaskModel was built on top of base transformers with three separate classification heads. This approach forces the model to learn contextual features from Type and Town, significantly boosting the primary Polarity classification performance.

## **Experiments & Results**

We conducted a series of experiments to establish a strong baseline and demonstrate the effectiveness of our final MTL model.

| Model | Training Strategy | GPUs | Time | Polarity F1-Score |  
| BETO | Single-Task | 1 | \~1h 25m | 0.7250 |  
| MarIA | Single-Task | 1 | \~1h 25m | 0.7265 |  
| MarIA | Single-Task (DDP) | 2 | \~30 min | 0.7318 |  
| MarIA | Multi-Task (DDP) | 2 | \~36 min | 0.7642 |  
The results clearly show that the Multi-Task Learning approach provided a significant performance leap, improving the F1-score by over **3 percentage points** compared to the best single-task model.

## **Setup and Installation**

1. **Clone the repository:**  
   git clone \[https://github.com/UzielLujan/LLMs-sentiment-analysis-mx.git\](https://github.com/UzielLujan/LLMs-sentiment-analysis-mx.git)  
   cd LLMs-sentiment-analysis-mx

2. **Create and activate the Conda environment:**  
   * **For** local **development:**  
     conda create \--name llms-mx-env python=3.10  
     conda activate llms-mx-env  
     pip install \-r requirements.txt  
     \# Install PyTorch with CUDA support for your local GPU  
     conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \-c pytorch \-c nvidia

   * **For** the CIMAT Lab-SB **cluster:** Follow the detailed steps discovered during our debugging process (manual environment creation with conda-forge and pip).

## **Usage**

### **Local Execution**

To train a model locally (for debugging or on a powerful machine), use the Python scripts directly. First, download a model from the Hub using the download\_model.py script, then run the training.

\# 1\. Download model (run once per model)  
python src/download\_model.py \--model\_name "BSC-TeMU/roberta-base-bne" \--output\_dir "models/MarIA\_local"

\# 2\. Run single-task training  
python src/train.py \--model\_name "models/MarIA\_local" \--run\_name "MarIA\_local\_test"

### **Usage on HPC Cluster (CIMAT Lab-SB)**

All experiments on the cluster are launched via the sbatch command from the project's root directory.

\# Launch a Multi-Task Learning training run with MarIA for 6 epochs  
sbatch run\_mtl\_2gpu.sh "models/MarIA\_local" "MarIA\_MTL\_6epochs" 6

The script handles environment activation and distributed training automatically. Monitor the job progress using squeue \-u \<your\_user\> and check the output in the logs/ directory.