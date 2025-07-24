# LLMs-sentiment-analysis-mx

Advanced sentiment analysis of Mexican tourism reviews using modern Transformer architectures and Multi-Task Learning. This project aims to improve upon a previous baseline by implementing reproducible, modular code and leveraging state-of-the-art models.

## Project Structure

```
LLMs-sentiment-analysis-mx/
├── data/                 # Raw and processed data
├── notebooks/            # Jupyter notebooks for EDA and experimentation
├── src/                  # Source code for data loading, training, evaluation
├── models/               # Saved model checkpoints (ignored by Git)
├── results/              # Metrics and plots from model evaluation (ignored by Git)
├── scripts/              # Shell scripts for running experiments (e.g., on SLURM)
├── .gitignore            # Specifies intentionally untracked files to ignore
├── requirements.txt      # Project dependencies
└── README.md             # Project overview
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/UzielLujan/LLMs-sentiment-analysis-mx.git](https://github.com/UzielLujan/LLMs-sentiment-analysis-mx.git)
    cd LLMs-sentiment-analysis-mx
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create --name llms-mx-env python=3.10
    conda activate llms-mx-env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To explore the data, run the EDA notebook located in the `notebooks/` directory.

To train a model, use the `train.py` script:
```bash
python src/train.py --model_name "BSC-TeMU/roberta-base-bne" --run_name "MarIA_baseline"
```