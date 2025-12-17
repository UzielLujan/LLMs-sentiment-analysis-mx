# **Sentiment Analysis for Mexican Tourism with Multi-Task Learning**

## **1. Summary**

This repository documents an advanced sentiment analysis project on tourism reviews in Mexico, originally developed for the REST-MEX 2025 competition. The project evolves from a single-task classification approach to a robust, modular, and optimized **Multi-Task Learning (MTL)** pipeline designed for execution on supercomputing clusters.

The main objective was to improve performance in **Polarity** (sentiment) classification by forcing Transformer models (BETO and MarIA) to simultaneously learn two auxiliary tasks: predicting the **Type** of establishment where the review was made (Type) and identifying the **Magical Town** (Town). The central hypothesis, successfully validated, was that this multitask approach would provide richer context, improving performance across all tasks.

The key strategic contribution of the project was the implementation of a loss function aligned with the competition's official evaluation metric. This strategic pivot, called **"Score-Optimized" (SO)**, resulted in a remarkable performance boost for this task with the **BETO (MTL-SO)** model, achieving an **official Score of 0.7822**.

## **2. Project Structure**
```bash
LLMs-sentiment-analysis-mx/  
├── data/               # Contains training and test datasets  
├── notebooks/          # Jupyter notebooks for EDA and initial prototyping  
├── src/                # Modular pipeline source code  
│   ├── data_loader.py  
│   ├── load_test.py  
│   ├── train_mtl.py  
│   ├── eval_utils_mtl.py  
│   └── inference.py  
├── models/             # Saved model checkpoints (ignored by Git)  
├── results/            # Metrics and confusion matrices (ignored by Git)  
├── submissions/        # Final prediction .txt files (ignored by Git)  
├── .gitignore  
├── requirements.txt    # Project dependencies  
├── README.md           # This file  
└── run_*.sh            # Launch scripts for the SLURM cluster
```
**Note on Launch Scripts:** The run_*.sh scripts are intentionally placed at the root of the project. Extensive debugging on the CIMAT SLURM cluster demonstrated that this is the most robust way to avoid complex relative path issues, ensuring a stable and reproducible workflow.

## **3. Methodology and Strategic Evolution**

The project was approached as a series of iterative experiments, each building on the lessons of the previous one.

### **Phase 1: Baselines and Pipeline Optimization**

* **Base Models:** Baselines were established with **BETO** (dccuchile/bert-base-spanish-wwm-cased) and **MarIA** (BSC-TeMU/roberta-base-bne) in a single-task approach.  
* **Optimization for HPC:** A modular pipeline was built and optimized for the CIMAT Lab-SB cluster, resolving numerous environment and dependency challenges.  
* **Distributed Training:** torchrun with Distributed Data Parallel (DDP) was implemented to train on 2 GPUs, achieving a **~3x speedup** (from ~1.5 hours to ~30 minutes per run).

### **Phase 2: The Leap to Multi-Task Learning (MTL)**

* **Hypothesis:** Training for all three tasks (Polarity, Type, Town) simultaneously would improve performance.  
* **Implementation:** A MultiTaskModel with three classification heads was developed. The initial loss function focused on Polarity.  
* **Result:** Resounding success. The Polarity F1-Score jumped from ~0.73 to **~0.765**, validating the MTL strategy.

### **Phase 3: The Strategic Pivot (Score-Optimized)**

* **The Discovery:** A detailed analysis of the competition documentation revealed that the final evaluation metric was not the Polarity F1 but a weighted "Score":  
  ```
  Score = 62×F1polarity + 1×F1type + 3×F1town
  ```
* **The New Hypothesis:** Aligning the training loss function with this formula should maximize the final Score.  
* **Final Implementation:** The loss function in train_mtl.py was modified to reflect the 62-1-3 weighting, and the Trainer's optimization metric was adjusted to save the checkpoint with the highest Score.

## **4. Final Results**

The "Score-Optimized" (SO) strategy was decisive and revealed **BETO** as the undisputed champion.

| Model | Strategy | F1 Polarity | F1 Type | F1 Town | Official Score |
| :---- | :---- | :---- | :---- | :---- | :---- |
| BETO (MTL) | Focus on Polarity | 0.7656 | 0.9770 | 0.6894 | 0.7627 |
| **BETO (MTL-SO)** | **Score-Optimized** | 0.7592 | 0.9782 | **0.7322** | **0.7822 (+0.0195)** |

The SO model intelligently sacrificed minimal performance in Polarity to achieve massive gains in Town, the most valuable task, resulting in a drastic increase in the overall Score.

## **5. Usage and Instructions**

### **5.1. Environment Setup**

Using conda to manage the environment is recommended.

# 1. Clone the repository  
git clone [https://github.com/UzielLujan/LLMs-sentiment-analysis-mx.git](https://github.com/UzielLujan/LLMs-sentiment-analysis-mx.git)  
cd LLMs-sentiment-analysis-mx

# 2. Create and activate the environment  
conda create --name llms-mx-env python=3.10  
conda activate llms-mx-env

# 3. Install dependencies  
pip install -r requirements.txt  
# (Optional, for local GPU)  
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

**Note:** For the CIMAT cluster, a more detailed manual installation of libraries is required.

### **5.2. Training on the Cluster**

To launch Score-optimized training with BETO for 6 epochs:

sbatch run_mtl_2gpu.sh "models/BETO_local" "BETO_MTL_SO_final" 6

### **5.3. Inference**

To generate the prediction file with the champion model (BETO_MTL_SO):

sbatch run_prediction.sh "models/BETO_MTL_SO" "BETO_final_submission" "dccuchile/bert-base-spanish-wwm-cased"

The CorpusChristi_Run.txt file will be generated in the submissions/BETO_final_submission/ folder.

## **6. Conclusion**

This project demonstrates the power of Multi-Task Learning and, more critically, the importance of **aligning training objectives with specific evaluation metrics**. This alignment was the key that unlocked the model's full potential.

Uziel Isaí Lujan López — M.Sc. in Statistical Computing at CIMAT  
[LinkedIn](https://www.linkedin.com/in/uziel-lujan/) | [GitHub](https://github.com/UzielLujan)
