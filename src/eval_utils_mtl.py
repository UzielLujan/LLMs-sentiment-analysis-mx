# src/eval_utils_mtl.py

import os
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_and_save_mtl_metrics(
    predictions, 
    labels, 
    run_name, 
    label_mappings,
    results_dir="results"
):
    """
    Computes, saves, and prints evaluation metrics and confusion matrices for a multi-task learning model.

    Args:
        predictions (tuple): A tuple containing the logit arrays for each task (polarity, type, town).
        labels (tuple): A tuple containing the true label arrays for each task.
        run_name (str): A unique name for the run.
        label_mappings (dict): The dictionary mapping label indices to names.
        results_dir (str): Directory where results will be saved.
    """
    os.makedirs(results_dir, exist_ok=True)

    logits_polarity, logits_type, logits_town = predictions
    labels_polarity, labels_type, labels_town = labels

    preds_polarity = np.argmax(logits_polarity, axis=1)
    preds_type = np.argmax(logits_type, axis=1)
    preds_town = np.argmax(logits_town, axis=1)

    # --- Calculate Metrics ---
    metrics = {
        "polarity_weighted_f1": f1_score(labels_polarity, preds_polarity, average="weighted"),
        "type_accuracy": accuracy_score(labels_type, preds_type),
        "town_accuracy": accuracy_score(labels_town, preds_town),
        "polarity_per_class_f1": f1_score(labels_polarity, preds_polarity, average=None).tolist()
    }

    print(f"\n--- Final Metrics for run: {run_name} ---")
    print(f"  Polarity Weighted F1: {metrics['polarity_weighted_f1']:.4f}")
    print(f"  Type Accuracy:        {metrics['type_accuracy']:.4f}")
    print(f"  Town Accuracy:        {metrics['town_accuracy']:.4f}")
    
    # Save metrics to a JSON file
    metrics_path = os.path.join(results_dir, f"{run_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {metrics_path}")

    # --- Generate and Save Confusion Matrices ---
    task_info = {
        "polarity": (labels_polarity, preds_polarity, list(label_mappings['polarity'].values())),
        "type": (labels_type, preds_type, list(label_mappings['type'].values())),
        "town": (labels_town, preds_town, list(label_mappings['town'].values()))
    }

    for task_name, (y_true, y_pred, display_labels) in task_info.items():
        cm = confusion_matrix(y_true, y_pred)
        
        # Adjust figure size for the 'town' task which has many labels
        figsize = (20, 18) if task_name == 'town' else (10, 8)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=display_labels, yticklabels=display_labels)
        plt.title(f'Confusion Matrix - {task_name.capitalize()} - {run_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        
        plot_path = os.path.join(results_dir, f"{run_name}_{task_name}_confusion_matrix.png")
        plt.savefig(plot_path)
        print(f"Confusion matrix for '{task_name}' saved to: {plot_path}")
        plt.close()

