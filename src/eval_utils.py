import os
import json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_and_save_metrics(y_true, y_pred, run_name, results_dir="results"):
    """
    Computes, saves, and prints evaluation metrics and the confusion matrix.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Model predictions.
        run_name (str): A unique name for the run (e.g., 'BETO_baseline_final').
        results_dir (str): Directory where results will be saved.
    
    Returns:
        dict: Dictionary with the calculated metrics.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Ensure labels are consistent
    labels = sorted(list(set(y_true)))
    
    # Calculate metrics
    metrics = {
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels).tolist()
    }

    print(f"\n--- Metrics for run: {run_name} ---")
    print(f"  Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    print("  F1 Score per Class:")
    for i, f1 in enumerate(metrics['per_class_f1']):
        # Assuming classes are 1, 2, 3, 4, 5
        print(f"    Class {i+1}: {f1:.4f}")
    
    # Save metrics to a JSON file
    metrics_path = os.path.join(results_dir, f"{run_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {metrics_path}")

    # Generate and save the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {run_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = os.path.join(results_dir, f"{run_name}_confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"Confusion matrix saved to: {plot_path}")
    plt.close() # Close the figure to avoid displaying it in script executions

    return metrics