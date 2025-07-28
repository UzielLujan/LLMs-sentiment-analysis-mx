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
    Calcula, guarda e imprime las métricas de evaluación y las matrices de confusión 
    para un modelo de aprendizaje multitarea, incluyendo el "Score" final de la competencia.

    Args:
        predictions (tuple): Una tupla con los arrays de logits para cada tarea (polarity, type, town).
        labels (tuple): Una tupla con los arrays de etiquetas verdaderas para cada tarea.
        run_name (str): Un nombre único para la ejecución.
        label_mappings (dict): El diccionario que mapea los índices de las etiquetas a sus nombres.
        results_dir (str): Directorio donde se guardarán los resultados.
    """
    os.makedirs(results_dir, exist_ok=True)

    logits_polarity, logits_type, logits_town = predictions
    labels_polarity, labels_type, labels_town = labels

    preds_polarity = np.argmax(logits_polarity, axis=1)
    preds_type = np.argmax(logits_type, axis=1)
    preds_town = np.argmax(logits_town, axis=1)

    # --- Calcular Métricas Individuales (F1 Ponderado) ---
    f1_polarity = f1_score(labels_polarity, preds_polarity, average="weighted")
    f1_type = f1_score(labels_type, preds_type, average="weighted")
    f1_town = f1_score(labels_town, preds_town, average="weighted")

    # --- ¡NUEVO! Calcular el "Score" Oficial de la Competencia ---
    # Ponderación: 2x Polarity, 1x Type, 3x Town
    final_score = (2 * f1_polarity + 1 * f1_type + 3 * f1_town) / 6.0
    
    # --- Guardar todas las métricas en un diccionario ---
    metrics = {
        "Official_Score": final_score,
        "polarity_weighted_f1": f1_polarity,
        "type_weighted_f1": f1_type,
        "town_weighted_f1": f1_town,
        "polarity_accuracy": accuracy_score(labels_polarity, preds_polarity),
        "type_accuracy": accuracy_score(labels_type, preds_type),
        "town_accuracy": accuracy_score(labels_town, preds_town),
        "polarity_per_class_f1": f1_score(labels_polarity, preds_polarity, average=None).tolist(),
        "type_per_class_f1": f1_score(labels_type, preds_type, average=None).tolist(),
        "town_per_class_f1": f1_score(labels_town, preds_town, average=None).tolist(),
    }

    print(f"\n--- Métricas Finales para la ejecución: {run_name} ---")
    print(f"  Score Oficial (2-1-3): {metrics['Official_Score']:.4f}")
    print("  ------------------------------------")
    print(f"  F1 Polarity: {metrics['polarity_weighted_f1']:.4f}")
    print(f"  F1 Type:     {metrics['type_weighted_f1']:.4f}")
    print(f"  F1 Town:     {metrics['town_weighted_f1']:.4f}")

    # Guardar métricas en un archivo JSON
    metrics_path = os.path.join(results_dir, f"{run_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMétricas guardadas en: {metrics_path}")

    # --- Generar y Guardar Matrices de Confusión ---
    task_info = {
        "polarity": (labels_polarity, preds_polarity, list(label_mappings['polarity'].values())),
        "type": (labels_type, preds_type, list(label_mappings['type'].values())),
        "town": (labels_town, preds_town, list(label_mappings['town'].values()))
    }

    for task_name, (y_true, y_pred, display_labels) in task_info.items():
        cm = confusion_matrix(y_true, y_pred)
        
        figsize = (20, 18) if task_name == 'town' else (10, 8)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=display_labels, yticklabels=display_labels)
        plt.title(f'Matriz de Confusión - {task_name.capitalize()} - {run_name}')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        
        plot_path = os.path.join(results_dir, f"{run_name}_{task_name}_confusion_matrix.png")
        plt.savefig(plot_path)
        print(f"Matriz de confusión para '{task_name}' guardada en: {plot_path}")
        plt.close()


