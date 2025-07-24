import os
import json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_and_save_metrics(y_true, y_pred, run_name, results_dir="results"):
    """
    Calcula, guarda e imprime las métricas de evaluación y la matriz de confusión.

    Args:
        y_true (np.array): Etiquetas verdaderas.
        y_pred (np.array): Predicciones del modelo.
        run_name (str): Un nombre único para la ejecución (ej. 'BETO_baseline_final').
        results_dir (str): Directorio donde se guardarán los resultados.
    
    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Asegurarse de que las etiquetas son consistentes
    labels = sorted(list(set(y_true)))
    
    # Calcular métricas
    metrics = {
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels).tolist()
    }

    print(f"\n--- Métricas para la ejecución: {run_name} ---")
    print(f"  F1 Score Ponderado (Weighted F1): {metrics['weighted_f1']:.4f}")
    print("  F1 Score por Clase:")
    for i, f1 in enumerate(metrics['per_class_f1']):
        # Asumiendo que las clases son 1, 2, 3, 4, 5
        print(f"    Clase {i+1}: {f1:.4f}")
    
    # Guardar métricas en un archivo JSON
    metrics_path = os.path.join(results_dir, f"{run_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMétricas guardadas en: {metrics_path}")

    # Generar y guardar la matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Matriz de Confusión - {run_name}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    
    plot_path = os.path.join(results_dir, f"{run_name}_confusion_matrix.png")
    plt.savefig(plot_path)
    print(f"Matriz de confusión guardada en: {plot_path}")
    plt.close() # Cierra la figura para no mostrarla en ejecuciones de script

    return metrics