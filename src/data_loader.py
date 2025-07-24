# src/data_loader.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_and_prepare_dataset(data_file="Train_Limpio.csv", test_size=0.2, random_state=42):
    """
    Carga el dataset desde un archivo CSV, realiza un preprocesamiento básico
    y lo divide en conjuntos de entrenamiento y evaluación.

    Args:
        data_file (str): Nombre del archivo de datos dentro de la carpeta 'data/'.
        test_size (float): Proporción del dataset a usar para evaluación.
        random_state (int): Semilla para la reproducibilidad de la división.

    Returns:
        tuple: Una tupla conteniendo (train_dataset, eval_dataset, unique_labels)
               donde los datasets están en formato Hugging Face `Dataset`.
    """
    data_path = os.path.join("data", data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"El archivo de datos no se encontró en: {data_path}")

    print(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocesamiento básico: combinar título y reseña
    df['text'] = df['Title'].fillna('') + ' ' + df['Review'].fillna('')
    
    # Las etiquetas deben empezar en 0 para los modelos de HF (Polarity 1-5 -> label 0-4)
    df['label'] = df['Polarity'] - 1
    
    unique_labels = sorted(df['label'].unique())

    # División estratificada para mantener la proporción de clases
    train_df, eval_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    
    # Convertir a formato Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    print("Datos cargados y divididos exitosamente.")
    print(f"Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
    print(f"Tamaño del conjunto de evaluación: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset, unique_labels