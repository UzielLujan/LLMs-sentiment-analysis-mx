# src/inference.py

import argparse
import os
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

# Importamos el modelo y la función para obtener los mapeos de etiquetas
from train_mtl import MultiTaskModel 
from data_loader import load_and_prepare_dataset_for_mtl

def load_test_data(test_file_path):
    """
    Carga el archivo de prueba .xlsx y lo convierte en un Dataset de Hugging Face.
    """
    print(f"Cargando datos de prueba desde: {test_file_path}")
    df = pd.read_excel(test_file_path, header=None)
    df.columns = ['ID', 'text']
    
    print("Columnas del DataFrame de prueba:", df.columns.tolist())
    print("Primeras 5 filas del DataFrame de prueba:")
    print(df.head())

    test_dataset = Dataset.from_pandas(df)
    print(f"Datos de prueba cargados. Número de ejemplos: {len(test_dataset)}")
    return test_dataset

def main(args):
    """
    Función principal para ejecutar la inferencia y generar el archivo de sumisión unificado.
    """
    print("--- Iniciando Pipeline de Inferencia ---")

    # --- 1. Cargar Mapeos, Modelo y Tokenizer ---
    print("Cargando mapeos de etiquetas desde el dataset de entrenamiento...")
    # Llamamos a la función de carga solo para obtener los mapeos. Es la forma más robusta
    # de asegurar que usamos exactamente los mismos mapeos que en el entrenamiento.
    data_info = load_and_prepare_dataset_for_mtl()
    label_mappings = data_info['label_mappings']
    
    # Creamos mapeos inversos para convertir índices a nombres de etiquetas
    idx_to_town = {v: k for k, v in label_mappings['town'].items()}
    idx_to_type = {v: k for k, v in label_mappings['type'].items()}

    print(f"Cargando modelo pre-entrenado desde: {args.model_path}")
    model = MultiTaskModel.from_pretrained(
        args.model_path,
        num_labels_polarity=len(label_mappings['polarity']),
        num_labels_type=len(label_mappings['type']),
        num_labels_town=len(label_mappings['town'])
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print("Modelo y tokenizer cargados exitosamente.")

    # --- 2. Cargar y Preparar Datos de Prueba ---
    test_dataset = load_test_data(args.test_file)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=args.max_length)

    print("Tokenizando datos de prueba...")
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # --- 3. Realizar Predicciones ---
    training_args = TrainingArguments(
        output_dir="./temp_results",
        per_device_eval_batch_size=args.batch_size,
        do_predict=True,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=training_args)

    print("Realizando predicciones en el conjunto de prueba...")
    predictions = trainer.predict(tokenized_test_dataset)
    
    logits_polarity, logits_type, logits_town = predictions.predictions
    preds_polarity = np.argmax(logits_polarity, axis=1)
    preds_type_idx = np.argmax(logits_type, axis=1)
    preds_town_idx = np.argmax(logits_town, axis=1)
    print("Predicciones generadas.")

    # --- 4. Generar Archivo de Salida Unificado ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # El nombre del equipo es "CorpusChristi" y el archivo es único
    output_file_path = os.path.join(args.output_dir, f"CorpusChristi_Run.txt")
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        # Iteramos sobre cada una de las predicciones
        for i in range(len(preds_polarity)):
            instance_id = test_dataset[i]['ID']
            
            # Polarity: El modelo predice 0-4, el formato pide 1-5
            polarity_pred = preds_polarity[i] + 1
            
            # Town y Type: Convertimos el índice predicho a su etiqueta de texto
            town_pred = idx_to_town[preds_town_idx[i]]
            type_pred = idx_to_type[preds_type_idx[i]]
            
            # Escribimos la línea con el formato exacto: rest-mex | id | Polarity | Town | Type
            output_line = f"rest-mex\t{instance_id}\t{polarity_pred}\t{town_pred}\t{type_pred}\n"
            f.write(output_line)
            
    print(f"\nArchivo de sumisión unificado generado en: {output_file_path}")
    print("\n✅ Proceso de inferencia completado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar inferencia con un modelo MTL entrenado.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al directorio del modelo MTL entrenado (ej. 'models/BETO_MTL').")
    parser.add_argument("--tokenizer_path", type=str, default="dccuchile/bert-base-spanish-wwm-cased", help="Ruta o nombre del tokenizer.")
    parser.add_argument("--test_file", type=str, default="data/Rest-Mex_2025_test.xlsx", help="Ruta al archivo de prueba .xlsx.")
    parser.add_argument("--output_dir", type=str, default="submissions", help="Directorio donde se guardará el archivo de predicción.")
    parser.add_argument("--max_length", type=int, default=256, help="Máxima longitud de la secuencia para el tokenizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño del lote por dispositivo para la predicción.")
    
    args = parser.parse_args()
    main(args)
