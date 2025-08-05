# src/inference.py

import argparse
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Importamos nuestros módulos personalizados
from train_mtl import MultiTaskModel 
from data_loader import load_and_prepare_dataset_for_mtl
from load_test import load_and_prepare_test_data

def main(args):
    """
    Función principal para ejecutar la inferencia y generar el archivo de sumisión unificado.
    """
    print("--- Iniciando Pipeline de Inferencia ---")

    # --- 1. Cargar Mapeos, Modelo y Tokenizer ---
    print("Cargando mapeos de etiquetas...")
    data_info = load_and_prepare_dataset_for_mtl()
    label_mappings = data_info['label_mappings']
    
    # --- ¡LA CORRECCIÓN ESTÁ AQUÍ! ---
    # Usamos los diccionarios de mapeo directamente, sin invertirlos.
    # El formato ya es {índice: nombre_etiqueta}.
    idx_to_town = label_mappings['town']
    idx_to_type = label_mappings['type']
    # --- FIN DE LA CORRECCIÓN ---

    print(f"Cargando modelo pre-entrenado desde: {args.model_path}")
    model = MultiTaskModel.from_pretrained(
        args.model_path,
        model_name=args.tokenizer_path,
        num_labels_polarity=len(label_mappings['polarity']),
        num_labels_type=len(label_mappings['type']),
        num_labels_town=len(label_mappings['town'])
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print("Modelo y tokenizer cargados.")

    # --- 2. Cargar y Preparar Datos de Prueba ---
    test_dataset = load_and_prepare_test_data(args.test_file)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")

    print("Tokenizando datos de prueba...")
    # Guardamos los IDs originales antes de que el mapeo los elimine
    original_ids = test_dataset['ID']
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['ID', 'text'])
    tokenized_test_dataset.set_format('torch')

    # --- 3. Bucle de Inferencia Manual ---
    print("Realizando predicciones con bucle manual...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=args.batch_size)

    all_preds_polarity = []
    all_preds_type = []
    all_preds_town = []

    for batch in tqdm(test_dataloader, desc="Prediciendo"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits_polarity = outputs['logits'][0]
        logits_type = outputs['logits'][1]
        logits_town = outputs['logits'][2]

        preds_polarity = torch.argmax(logits_polarity, dim=1).cpu().numpy()
        preds_type = torch.argmax(logits_type, dim=1).cpu().numpy()
        preds_town = torch.argmax(logits_town, dim=1).cpu().numpy()

        all_preds_polarity.extend(preds_polarity)
        all_preds_type.extend(preds_type)
        all_preds_town.extend(preds_town)

    print("Predicciones generadas.")

    # --- 4. Generar Archivo de Salida Unificado ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, f"CorpusChristi_Run.txt")
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        for i in range(len(all_preds_polarity)):
            # Usamos la lista de IDs que guardamos antes
            instance_id = original_ids[i]
            
            polarity_pred = all_preds_polarity[i] + 1
            town_pred = idx_to_town[all_preds_town[i]]
            type_pred = idx_to_type[all_preds_type[i]]
            
            output_line = f"rest-mex\t{instance_id}\t{polarity_pred}\t{town_pred}\t{type_pred}\n"
            f.write(output_line)
            
    print(f"\nArchivo de sumisión unificado generado en: {output_file_path}")
    print("\n✅ Proceso de inferencia completado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar inferencia con un modelo MTL entrenado.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al directorio del modelo MTL entrenado.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Ruta o nombre del tokenizer original.")
    parser.add_argument("--test_file", type=str, default="data/Rest-Mex_2025_test.xlsx", help="Ruta al archivo de prueba .xlsx.")
    parser.add_argument("--output_dir", type=str, default="submissions", help="Directorio donde se guardará el archivo de predicción.")
    parser.add_argument("--max_length", type=int, default=256, help="Máxima longitud de la secuencia para el tokenizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño del lote por dispositivo para la predicción.")
    
    args = parser.parse_args()
    main(args)

