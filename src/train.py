# src/train.py

import os
import argparse
import numpy as np
import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.utils.class_weight import compute_class_weight

# <-- CAMBIO: Importamos nuestra función de data_loader
from data_loader import load_and_prepare_dataset
from eval_utils import compute_and_save_metrics

# El Custom Trainer sigue siendo necesario aquí para los class weights
class WeightedLossTrainer(Trainer):
    # ... (El código del WeightedLossTrainer que te di antes va aquí, sin cambios)
    def __init__(self, *args, **kwargs):
        self.class_weights = kwargs.pop("class_weights", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main(args):
    # --- 1. Inicialización y Configuración ---
    if args.use_wandb:
        wandb.init(project="LLMs-sentiment-analysis-mx", name=args.run_name, config=args)

    # --- 2. Carga de Datos (Ahora delegada) ---
    # <-- CAMBIO: Toda la carga de datos se delega a nuestro módulo
    train_dataset, eval_dataset, unique_labels = load_and_prepare_dataset()

    # --- 3. Tokenización ---
    print(f"Cargando tokenizer para el modelo: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # --- 4. Configuración del Modelo y Class Weights ---
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(unique_labels))
    
    print("Calculando pesos de clase para manejar desbalance...")
    # <-- CAMBIO: Usamos los datos del Hugging Face Dataset para calcular los pesos
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(unique_labels),
        y=np.array(train_dataset['label'])
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    print(f"Pesos calculados: {class_weights}")

    # --- 5. Lógica de Entrenamiento ---
    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average="weighted")
        return {"weighted_f1": f1}

    training_args = TrainingArguments(
        output_dir=os.path.join("models", args.run_name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        logging_dir=os.path.join("logs", args.run_name),
        report_to="wandb" if args.use_wandb else "none",
        fp16=torch.cuda.is_available(), # Activa mixed precision si hay GPU
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print(f"\n--- Iniciando entrenamiento para: {args.run_name} ---")
    trainer.train()

    # --- 6. Evaluación Final y Guardado de Resultados ---
    print("\nEvaluando el modelo final en el conjunto de evaluación...")
    final_predictions = trainer.predict(eval_dataset)
    y_true = final_predictions.label_ids
    y_pred = np.argmax(final_predictions.predictions, axis=1)

    compute_and_save_metrics(y_true, y_pred, args.run_name)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo Transformer para análisis de sentimientos.")
    parser.add_argument("--model_name", type=str, required=True, help="Nombre del modelo de Hugging Face.")
    parser.add_argument("--run_name", type=str, required=True, help="Nombre para esta ejecución (usado para carpetas y W&B).")
    parser.add-argument("--epochs", type=int, default=3, help="Número de épocas de entrenamiento.")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño del lote por dispositivo.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Tasa de aprendizaje.")
    parser.add_argument("--use_wandb", action="store_true", help="Activa el logging con Weights & Biases.")
    
    args = parser.parse_args()
    main(args)