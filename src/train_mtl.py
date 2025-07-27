# src/train_mtl.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import wandb
from datasets import Dataset, ClassLabel
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    PreTrainedModel,
    AutoConfig
)
from sklearn.metrics import f1_score, accuracy_score

# Import our custom modules
from data_loader import load_and_prepare_dataset_for_mtl
from eval_utils_mtl import compute_and_save_mtl_metrics
# --- Custom Model for Multi-Task Learning (no changes here) ---
class MultiTaskModel(PreTrainedModel):
    def __init__(self, config, model_name, num_labels_polarity, num_labels_type, num_labels_town):
        super().__init__(config)
        self.num_labels_polarity = num_labels_polarity
        self.num_labels_type = num_labels_type
        self.num_labels_town = num_labels_town

        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier_polarity = nn.Linear(config.hidden_size, self.num_labels_polarity)
        self.classifier_type = nn.Linear(config.hidden_size, self.num_labels_type)
        self.classifier_town = nn.Linear(config.hidden_size, self.num_labels_town)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        polarity_label=None,
        type_label=None,
        town_label=None,
        return_dict=None,
        **kwargs, 
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits_polarity = self.classifier_polarity(pooled_output)
        logits_type = self.classifier_type(pooled_output)
        logits_town = self.classifier_town(pooled_output)

        loss = None
        if polarity_label is not None and type_label is not None and town_label is not None:
            loss_fct = CrossEntropyLoss()
            loss_polarity = loss_fct(logits_polarity.view(-1, self.num_labels_polarity), polarity_label.view(-1))
            loss_type = loss_fct(logits_type.view(-1, self.num_labels_type), type_label.view(-1))
            loss_town = loss_fct(logits_town.view(-1, self.num_labels_town), town_label.view(-1))
            
            loss = (0.6 * loss_polarity) + (0.2 * loss_type) + (0.2 * loss_town)

        if not return_dict:
            output = (logits_polarity, logits_type, logits_town) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits_polarity": logits_polarity,
            "logits_type": logits_type,
            "logits_town": logits_town,
        }

# --- Custom Trainer for Multi-Task Learning (no changes here) ---
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def main(args):
    """
    Main function for the Multi-Task Learning pipeline.
    """
    if args.use_wandb:
        wandb.init(project="LLMs-sentiment-analysis-mx", name=args.run_name, config=args)

    print("Loading and preparing data for MTL...")
    data = load_and_prepare_dataset_for_mtl()
    train_dataset = data['train']
    eval_dataset = data['eval']
    label_mappings = data['label_mappings']
    
    print(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=args.max_length)

    print(f"Tokenizing datasets with max_length: {args.max_length}...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # --- CORRECCIÓN DEFINITIVA AQUÍ ---
    # En lugar de usar un objeto `Features` rígido, hacemos un cast
    # individual de cada columna de etiquetas. Es más flexible y robusto.
    print("Casting label columns to correct data types...")
    for label_name, mapping in label_mappings.items():
        num_classes = len(mapping)
        class_label_feature = ClassLabel(num_classes=num_classes)
        train_dataset = train_dataset.cast_column(f"{label_name}_label", class_label_feature)
        eval_dataset = eval_dataset.cast_column(f"{label_name}_label", class_label_feature)
    # --- FIN DE LA CORRECCIÓN ---
    
    print(f"Loading base model config: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    
    print("Instantiating Multi-Task Model...")
    model = MultiTaskModel(
        config=config,
        model_name=args.model_name,
        num_labels_polarity=len(label_mappings['polarity']),
        num_labels_type=len(label_mappings['type']),
        num_labels_town=len(label_mappings['town'])
    )

    def compute_metrics_mtl(p: EvalPrediction):
        logits_polarity, logits_type, logits_town = p.predictions
        preds_polarity = np.argmax(logits_polarity, axis=1)
        preds_type = np.argmax(logits_type, axis=1)
        preds_town = np.argmax(logits_town, axis=1)
        
        labels_polarity, labels_type, labels_town = p.label_ids

        f1_polarity = f1_score(labels_polarity, preds_polarity, average="weighted")
        acc_type = accuracy_score(labels_type, preds_type)
        acc_town = accuracy_score(labels_town, preds_town)
        
        return {
            "polarity_weighted_f1": f1_polarity,
            "type_accuracy": acc_type,
            "town_accuracy": acc_town,
        }

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
        metric_for_best_model="polarity_weighted_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="wandb" if args.use_wandb else "none",
        fp16=torch.cuda.is_available(),
        label_names=["polarity_label", "type_label", "town_label"],
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mtl,
    )

    print(f"\n--- Starting MTL training for run: {args.run_name} ---")
    trainer.train()
    print("--- Training finished ---")

    print("\nEvaluating the best MTL model on the evaluation set...")
    final_predictions = trainer.predict(eval_dataset)
    
    compute_and_save_mtl_metrics(
        predictions=final_predictions.predictions,
        labels=final_predictions.label_ids,
        run_name=args.run_name,
        label_mappings=label_mappings
    )

    output_dir = os.path.join("models", args.run_name)
    trainer.save_model(output_dir)
    print(f"Final MTL model saved to {output_dir}")
    
    print(f"\n✅ Run '{args.run_name}' completed. Final metrics: {final_predictions.metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Multi-Task Transformer model.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Path to the local base model or name from Hugging Face Hub.")
    parser.add_argument("--run_name", type=str, required=True, help="A unique name for this training run.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size PER DEVICE.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization.")
    parser.add_argument("--use_wandb", action="store_true", help="Set this flag to enable logging with Weights & Biases.")
    
    args = parser.parse_args()
    main(args)


