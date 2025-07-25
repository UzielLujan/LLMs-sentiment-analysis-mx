import os
import argparse
import numpy as np
import torch
import wandb
from datasets import Dataset, Features, ClassLabel
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# Import our custom modules
from data_loader import load_and_prepare_dataset_for_mtl
from eval_utils import compute_and_save_metrics

class WeightedLossTrainer(Trainer):
    """
    A custom Trainer that takes full control of the loss calculation
    to apply class weights correctly.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.model.config.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main(args):
    """
    Main function to orchestrate the model training and evaluation pipeline.
    """
    if args.use_wandb:
        wandb.init(project="LLMs-sentiment-analysis-mx", name=args.run_name, config=args)

    print("Loading and preparing data...")
    data = load_and_prepare_dataset_for_mtl()
    train_dataset = data['train']
    eval_dataset = data['eval']
    label_mappings = data['label_mappings']
    
    train_dataset = train_dataset.rename_column("polarity_label", "labels")
    eval_dataset = eval_dataset.rename_column("polarity_label", "labels")

    print("Casting label column to correct data type...")
    features = train_dataset.features.copy()
    features['labels'] = ClassLabel(num_classes=len(label_mappings['polarity']))
    train_dataset = train_dataset.cast(features)
    eval_dataset = eval_dataset.cast(features)

    print(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- CAMBIO CLAVE: Usamos el max_length del argumento ---
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=args.max_length)

    print(f"Tokenizing datasets with max_length: {args.max_length}...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    num_polarity_labels = len(label_mappings['polarity'])
    
    print("Calculating class weights to handle data imbalance...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(list(label_mappings['polarity'].keys())),
        y=np.array(train_dataset['labels'])
    )
    
    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_polarity_labels
    )
    model.config.class_weights = torch.tensor(class_weights, dtype=torch.float)

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
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        report_to="wandb" if args.use_wandb else "none",
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"\n--- Starting training for run: {args.run_name} ---")
    trainer.train()
    print("--- Training finished ---")

    print("\nEvaluating the best model on the evaluation set...")
    final_predictions = trainer.predict(eval_dataset)
    
    y_pred = np.argmax(final_predictions.predictions, axis=1)
    y_true = final_predictions.label_ids

    compute_and_save_metrics(y_true, y_pred, args.run_name)
    
    if args.use_wandb:
        wandb.finish()
    
    print(f"\n✅ Run '{args.run_name}' completed. Results saved in 'results/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for sentiment analysis.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model from Hugging Face Hub.")
    parser.add_argument("--run_name", type=str, required=True, help="A unique name for this training run.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization.")
    parser.add_argument("--use_wandb", action="store_true", help="Set this flag to enable logging with Weights & Biases.")
    
    args = parser.parse_args()
    main(args)

