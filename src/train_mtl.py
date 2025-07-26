# src/train_mtl.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import wandb
from datasets import Dataset, Features, ClassLabel
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
from eval_utils import compute_and_save_metrics # Lo adaptaremos para MTL

# --- 1. Custom Model for Multi-Task Learning ---
# This is the core of our MTL setup. We create a new model class
# that takes a base transformer (like BETO or MarIA) and adds three
# separate output layers (heads), one for each task.

class MultiTaskModel(PreTrainedModel):
    def __init__(self, config, model_name, num_labels_polarity, num_labels_type, num_labels_town):
        super().__init__(config)
        self.num_labels_polarity = num_labels_polarity
        self.num_labels_type = num_labels_type
        self.num_labels_town = num_labels_town

        # Load the base transformer model
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        
        # Define a dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Define the three classification heads
        self.classifier_polarity = nn.Linear(config.hidden_size, self.num_labels_polarity)
        self.classifier_type = nn.Linear(config.hidden_size, self.num_labels_type)
        self.classifier_town = nn.Linear(config.hidden_size, self.num_labels_town)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        polarity_label=None, # We'll use custom names for labels
        type_label=None,
        town_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through the base model
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Use the [CLS] token's representation for classification
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # Get logits from each classification head
        logits_polarity = self.classifier_polarity(pooled_output)
        logits_type = self.classifier_type(pooled_output)
        logits_town = self.classifier_town(pooled_output)

        # --- Loss Calculation ---
        loss = None
        if polarity_label is not None and type_label is not None and town_label is not None:
            loss_fct = CrossEntropyLoss()
            loss_polarity = loss_fct(logits_polarity.view(-1, self.num_labels_polarity), polarity_label.view(-1))
            loss_type = loss_fct(logits_type.view(-1, self.num_labels_type), type_label.view(-1))
            loss_town = loss_fct(logits_town.view(-1, self.num_labels_town), town_label.view(-1))
            
            # Combine the losses. We can weigh them if one task is more important.
            # For now, we'll weigh Polarity highest.
            loss = (0.6 * loss_polarity) + (0.2 * loss_type) + (0.2 * loss_town)

        if not return_dict:
            output = (logits_polarity, logits_type, logits_town) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Return a dictionary of outputs
        return {
            "loss": loss,
            "logits_polarity": logits_polarity,
            "logits_type": logits_type,
            "logits_town": logits_town,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

# --- 2. Custom Trainer for Multi-Task Learning ---
# This trainer needs to be aware of our multiple label columns.
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # The model's forward pass already computes the combined loss
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
    
    # We don't rename columns this time, we'll use their specific names
    
    print(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=args.max_length)

    print(f"Tokenizing datasets with max_length: {args.max_length}...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # Get the number of labels for each task
    num_labels_polarity = len(label_mappings['polarity'])
    num_labels_type = len(label_mappings['type'])
    num_labels_town = len(label_mappings['town'])
    
    print(f"Loading base model config: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    
    print("Instantiating Multi-Task Model...")
    model = MultiTaskModel(
        config=config,
        model_name=args.model_name,
        num_labels_polarity=num_labels_polarity,
        num_labels_type=num_labels_type,
        num_labels_town=num_labels_town
    )

    def compute_metrics_mtl(p: EvalPrediction):
        # p.predictions is now a tuple of logit arrays
        logits_polarity, logits_type, logits_town = p.predictions
        
        preds_polarity = np.argmax(logits_polarity, axis=1)
        preds_type = np.argmax(logits_type, axis=1)
        preds_town = np.argmax(logits_town, axis=1)
        
        # p.label_ids is also a tuple of label arrays
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
        metric_for_best_model="polarity_weighted_f1", # Our primary metric
        greater_is_better=True,
        save_total_limit=1,
        report_to="wandb" if args.use_wandb else "none",
        fp16=torch.cuda.is_available(),
        # This is needed to tell the Trainer which columns are labels
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
    
    # Save final model
    output_dir = os.path.join("models", args.run_name)
    trainer.save_model(output_dir)
    print(f"Final MTL model saved to {output_dir}")
    
    # We can add more detailed final evaluation later if needed
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
