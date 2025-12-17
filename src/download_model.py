# src/download_model.py
import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_model(model_name, output_dir):
    """
    Downloads a model and its tokenizer from Hugging Face Hub and saves them to a local directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Downloading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    print(f"Downloading model '{model_name}'...")
    # We download the base model, not the one for sequence classification,
    # to avoid issues with the classification head. We will load it correctly in train.py
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    print("\n✅ Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on the Hub.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model and tokenizer.")
    args = parser.parse_args()
    download_model(args.model_name, args.output_dir)

