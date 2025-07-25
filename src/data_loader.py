import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Function to preprocess text consistently
def preprocess_text(text):

    if not isinstance(text, str) or text == '':
        return ""
    
    # Try to fix mojibake issues
    try:
        # This assumes the text is in latin1 encoding and tries to decode it to utf-8
        text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If it fails, we simply continue with the text as is
        pass

    # Delete URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Clean newlines and multiple spaces
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text) # Busca 2 o más espacios

    return text.strip()


def load_and_prepare_dataset_for_mtl(
    data_file="Rest-Mex_2025_train.csv", 
    test_size=0.2, 
    random_state=42
):
    """
    Loads and prepares the dataset for multi-task learning (MTL).
    """
    # This ensures the script can be run from any directory.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, "data", data_file)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"El archivo de datos no se encontró en: {data_path}")
    
    print(f"Cargando y procesando datos desde {data_path}...")
    df = pd.read_csv(data_path)

    # --- Apply the preprocessing function ---
    df['Title'] = df['Title'].apply(preprocess_text)
    df['Review'] = df['Review'].apply(preprocess_text)
    
    df['text'] = (df['Title'] + ' ' + df['Review']).str.strip()

    # --- Mapping labels for Polarity, Type, and Town ---
    label_mappings = {}
    df['polarity_label'] = df['Polarity'] - 1
    label_mappings['polarity'] = {i: i + 1 for i in range(5)}
    
    type_labels, type_categories = pd.factorize(df['Type'])
    df['type_label'] = type_labels
    label_mappings['type'] = {i: category for i, category in enumerate(type_categories)}

    town_labels, town_categories = pd.factorize(df['Town'])
    df['town_label'] = town_labels
    label_mappings['town'] = {i: category for i, category in enumerate(town_categories)}
    
    print("Etiquetas para Polarity, Type y Town creadas.")

    final_cols = ['text', 'polarity_label', 'type_label', 'town_label']
    
    train_df, eval_df = train_test_split(
        df[final_cols], 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['polarity_label']
    )
    
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    print("Datos listos y convertidos a formato Hugging Face Dataset.")

    return {
        'train': train_dataset,
        'eval': eval_dataset,
        'label_mappings': label_mappings
    }

# To run the script and see an example of the processed training data
if __name__ == '__main__':
    data = load_and_prepare_dataset_for_mtl()
    print("\n--- Ejemplo de un registro del dataset de entrenamiento ---")
    print(data['train'][10])
    
    print("\n--- Mapeos de Etiquetas Generados ---")
    print("Polarity:", data['label_mappings']['polarity'])
    print("Type:", data['label_mappings']['type'])