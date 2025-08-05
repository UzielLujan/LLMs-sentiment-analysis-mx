# src/load_test.py

import os
import pandas as pd
from datasets import Dataset

 # Import the preprocessing function from our original data_loader
 # to ensure the text is cleaned in exactly the same way.
from data_loader import preprocess_text

def load_and_prepare_test_data(test_file="data/Rest-Mex_2025_test.xlsx"):
    """
    Loads the .xlsx test file, processes it consistently
    with the training data, and returns it as a Hugging Face Dataset.
    """
    # Build the absolute path to the test file, same as in data_loader.py
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, "data", os.path.basename(test_file))

    print(f"Loading and processing test data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test file not found at: {data_path}")

    # Read the Excel file, assuming it has a header.
    df = pd.read_excel(data_path)

    # Preprocess and concatenate 'Title' and 'Review' into a single 'text' column.
    df['Title'] = df['Title'].apply(preprocess_text)
    df['Review'] = df['Review'].apply(preprocess_text)
    df['text'] = df['Title'] + ' ' + df['Review']

    # Keep only the columns needed for inference.
    df = df[['ID', 'text']]

    # Convert the pandas DataFrame to a Hugging Face Dataset.
    test_dataset = Dataset.from_pandas(df)

    print(f"Test data ready. Number of examples: {len(test_dataset)}")

    return test_dataset

 # This block allows you to run "python src/load_test.py" to verify
 # that loading and processing work correctly locally.
if __name__ == "__main__":
    print("--- Running test data loading check ---")
    try:
        # No need to build the path here, the function already does it
        dataset = load_and_prepare_test_data()
        print("\n✅ Test data loaded successfully.")
        print("Example of the first 3 processed records:")
        for i in range(3):
            print(dataset[i])

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Make sure the file 'Rest-Mex_2025_test.xlsx' is in the 'data/' folder.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

