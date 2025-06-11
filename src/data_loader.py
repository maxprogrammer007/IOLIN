# src/data_loader.py
# This module is now updated to process the Wine Quality dataset.
# This will be a cleaner, easier benchmark to test our model's maximum potential.

import pandas as pd
import os
import requests
from io import StringIO

# --- Configuration for the Wine Dataset ---
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_wine_quality.csv")

def fetch_wine_data(url=DATASET_URL):
    """Downloads the dataset and returns it as a DataFrame."""
    print(f"Downloading Wine Quality dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status() # Will raise an exception for bad status codes
        return pd.read_csv(StringIO(response.text), sep=';')
    except Exception as e:
        print(f"Failed to download dataset. Error: {e}")
        return None

def preprocess_wine_data(processed_path=PROCESSED_DATA_PATH):
    """
    Loads the raw wine data, discretizes the target, and saves the processed data.
    """
    df = fetch_wine_data()
    if df is None:
        return

    print("--- Starting Wine Quality Data Preprocessing ---")

    # The dataset is already clean (no missing values).
    # Our main task is to discretize the 'quality' target variable.
    # Original quality is a score from 3 to 8.
    
    # We will create 3 classes:
    # 0: Low Quality (quality < 6)
    # 1: Medium Quality (quality = 6)
    # 2: High Quality (quality > 6)
    
    bins = [0, 5, 6, 10]
    labels = [0, 1, 2] # Low, Medium, High
    df['Target'] = pd.cut(df['quality'], bins=bins, labels=labels, right=True)
    
    # We can drop the original 'quality' column as 'Target' is our new objective
    df_processed = df.drop(columns=['quality'])
    
    # Save the processed data
    # Ensure the directory exists
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    
    print("Preprocessing complete. New dataset saved.")
    print(f"Final dataset shape: {df_processed.shape}")
    print("\nTarget Class Distribution:")
    print(df_processed['Target'].value_counts(normalize=True))

if __name__ == '__main__':
    preprocess_wine_data()
