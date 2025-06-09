# src/data_loader.py
# This module will handle loading, cleaning, and preprocessing the data.
# It will implement the steps from Phase 0 of our plan, including
# creating lagged features and discretizing the target variable.

import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
RAW_DATA_PATH = os.path.join("data", "raw", "beijing_pm25.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_beijing_pm25.csv")

def fetch_data(url=DATASET_URL, path=RAW_DATA_PATH):
    """Downloads the dataset if it doesn't already exist."""
    if os.path.exists(path):
        print(f"Dataset already exists at {path}")
        return
    print(f"Downloading dataset from {url}...")
    try:
        df = pd.read_csv(url)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Dataset saved to {path}")
    except Exception as e:
        print(f"Failed to download dataset. Error: {e}")
        return None

def preprocess_data(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_DATA_PATH):
    """
    Loads the raw data, cleans it, engineers features based on the IOLIN paper,
    and saves the processed data.
    """
    if not os.path.exists(raw_path):
        print(f"Raw data not found at {raw_path}. Please fetch the data first.")
        return

    print("Starting data preprocessing...")
    df = pd.read_csv(raw_path)

    # 1. --- Data Cleaning ---
    # Combine date/time columns into a single datetime index
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)

    # The paper fills missing values with an average of preceding and successive values.
    # We'll use a forward-fill then a backward-fill, which is a robust way to handle this.
    df['pm2.5'].fillna(method='ffill', inplace=True)
    df['pm2.5'].fillna(method='bfill', inplace=True)
    print("Missing values in 'pm2.5' handled.")

    # 2. --- Feature Engineering (replicating Table 1 from the paper) ---
    df_processed = pd.DataFrame(index=df.index)

    # Time-based attributes
    df_processed['Year'] = df.index.year
    df_processed['Month_of_year'] = df.index.month
    df_processed['Week_of_year'] = df.index.isocalendar().week.astype(int)
    df_processed['Day_of_year'] = df.index.dayofyear
    df_processed['Day_of_month'] = df.index.day
    df_processed['Day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
    df_processed['Hour'] = df.index.hour

    # Day type attribute (Simplified: 0=Weekday, 1=Weekend)
    df_processed['Day_type'] = (df.index.dayofweek >= 5).astype(int)

    # Other relevant attributes from the original dataset
    df_processed['Dew_Point'] = df['DEWP']
    df_processed['Temperature'] = df['TEMP']
    df_processed['Pressure'] = df['PRES']
    df_processed['Wind_Speed'] = df['Iws']

    # Lagged features (replicating "Previous ... volume")
    # Note: 1 day = 24 hours, 1 week = 168 hours
    df_processed['pm2.5_prev_hour'] = df['pm2.5'].shift(1)
    df_processed['pm2.5_prev_day'] = df['pm2.5'].shift(24)
    df_processed['pm2.5_prev_week'] = df['pm2.5'].shift(168)
    print("Time-based and lagged features created.")

    # 3. --- Target Attribute Creation ---
    # The paper discretizes the target into 7 equal-frequency intervals.
    # We use qcut for this. The target is the NEXT hour's pm2.5 value.
    df_processed['Target'] = pd.qcut(
        df['pm2.5'].shift(-1),
        q=7,
        labels=False,
        duplicates='drop' # Handles cases where bin edges are not unique
    )
    print("Target attribute created and discretized.")

    # 4. --- Final Cleanup ---
    # Drop rows with NaN values resulting from the shift operations
    df_processed.dropna(inplace=True)
    
    # Convert target to integer type
    df_processed['Target'] = df_processed['Target'].astype(int)

    # Save the processed data
    df_processed.to_csv(processed_path)
    print(f"Preprocessing complete. Processed data saved to {processed_path}")
    print(f"Final dataset shape: {df_processed.shape}")
    print("\nFinal columns:")
    print(df_processed.columns.tolist())

if __name__ == '__main__':
    fetch_data()
    preprocess_data()
