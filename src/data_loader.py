# src/data_loader.py
# This module will handle loading, cleaning, and preprocessing the data.
# This version includes advanced feature engineering for high-performance models.

import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
RAW_DATA_PATH = os.path.join("data", "raw", "beijing_pm25.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_beijing_pm25_advanced.csv") # New file name

def fetch_data(url=DATASET_URL, path=RAW_DATA_PATH):
    """Downloads the dataset if it doesn't already exist."""
    if os.path.exists(path):
        print(f"Dataset already exists at {path}")
        return
    print(f"Downloading dataset from {url}...")
    try:
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Dataset saved to {path}")
    except Exception as e:
        print(f"Failed to download dataset. Error: {e}")
        return None

def preprocess_data_advanced(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_DATA_PATH):
    """
    Loads raw data, cleans it, and engineers advanced time-series features.
    """
    if not os.path.exists(raw_path):
        print(f"Raw data not found at {raw_path}. Please fetch the data first.")
        return

    print("Starting ADVANCED data preprocessing...")
    df = pd.read_csv(raw_path)

    # 1. --- Data Cleaning ---
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('datetime', inplace=True)
    df['pm2.5'].fillna(method='ffill', inplace=True)
    df['pm2.5'].fillna(method='bfill', inplace=True)
    print("Missing values in 'pm2.5' handled.")

    # 2. --- Advanced Feature Engineering ---
    df_processed = pd.DataFrame(index=df.index)
    
    # Keep some original features
    df_processed['Dew_Point'] = df['DEWP']
    df_processed['Temperature'] = df['TEMP']
    df_processed['Pressure'] = df['PRES']
    df_processed['Wind_Speed'] = df['Iws']

    # --- Feature: Rolling Window Statistics ---
    # Gives the model a sense of recent trends and volatility
    for window in [3, 6, 12, 24]: # Last 3, 6, 12, 24 hours
        df_processed[f'pm2.5_roll_mean_{window}h'] = df['pm2.5'].rolling(window=window).mean()
        df_processed[f'pm2.5_roll_std_{window}h'] = df['pm2.5'].rolling(window=window).std()
        df_processed[f'pm2.5_roll_min_{window}h'] = df['pm2.5'].rolling(window=window).min()
        df_processed[f'pm2.5_roll_max_{window}h'] = df['pm2.5'].rolling(window=window).max()
    print("Rolling window features created.")

    # --- Feature: Lagged Features ---
    for lag in [1, 2, 3, 24, 48, 168]: # 1h, 2h, 3h, 1d, 2d, 1w
        df_processed[f'pm2.5_lag_{lag}h'] = df['pm2.5'].shift(lag)
    print("Lagged features created.")

    # --- Feature: Cyclical Time Features ---
    # Helps the model understand the cyclical nature of time
    df_processed['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df_processed['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df_processed['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df_processed['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df_processed['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    print("Cyclical time features created.")
    
    # 3. --- Target Attribute Creation ---
    df_processed['Target'] = pd.qcut(
        df['pm2.5'].shift(-1),
        q=7,
        labels=False,
        duplicates='drop'
    )
    print("Target attribute created and discretized.")

    # 4. --- Final Cleanup ---
    df_processed.dropna(inplace=True)
    df_processed['Target'] = df_processed['Target'].astype(int)

    # Save the processed data
    df_processed.to_csv(processed_path)
    print(f"Preprocessing complete. Advanced features saved to {processed_path}")
    print(f"Final dataset shape: {df_processed.shape}")

if __name__ == '__main__':
    fetch_data()
    preprocess_data_advanced()
