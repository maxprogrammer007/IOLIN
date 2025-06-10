# src/data_loader.py
# This module is now updated to process the Household Electric Power Consumption dataset.
# It will download, clean, resample, and create advanced features for the new experiment.

import pandas as pd
import numpy as np
import os
import zipfile
import requests
from io import BytesIO

# --- Configuration for the NEW Dataset ---
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
RAW_DATA_PATH = os.path.join("data", "raw", "household_power_consumption.txt")
# We will save the new processed data to the same path as before, so the experiment script
# can run without modification.
PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_beijing_pm25_advanced.csv")

def fetch_power_data(url=DATASET_URL, raw_path=RAW_DATA_PATH):
    """Downloads and extracts the household power consumption dataset."""
    if os.path.exists(raw_path):
        print(f"Dataset already exists at {raw_path}")
        return

    print(f"Downloading dataset from {url}...")
    try:
        # Download the zip file
        response = requests.get(url)
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        # Extract the specific text file
        zip_file.extract('household_power_consumption.txt', path=os.path.join("data", "raw"))
        print(f"Dataset extracted and saved to {raw_path}")
    except Exception as e:
        print(f"Failed to download or extract dataset. Error: {e}")
        return None

def preprocess_power_data(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_DATA_PATH):
    """
    Loads the raw power data, cleans it, resamples to hourly, engineers features,
    and saves the processed data.
    """
    if not os.path.exists(raw_path):
        print(f"Raw data not found at {raw_path}. Please fetch the data first.")
        return

    print("--- Starting Power Consumption Data Preprocessing ---")
    
    # 1. --- Data Loading and Cleaning ---
    df = pd.read_csv(
        raw_path,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        infer_datetime_format=True,
        na_values=['?'],
        low_memory=False
    )
    df.set_index('datetime', inplace=True)
    
    # Convert all columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Handle missing values using forward fill
    df.fillna(method='ffill', inplace=True)
    print("Loaded and cleaned raw data.")

    # 2. --- Resample to Hourly Data ---
    # Resample the minute-by-minute data to hourly averages. This is crucial for performance.
    df_hourly = df.resample('H').mean()
    df_hourly.fillna(method='ffill', inplace=True)
    print(f"Resampled data to hourly. New shape: {df_hourly.shape}")

    # 3. --- Advanced Feature Engineering ---
    df_processed = pd.DataFrame(index=df_hourly.index)
    
    # Target is the Global_active_power for the next hour
    target_col = 'Global_active_power'
    
    # Rolling Window Statistics
    for window in [3, 6, 12, 24]:
        df_processed[f'{target_col}_roll_mean_{window}h'] = df_hourly[target_col].rolling(window=window).mean()
        df_processed[f'{target_col}_roll_std_{window}h'] = df_hourly[target_col].rolling(window=window).std()
    print("Rolling window features created.")

    # Lagged Features
    for lag in [1, 2, 3, 24, 168]:
        df_processed[f'{target_col}_lag_{lag}h'] = df_hourly[target_col].shift(lag)
    print("Lagged features created.")

    # Cyclical Time Features
    df_processed['hour_sin'] = np.sin(2 * np.pi * df_hourly.index.hour / 24)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df_hourly.index.hour / 24)
    df_processed['day_of_week_sin'] = np.sin(2 * np.pi * df_hourly.index.dayofweek / 7)
    df_processed['day_of_week_cos'] = np.cos(2 * np.pi * df_hourly.index.dayofweek / 7)
    print("Cyclical time features created.")
    
    # Other sensor features
    other_features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for feat in other_features:
        df_processed[feat] = df_hourly[feat]
        
    # 4. --- Target Attribute Creation ---
    # Discretize the target into 7 equal-frequency intervals
    df_processed['Target'] = pd.qcut(
        df_hourly[target_col].shift(-1),
        q=7,
        labels=False,
        duplicates='drop'
    )
    print("Target attribute created and discretized.")

    # 5. --- Final Cleanup ---
    df_processed.dropna(inplace=True)
    df_processed['Target'] = df_processed['Target'].astype(int)

    # Save the processed data
    df_processed.to_csv(processed_path)
    print(f"Preprocessing complete. New dataset saved to {processed_path}")
    print(f"Final dataset shape: {df_processed.shape}")

if __name__ == '__main__':
    fetch_power_data()
    preprocess_power_data()
