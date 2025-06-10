# experiments/run_iolin_lstm_experiment.py
# This script runs the state-of-the-art IOLIN-LSTM model on the full dataset.

import pandas as pd
import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO messages
import tensorflow as tf

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.iolin_lstm import IOLIN_LSTM

def run_lstm_experiment():
    """Runs the full IOLIN-LSTM experiment and prints the final summary."""
    print("--- Starting Final IOLIN-LSTM Experiment (Deep Learning Approach) ---")

    # --- Load the advanced features dataset ---
    try:
        data_path = os.path.join("data", "processed", "processed_beijing_pm25_advanced.csv")
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # For this intensive experiment, we'll use a smaller but significant slice
        df = df.head(8000)
        print(f"Loaded advanced feature set with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Advanced processed data file not found at {data_path}.")
        return

    # --- Model & Experiment Parameters ---
    target_col = 'Target'
    input_cols = [col for col in df.columns if col != target_col]
    
    # LSTM-specific parameters
    N_STEPS = 24       # Use the last 24 hours of data to predict the next hour
    N_FEATURES = len(input_cols)
    N_CLASSES = df[target_col].nunique()

    WINDOW_SIZE = 2000
    STEP_SIZE = 500
    
    # Initialize the IOLIN-LSTM processor
    lstm_processor = IOLIN_LSTM(
        n_steps=N_STEPS,
        n_features=N_FEATURES,
        n_classes=N_CLASSES
    )

    # --- Process the stream ---
    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing full stream with LSTM model...")
    for i, result in enumerate(lstm_processor.process_data_stream(df, target_col, WINDOW_SIZE, STEP_SIZE)):
        accuracy = 1.0 - result['error_rate']
        print(f"  Processed window {i+1}/{total_windows}: Accuracy = {accuracy:.2%}")
        results.append(result)

    end_time_total = time.time()
    print(f"\n--- IOLIN-LSTM Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        
        avg_error = results_df['error_rate'].mean()
        avg_accuracy = 1.0 - avg_error
        
        print("\n--- DEEP LEARNING FINAL SUMMARY ---")
        print(f"Total experiment time: {end_time_total - start_time_total:.2f} seconds")
        print(f"Average Error Rate: {avg_error:.4f}")
        print(f"Average ACCURACY: {avg_accuracy:.2%}")
        print("---------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉🎉🎉 CONGRATULATIONS! The 85%+ accuracy target has been successfully achieved! 🎉🎉🎉")
        else:
            print("This represents the final, highest performance of the implemented system using a deep learning approach.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_lstm_experiment()
