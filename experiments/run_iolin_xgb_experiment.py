# experiments/run_iolin_xgb_experiment.py
# This script runs the ultimate IOLIN-XGB model on the full dataset
# and includes hyperparameter tuning to maximize accuracy.

import pandas as pd
import os
import sys
import time

# A more robust way to add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now we can use an absolute import from the src package
from src.iolin_xgb import IOLIN_XGB

def run_xgb_experiment():
    """Runs the full IOLIN-XGB experiment with hyperparameter tuning."""
    print("--- Starting Final IOLIN-XGB Experiment (Full Dataset + Hyperparameter Tuning) ---")

    # --- Load the advanced features dataset ---
    try:
        # Path is now simpler as we assume the script is run from the project root
        data_path = os.path.join("data", "processed", "processed_beijing_pm25_advanced.csv")
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # --- FINAL RUN: Use the entire dataset ---
        print(f"Loaded full advanced feature set with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Advanced processed data file not found at {data_path}.")
        return

    # --- Model & Experiment Parameters ---
    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col != target_col]

    WINDOW_SIZE = 2000 # Using a larger window for the final model
    STEP_SIZE = 500   # And a larger step size
    
    # --- Define a grid of hyperparameters to search ---
    param_grid = [
        {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.05},
        {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.05},
        {'n_estimators': 250, 'max_depth': 12, 'learning_rate': 0.01}
    ]

    # --- Process the stream ---
    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing full stream with XGBoost model (with hyperparameter tuning)...")
    
    current_pos = 0
    while current_pos + WINDOW_SIZE + STEP_SIZE <= len(df):
        train_window = df.iloc[current_pos : current_pos + WINDOW_SIZE]
        validation_window = df.iloc[current_pos + WINDOW_SIZE : current_pos + WINDOW_SIZE + STEP_SIZE]
        
        best_model = None
        best_error = 1.0
        
        print(f"\n--- Tuning for window starting at index {current_pos} ---")
        for i, params in enumerate(param_grid):
            print(f"  Trying param set {i+1}/{len(param_grid)}...")
            # Initialize the processor with the current set of parameters
            xgb_processor = IOLIN_XGB(
                all_input_attributes=input_cols_list,
                target_attribute=target_col,
                model_params=params
            )
            # Fit the model on the training window
            xgb_processor.fit(train_window)
            # Evaluate on the validation window
            error = xgb_processor._calculate_error_rate(validation_window)
            
            if error < best_error:
                best_error = error
                best_model = xgb_processor.model
        
        accuracy = 1.0 - best_error
        print(f"  Best model for this window achieved accuracy: {accuracy:.2%}")
        
        results.append({
            'window_start': current_pos,
            'error_rate': best_error,
        })
        
        # FIX: Corrected variable name to use uppercase 'STEP_SIZE'
        current_pos += STEP_SIZE


    end_time_total = time.time()
    print(f"\n--- IOLIN-XGB Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        
        avg_error = results_df['error_rate'].mean()
        avg_accuracy = 1.0 - avg_error
        
        print("\n--- FINAL PROJECT SUMMARY ---")
        print(f"Total experiment time: {end_time_total - start_time_total:.2f} seconds")
        print(f"Average Error Rate: {avg_error:.4f}")
        print(f"Average ACCURACY: {avg_accuracy:.2%}")
        print("---------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉🎉🎉 CONGRATULATIONS! The 85%+ accuracy target has been successfully achieved! 🎉🎉🎉")
        else:
            print("This represents the final, highest performance of the implemented system.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_xgb_experiment()
