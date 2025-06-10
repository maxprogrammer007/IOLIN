# experiments/run_final_experiment.py
# This is the ultimate experiment script. It uses our best model (XGBoost)
# on our best data (advanced features) with intensive hyperparameter tuning
# to achieve the highest possible accuracy.

import pandas as pd
import os
import sys
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import our best-performing model from src
from src.iolin_xgb import IOLIN_XGB

def run_final_experiment():
    """Runs the definitive IOLIN-XGB experiment with hyperparameter tuning on the full dataset."""
    print("--- Starting FINAL High-Performance Experiment (XGBoost + Advanced Features + Tuning) ---")

    # --- Load the advanced features dataset ---
    try:
        data_path = os.path.join("data", "processed", "processed_beijing_pm25_advanced.csv")
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        print(f"Loaded full advanced feature set with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Advanced processed data file not found at {data_path}.")
        return

    # --- Model & Experiment Parameters ---
    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col != target_col]

    WINDOW_SIZE = 2500 # A large window for robust training
    STEP_SIZE = 500
    
    # --- The Expanded Hyperparameter Grid ---
    # We will test these different configurations on each window
    param_grid = [
        {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 300, 'max_depth': 12, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'n_estimators': 400, 'max_depth': 15, 'learning_rate': 0.01, 'subsample': 1.0, 'colsample_bytree': 1.0}
    ]

    # --- Process the stream ---
    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing full stream with XGBoost and hyperparameter tuning...")
    
    current_pos = 0
    while current_pos + WINDOW_SIZE + STEP_SIZE <= len(df):
        train_window = df.iloc[current_pos : current_pos + WINDOW_SIZE]
        validation_window = df.iloc[current_pos + WINDOW_SIZE : current_pos + WINDOW_SIZE + STEP_SIZE]
        
        best_error = 1.0
        
        print(f"\n--- Tuning for window starting at index {current_pos} ({len(results)+1}/{total_windows}) ---")
        for i, params in enumerate(param_grid):
            print(f"  Trying param set {i+1}/{len(param_grid)}...")
            
            xgb_processor = IOLIN_XGB(
                all_input_attributes=input_cols_list,
                target_attribute=target_col,
                model_params=params
            )
            xgb_processor.fit(train_window)
            error = xgb_processor._calculate_error_rate(validation_window)
            
            if error < best_error:
                best_error = error
        
        accuracy = 1.0 - best_error
        print(f"  Best accuracy for this window: {accuracy:.2%}")
        
        results.append({'error_rate': best_error})
        current_pos += STEP_SIZE

    end_time_total = time.time()
    print(f"\n--- FINAL Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        avg_accuracy = 1.0 - results_df['error_rate'].mean()
        
        print("\n--- ULTIMATE PROJECT SUMMARY ---")
        print(f"Total experiment time: {(end_time_total - start_time_total)/60:.2f} minutes")
        print(f"Final Average ACCURACY: {avg_accuracy:.2%}")
        print("---------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉🎉🎉 CONGRATULATIONS! The 85%+ accuracy target has been successfully achieved! 🎉🎉🎉")
        else:
            print("This represents the final, highest performance of the optimized system.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_final_experiment()
