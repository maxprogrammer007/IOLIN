# experiments/run_improved_in_experiment.py
# This script tests the performance of the IOLIN framework when powered by
# our new, improved Greedy Information Network engine.

import pandas as pd
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# We use the existing IOLIN framework but it will now import the improved IN model
from src.iolin import IOLIN 

def run_improved_experiment():
    """Runs the IOLIN experiment with the improved IN model."""
    print("--- Starting Experiment with Improved Greedy IN Engine ---")

    try:
        data_path = os.path.join("data", "processed", "processed_beijing_pm25_advanced.csv")
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # Use a significant slice of data for the test
        df = df.head(10000)
        print(f"Loaded advanced feature set with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Advanced processed data file not found at {data_path}.")
        return

    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col != target_col]
    input_cols_with_uniques = {col: df[col].unique() for col in input_cols_list}

    WINDOW_SIZE = 2500
    STEP_SIZE = 500
    
    # Initialize the IOLIN processor, which will now use our improved IN model
    iolin_processor = IOLIN(input_cols_with_uniques, target_col, significance_level=0.05)

    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing stream with Improved Greedy IN model...")
    for i, result in enumerate(iolin_processor.process_data_stream(df, WINDOW_SIZE, STEP_SIZE)):
        accuracy = 1.0 - result['final_error_rate']
        print(f"  Processed window {i+1}/{total_windows}: Accuracy = {accuracy:.2%}, Action = {result['action']}")
        results.append(result)

    end_time_total = time.time()
    print(f"\n--- Improved IN Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        avg_accuracy = 1.0 - results_df['final_error_rate'].mean()
        
        print("\n--- Improved IN Final Summary ---")
        print(f"Total experiment time: {(end_time_total - start_time_total)/60:.2f} minutes")
        print(f"Final Average ACCURACY: {avg_accuracy:.2%}")
        print("---------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉🎉🎉 CONGRATULATIONS! The 85%+ accuracy target has been successfully achieved! 🎉🎉🎉")
        else:
            print("This represents the performance of the system with the improved internal model structure.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_improved_experiment()
