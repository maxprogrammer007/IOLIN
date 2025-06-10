# experiments/run_fuzzy_forest_experiment.py
# This is the ultimate experiment script, testing our innovative
# Fuzzy Random Forest model.

import pandas as pd
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.fuzzy_random_forest import FuzzyRandomForest

def run_final_fuzzy_experiment():
    """Runs the full Fuzzy Random Forest experiment."""
    print("--- Starting Final Experiment: Fuzzy Random Forest ---")

    try:
        data_path = os.path.join("data", "processed", "processed_beijing_pm25_advanced.csv")
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # Use a significant slice for this intensive final test
        df = df.head(10000)
        print(f"Loaded advanced feature set with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Advanced processed data file not found at {data_path}.")
        return

    target_col = 'Target'
    input_cols = [col for col in df.columns if col != target_col]

    WINDOW_SIZE = 2500
    STEP_SIZE = 500
    N_ESTIMATORS = 10 # Start with 10 trees for a manageable experiment time

    # Initialize the Fuzzy Random Forest processor
    fuzzy_forest_processor = FuzzyRandomForest(
        n_estimators=N_ESTIMATORS
    )

    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing stream with Fuzzy Random Forest ({N_ESTIMATORS} trees per window)...")
    for i, result in enumerate(fuzzy_forest_processor.process_data_stream(df, input_cols, target_col, WINDOW_SIZE, STEP_SIZE)):
        accuracy = 1.0 - result['error_rate']
        print(f"  > Window {i+1}/{total_windows} complete. Accuracy = {accuracy:.2%}, Time = {result['processing_time_s']:.2f}s")
        results.append(result)

    end_time_total = time.time()
    print(f"\n--- Fuzzy Random Forest Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        avg_accuracy = 1.0 - results_df['error_rate'].mean()
        
        print("\n--- FUZZY RANDOM FOREST: FINAL SUMMARY ---")
        print(f"Total experiment time: {(end_time_total - start_time_total)/60:.2f} minutes")
        print(f"Final Average ACCURACY: {avg_accuracy:.2%}")
        print("------------------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉🎉🎉 CONGRATULATIONS! The 85%+ accuracy target has been successfully achieved! 🎉🎉🎉")
        else:
            print("This represents the final performance of our innovative Fuzzy Information Network ensemble.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_final_fuzzy_experiment()
