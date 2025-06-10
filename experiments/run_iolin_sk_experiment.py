# experiments/run_iolin_sk_experiment.py
# This script runs the high-performance IOLIN-SK model to test its accuracy.

import pandas as pd
import os
import sys
import time

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from iolin_sk import IOLIN_SK_Forest

def run_sk_experiment():
    """Runs the full IOLIN-SK experiment and prints the summary."""
    print("--- Starting IOLIN-SK High-Performance Experiment (Improved) ---")

    # Load the full processed dataset
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_beijing_pm25.csv')
        df = pd.read_csv(data_path)
        # --- IMPROVEMENT 1: Use more data for a more stable accuracy measure ---
        df = df.head(10000) 
        print(f"Loaded and using the first {len(df)} records for the experiment.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {data_path}.")
        return

    # --- Model & Experiment Parameters ---
    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col not in [target_col, 'datetime', 'Unnamed: 0']]

    WINDOW_SIZE = 500
    STEP_SIZE = 100
    # --- IMPROVEMENT 2: Increase the number of trees for a more powerful ensemble ---
    N_ESTIMATORS = 100 
    
    # Initialize IOLIN-SK processor
    forest_processor = IOLIN_SK_Forest(
        all_input_attributes=input_cols_list,
        target_attribute=target_col,
        n_estimators=N_ESTIMATORS
    )

    # --- Process the stream ---
    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing stream with {N_ESTIMATORS} scikit-learn trees per window...")
    for i, result in enumerate(forest_processor.process_data_stream(df, WINDOW_SIZE, STEP_SIZE)):
        accuracy = 1.0 - result['error_rate']
        print(f"  Processed window {i+1}/{total_windows}: Accuracy = {accuracy:.2%}")
        results.append(result)

    end_time_total = time.time()
    print(f"\n--- IOLIN-SK Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        
        avg_error = results_df['error_rate'].mean()
        avg_accuracy = 1.0 - avg_error
        
        print("\n--- IOLIN-SK Final Summary ---")
        print(f"Total experiment time: {end_time_total - start_time_total:.2f} seconds")
        print(f"Average Error Rate: {avg_error:.4f}")
        print(f"Average ACCURACY: {avg_accuracy:.2%}")
        print("--------------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉 SUCCESS: Model performance target of 85%+ accuracy has been achieved!")
        else:
            print("Performance has significantly improved but may not have reached the 85% target.")
            print("Suggestions for further improvement: Increase N_ESTIMATORS, or optimize DecisionTreeClassifier hyperparameters.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_sk_experiment()
