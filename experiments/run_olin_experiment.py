# experiments/run_olin_experiment.py
# This script will run the full OLIN experiment.
# It will load the processed data, instantiate the OLIN algorithm,
# run it on the data stream, and save the performance metrics
# to a CSV file in the /results directory.

import pandas as pd
import os
import sys
import time

# FIX: Add the project root's 'src' directory to the Python path
# This allows us to import modules from the 'src' folder.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

# Now we can import directly from the src modules
from olin import OLIN

def run_experiment():
    """Runs the full OLIN experiment and saves the results."""
    print("--- Starting Full OLIN Experiment ---")

    # Load the full processed dataset
    try:
        # Go up one level from 'experiments' to the project root to find 'data'
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_beijing_pm25.csv')
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {data_path}.")
        print("Please run src/data_loader.py first.")
        return

    # Define attributes and parameters
    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col not in [target_col, 'datetime', 'Unnamed: 0']]
    input_cols_with_uniques = {col: df[col].unique() for col in input_cols_list}

    WINDOW_SIZE = 500
    STEP_SIZE = 100
    SIGNIFICANCE_LEVEL = 0.05

    # Initialize OLIN processor
    olin_processor = OLIN(input_cols_with_uniques, target_col, significance_level=SIGNIFICANCE_LEVEL)

    # Process the stream and collect results
    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing data stream with window_size={WINDOW_SIZE} and step_size={STEP_SIZE}...")
    for i, result in enumerate(olin_processor.process_data_stream(df, WINDOW_SIZE, STEP_SIZE)):
        print(f"  Processing window {i+1}/{total_windows}...")
        results.append(result)

    end_time_total = time.time()
    print(f"\n--- OLIN Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        # Save results to CSV
        results_path = os.path.join(os.path.dirname(__file__), '..', 'results', "olin_metrics.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

        # Print final summary
        avg_error = results_df['error_rate'].mean()
        total_time = end_time_total - start_time_total
        avg_time_per_window = results_df['processing_time_s'].mean()

        print("\n--- OLIN Final Summary ---")
        print(f"Total windows processed: {len(results_df)}")
        print(f"Total experiment time: {total_time:.2f} seconds")
        print(f"Average Error Rate: {avg_error:.4f}")
        print(f"Average Processing Time per Window: {avg_time_per_window:.4f} seconds")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_experiment()
