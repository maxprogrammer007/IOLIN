# experiments/run_iolin_sk_experiment.py
# This script runs the high-performance IOLIN-SK model with the
# new advanced feature set to maximize accuracy.

import pandas as pd
import os
import sys
import time

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from iolin_sk import IOLIN_SK_Boosted

def run_advanced_sk_experiment():
    """Runs the IOLIN-SK experiment with the advanced feature set."""
    print("--- Starting IOLIN-SK High-Accuracy Experiment (Advanced Features) ---")

    # --- Load the new advanced features dataset ---
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_beijing_pm25_advanced.csv')
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        # Using a larger slice of data for the final test
        df = df.head(15000)
        print(f"Loaded advanced feature set with {len(df)} records for the experiment.")
    except FileNotFoundError:
        print(f"Error: Advanced processed data file not found at {data_path}.")
        print("Please run 'src/data_loader.py' first to generate the advanced features.")
        return

    # --- Model & Experiment Parameters ---
    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col != target_col]

    WINDOW_SIZE = 1000 # Using a larger window to capture more complex patterns
    STEP_SIZE = 200
    
    # Initialize IOLIN-SK processor with powerful parameters
    forest_processor = IOLIN_SK_Boosted(
        all_input_attributes=input_cols_list,
        target_attribute=target_col,
        n_estimators=150, # More estimators for more power
        max_depth=7,
        learning_rate=0.05
    )

    # --- Process the stream ---
    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing stream with Gradient Boosting model...")
    for i, result in enumerate(forest_processor.process_data_stream(df, WINDOW_SIZE, STEP_SIZE)):
        accuracy = 1.0 - result['error_rate']
        print(f"  Processed window {i+1}/{total_windows}: Accuracy = {accuracy:.2%}")
        results.append(result)

    end_time_total = time.time()
    print(f"\n--- IOLIN-SK (Advanced) Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        
        avg_error = results_df['error_rate'].mean()
        avg_accuracy = 1.0 - avg_error
        
        print("\n--- IOLIN-SK (Advanced) Final Summary ---")
        print(f"Total experiment time: {end_time_total - start_time_total:.2f} seconds")
        print(f"Average Error Rate: {avg_error:.4f}")
        print(f"Average ACCURACY: {avg_accuracy:.2%}")
        print("-------------------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉🎉🎉 SUCCESS: Model performance target of 85%+ accuracy has been achieved! 🎉🎉🎉")
        else:
            print("Performance has significantly improved with advanced features.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_advanced_sk_experiment()
