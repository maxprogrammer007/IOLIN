# experiments/run_iolin_forest_experiment.py
# This script runs the high-accuracy IOLIN-Forest model to test its performance.

import pandas as pd
import os
import sys
import time

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

# Import our new, improved model
from iolin_forest import IOLIN_Forest

def run_forest_experiment():
    """Runs the full IOLIN-Forest experiment and saves the results."""
    print("--- Starting IOLIN-Forest High-Accuracy Experiment ---")

    # Load the full processed dataset
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_beijing_pm25.csv')
        df = pd.read_csv(data_path)
        # For a quicker test of the improvisation, let's use a smaller slice of the data
        # We will use the first 5000 records.
        df = df.head(5000)
        print(f"Loaded and using the first {len(df)} records for the experiment.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {data_path}.")
        return

    # --- Model & Experiment Parameters ---
    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col not in [target_col, 'datetime', 'Unnamed: 0']]
    input_cols_with_uniques = {col: df[col].unique() for col in input_cols_list}

    WINDOW_SIZE = 500
    STEP_SIZE = 100
    N_ESTIMATORS = 20 # Number of trees in the forest. Increase for higher accuracy.
    MAX_FEATURES = 'sqrt' # Number of features per tree. 'sqrt' is a common default.
    SIGNIFICANCE_LEVEL = 0.05

    # Initialize IOLIN-Forest processor
    forest_processor = IOLIN_Forest(
        input_attributes_with_uniques=input_cols_with_uniques,
        target_attribute=target_col,
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        significance_level=SIGNIFICANCE_LEVEL
    )

    # --- Process the stream ---
    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print(f"Processing stream with {N_ESTIMATORS} trees per window...")
    for i, result in enumerate(forest_processor.process_data_stream(df, WINDOW_SIZE, STEP_SIZE)):
        print(f"  Processed window {i+1}/{total_windows}: Error Rate = {result['error_rate']:.4f}")
        results.append(result)

    end_time_total = time.time()
    print(f"\n--- IOLIN-Forest Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        # Save results to a new CSV file
        results_path = os.path.join(os.path.dirname(__file__), '..', 'results', "iolin_forest_metrics.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

        # --- Final Summary ---
        avg_error = results_df['error_rate'].mean()
        avg_accuracy = 1.0 - avg_error
        
        print("\n--- IOLIN-Forest Final Summary ---")
        print(f"Total experiment time: {end_time_total - start_time_total:.2f} seconds")
        print(f"Average Error Rate: {avg_error:.4f}")
        print(f"Average ACCURACY: {avg_accuracy:.2%}")
        print("--------------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉 SUCCESS: Model performance target of 85%+ accuracy has been achieved!")
        else:
            print("Further improvements may be needed to reach the 85% target.")
            print("Suggestions: Increase N_ESTIMATORS in the script, or perform more feature engineering.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_forest_experiment()
    print("Running IOLIN-Forest experiment...")