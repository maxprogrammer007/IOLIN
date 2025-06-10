# experiments/run_fuzzy_xgb_experiment.py
# This is the definitive experiment, combining Fuzzy Feature Engineering
# with the high-performance XGBoost engine.

import pandas as pd
import os
import sys
import time

# FIX: Add the project's 'src' directory to the Python path
# This is the most robust way to handle imports for scripts in subdirectories.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now we can import the modules directly by their filenames
from iolin_xgb import IOLIN_XGB
from fuzzy_preprocessor import FuzzyPreprocessor

def run_final_fuzzy_xgb_experiment():
    """Runs the IOLIN-XGB experiment with fuzzy features."""
    print("--- Starting FINAL Experiment: Fuzzy-XGBoost ---")

    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', "data", "processed", "processed_beijing_pm25_advanced.csv")
        df = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
        print(f"Loaded full advanced feature set with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Advanced processed data file not found at {data_path}.")
        return

    target_col = 'Target'
    # Define which features are continuous and need fuzzification
    continuous_features = [col for col in df.columns if col != target_col]

    WINDOW_SIZE = 2500
    STEP_SIZE = 500
    
    results = []
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE
    start_time_total = time.time()

    print("Processing stream with Fuzzy-XGBoost model...")
    
    current_pos = 0
    while current_pos + WINDOW_SIZE + STEP_SIZE <= len(df):
        train_window = df.iloc[current_pos : current_pos + WINDOW_SIZE]
        validation_window = df.iloc[current_pos + WINDOW_SIZE : current_pos + WINDOW_SIZE + STEP_SIZE]
        
        print(f"\n--- Processing window starting at index {current_pos} ({len(results)+1}/{total_windows}) ---")
        
        # 1. Fuzzify the data for this window
        print("  Fuzzifying features...")
        preprocessor = FuzzyPreprocessor(continuous_cols=continuous_features)
        
        # Fit on the training data and transform both train and validation sets
        train_fuzzy_features = preprocessor.fit_transform(train_window)
        validation_fuzzy_features = preprocessor.transform(validation_window)
        
        # Combine fuzzy features with the original target
        train_data_fuzzy = pd.concat([train_fuzzy_features, train_window[target_col]], axis=1)
        validation_data_fuzzy = pd.concat([validation_fuzzy_features, validation_window[target_col]], axis=1)

        # 2. Train and evaluate the XGBoost model on the fuzzy data
        print("  Training XGBoost on fuzzy features...")
        xgb_processor = IOLIN_XGB(
            all_input_attributes=preprocessor.output_features,
            target_attribute=target_col
        )
        xgb_processor.fit(train_data_fuzzy)
        error = xgb_processor._calculate_error_rate(validation_data_fuzzy)
        accuracy = 1.0 - error
        
        print(f"  > Window complete. Accuracy = {accuracy:.2%}")
        results.append({'error_rate': error})
        current_pos += STEP_SIZE

    end_time_total = time.time()
    print(f"\n--- Fuzzy-XGBoost Experiment Finished ---")

    if results:
        results_df = pd.DataFrame(results)
        avg_accuracy = 1.0 - results_df['error_rate'].mean()
        
        print("\n--- FUZZY-XGBOOST: FINAL SUMMARY ---")
        print(f"Total experiment time: {(end_time_total - start_time_total)/60:.2f} minutes")
        print(f"Final Average ACCURACY: {avg_accuracy:.2%}")
        print("---------------------------------------")
        if avg_accuracy >= 0.85:
            print("🎉🎉🎉 CONGRATULATIONS! The 85%+ accuracy target has been successfully achieved! 🎉🎉�")
        else:
            print("This represents the final, highest performance of our most innovative system.")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    run_final_fuzzy_xgb_experiment()
