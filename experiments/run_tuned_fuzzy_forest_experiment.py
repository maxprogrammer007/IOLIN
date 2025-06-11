# experiments/run_tuned_fuzzy_forest_experiment.py
# This is the ultimate experiment script, testing our tuned and improved
# custom-built Fuzzy Random Forest model using K-Fold Cross-Validation.

import pandas as pd
import os
import sys
import time
import numpy as np
from sklearn.model_selection import KFold

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.fuzzy_random_forest import FuzzyRandomForest

def run_ultimate_custom_experiment():
    """Runs the tuned Fuzzy Random Forest experiment with K-Fold Cross-Validation."""
    print("--- Starting Ultimate Experiment: Tuned Fuzzy Random Forest with Cross-Validation ---")

    try:
        data_path = os.path.join("data", "processed", "processed_wine_quality.csv")
        df = pd.read_csv(data_path)
        print(f"Loaded Wine Quality dataset with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {data_path}.")
        return

    target_col = 'Target'
    input_cols = [col for col in df.columns if col != target_col]
    
    # --- ADVANCED TUNED Hyperparameters ---
    N_ESTIMATORS = 100 # More trees for a more robust forest
    MAX_DEPTH = 20     # Allow trees to grow deeper to find more patterns
    N_SPLITS = 5       # Number of folds for cross-validation

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accuracies = []
    fold = 1
    total_start_time = time.time()

    print(f"\nStarting {N_SPLITS}-Fold Cross-Validation...")
    print(f"Model parameters: N_ESTIMATORS={N_ESTIMATORS}, MAX_DEPTH={MAX_DEPTH}")

    for train_index, test_index in kf.split(df):
        print(f"\n--- Processing Fold {fold}/{N_SPLITS} ---")
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        # Initialize the Fuzzy Random Forest processor for this fold
        fuzzy_forest_processor = FuzzyRandomForest(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH
        )

        fold_start_time = time.time()

        print(f"Training Tuned Fuzzy Random Forest on {len(train_data)} samples...")
        fuzzy_forest_processor.fit(train_data, input_cols, target_col)
        
        print("Evaluating model performance on the test set ({len(test_data)} samples)...")
        error_rate = fuzzy_forest_processor._calculate_error_rate(test_data)
        accuracy = 1.0 - error_rate
        fold_accuracies.append(accuracy)

        fold_end_time = time.time()
        print(f"> Fold {fold} complete. Accuracy: {accuracy:.2%}, Time: {(fold_end_time - fold_start_time)/60:.2f} minutes")
        fold += 1

    total_end_time = time.time()
    print(f"\n--- Cross-Validation Experiment Finished ---")

    print("\n--- TUNED FUZZY RANDOM FOREST: FINAL SUMMARY ---")
    print(f"Total experiment time: {(total_end_time - total_start_time)/60:.2f} minutes")
    print(f"Accuracies per fold: {[f'{acc:.2%}' for acc in fold_accuracies]}")
    print(f"Final Average ACCURACY (over {N_SPLITS} folds): {np.mean(fold_accuracies):.2%}")
    print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies):.4f}")
    print("-------------------------------------------------")
    if np.mean(fold_accuracies) >= 0.85:
        print("🎉🎉🎉 CONGRATULATIONS! The 85%+ accuracy target has been successfully achieved with our custom model! 🎉🎉🎉")
    else:
        print("This represents the final, most robust performance measurement of our custom model.")


if __name__ == '__main__':
    run_ultimate_custom_experiment()
