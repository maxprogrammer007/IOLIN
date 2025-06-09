# src/iolin.py
# This module will implement the main IOLIN algorithm.
# Key components:
# - Drift detection logic (using Eq. 4 and 5 from the paper).
# - `Update_Current_Network` function containing the three update strategies:
#   1. Check_Split_Validity
#   2. Replace_Last_Layer
#   3. New_Split_Process

import pandas as pd
import numpy as np
import os
import time
from information_network import InformationNetwork
from scipy.stats import norm

class IOLIN:
    """
    The Incremental On-line Information Network (IOLIN) algorithm.
    This version updates an existing model if no concept drift is detected,
    and rebuilds it otherwise.
    """
    def __init__(self, input_attributes_with_uniques, target_attribute, significance_level=0.01):
        self.input_attributes_with_uniques = input_attributes_with_uniques
        self.target_attribute = target_attribute
        self.significance_level = significance_level
        self.model = None # The current InformationNetwork model

    def _calculate_error_rate(self, data):
        """Calculates the classification error rate of the current model on given data."""
        if self.model is None or data.empty:
            return 1.0 # Max error if no model or data
        
        predictions = self.model.predict(data)
        actuals = data[self.target_attribute].tolist()
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return 1.0 - (correct / len(predictions))

    def _detect_concept_drift(self, e_tr, e_val, w_size, v_size):
        """
        Detects concept drift based on a statistically significant increase
        in the validation error rate compared to the training error rate.
        Uses Eq. (4) and (5) from the paper.
        """
        if w_size == 0 or v_size == 0:
            return True # Assume drift if windows are empty

        # Eq. (4): Variance of the difference between error rates
        var_diff = (e_tr * (1 - e_tr) / w_size) + (e_val * (1 - e_val) / v_size)
        if var_diff < 0: return True # Should not happen, but as a safeguard

        # Eq. (5): Maximum expected difference at 99% confidence level
        z_99 = norm.ppf(0.99) # Z-score for 99% confidence is approx 2.326
        max_diff = z_99 * np.sqrt(var_diff)
        
        actual_diff = e_val - e_tr

        # If the actual increase in error is greater than the max expected difference, a drift is detected.
        return actual_diff > max_diff

    def _update_current_network(self, train_data):
        """
        Placeholder for the three incremental update operations.
        For now, this is a fast operation that does nothing.
        """
        # TODO: Implement the three update strategies from the paper:
        # 1. Check_Split_Validity
        # 2. Replace_Last_Layer
        # 3. New_Split_Process
        time.sleep(0.1) # Simulate a very fast update process
        return "Updated"

    def process_data_stream(self, data_stream, window_size, step_size):
        """
        Processes the data stream using a sliding window, updating or rebuilding the model.
        """
        num_records = len(data_stream)
        current_pos = 0

        # Initial model creation
        print("Building initial model...")
        initial_train_data = data_stream.iloc[current_pos : current_pos + window_size]
        self.model = InformationNetwork(significance_level=self.significance_level)
        self.model.fit(initial_train_data, self.input_attributes_with_uniques, self.target_attribute)
        current_pos += window_size

        while current_pos + step_size <= num_records:
            start_time = time.time()

            # Define the current training window (for error calculation) and validation window
            # The training window for error calculation is the *previous* validation window
            train_error_window = data_stream.iloc[current_pos - step_size : current_pos]
            validation_window = data_stream.iloc[current_pos : current_pos + step_size]

            if train_error_window.empty or validation_window.empty:
                break
                
            # --- Drift Detection Step ---
            e_tr = self._calculate_error_rate(train_error_window)
            e_val = self._calculate_error_rate(validation_window)
            
            is_drift = self._detect_concept_drift(e_tr, e_val, len(train_error_window), len(validation_window))
            
            action_taken = ""
            if is_drift:
                # --- Rebuild Step (Major Concept Drift) ---
                action_taken = "Rebuilt"
                # The new training window is the validation window where drift was detected
                new_train_data = validation_window
                self.model = InformationNetwork(significance_level=self.significance_level)
                self.model.fit(new_train_data, self.input_attributes_with_uniques, self.target_attribute)
            else:
                # --- Update Step (Concept Stable) ---
                action_taken = self._update_current_network(validation_window)


            # --- Evaluate and Yield Results ---
            final_error = self._calculate_error_rate(validation_window)
            end_time = time.time()
            processing_time = end_time - start_time
            classification_rate = len(validation_window) / processing_time if processing_time > 0 else float('inf')

            yield {
                'window_start': current_pos,
                'window_end': current_pos + step_size,
                'action': action_taken,
                'e_tr': e_tr,
                'e_val': e_val,
                'is_drift': is_drift,
                'final_error_rate': final_error,
                'processing_time_s': processing_time,
                'classification_rate_rec_s': classification_rate,
            }

            current_pos += step_size


if __name__ == '__main__':
    print("Running a test on the IOLIN implementation...")

    try:
        df = pd.read_csv(os.path.join("data", "processed", "processed_beijing_pm25.csv"))
    except FileNotFoundError:
        print("Error: Processed data file not found. Please run src/data_loader.py first.")
        exit()

    test_stream_data = df.head(2000)

    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col not in [target_col, 'datetime', 'Unnamed: 0']]
    input_cols_with_uniques = {col: df[col].unique() for col in input_cols_list}

    WINDOW_SIZE = 500
    STEP_SIZE = 100

    iolin_processor = IOLIN(input_cols_with_uniques, target_col, significance_level=0.05)

    results = []
    print(f"Processing data stream with window_size={WINDOW_SIZE} and step_size={STEP_SIZE}...")
    for result in iolin_processor.process_data_stream(test_stream_data, WINDOW_SIZE, STEP_SIZE):
        status = "Drift Detected" if result['is_drift'] else "Stable"
        print(f"  Window [{result['window_start']}-{result['window_end']}]: "
              f"Action={result['action']}, Status={status}, "
              f"Final Error={result['final_error_rate']:.3f}, "
              f"Time={result['processing_time_s']:.3f}s")
        results.append(result)

    if results:
        results_df = pd.DataFrame(results)
        avg_error = results_df['final_error_rate'].mean()
        avg_time = results_df['processing_time_s'].mean()
        rebuilds = results_df[results_df['action'] == 'Rebuilt'].shape[0]

        print("\n--- IOLIN Test Summary ---")
        print(f"Total windows processed: {len(results_df)}")
        print(f"Number of rebuilds (drifts): {rebuilds}")
        print(f"Average Error Rate: {avg_error:.3f}")
        print(f"Average Processing Time per Window: {avg_time:.3f} seconds")
        print("\nNote: Update functions are currently placeholders. Time will be much faster than OLIN.")
    else:
        print("No results generated. The test data stream might be too small for the window settings.")
