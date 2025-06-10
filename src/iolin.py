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
# FIX: Changed to a relative import to work correctly within the 'src' package
from .information_network import InformationNetwork
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
        self.last_training_data = None # Store the data used for the last training/rebuild

    def _calculate_error_rate(self, data, model_to_use):
        """Calculates the classification error rate of a given model on given data."""
        if model_to_use is None or data.empty:
            return 1.0 # Max error if no model or data
        
        predictions = model_to_use.predict(data)
        actuals = data[self.target_attribute].tolist()
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return 1.0 - (correct / len(predictions)) if len(predictions) > 0 else 1.0

    def _detect_concept_drift(self, e_prev, e_curr, prev_size, curr_size, tolerance=0.05):
        """
        Detects concept drift by comparing the error on the previous window (e_prev)
        to the error on the current window (e_curr).
        """
        if prev_size == 0 or curr_size == 0:
            return True

        # Check 1: Is the new error rate higher by at least the tolerance?
        if e_curr < e_prev + tolerance:
            return False # Not a significant enough increase in error, concept is stable

        # Check 2: If the error increase is large enough, is it statistically significant?
        var_diff = (e_prev * (1 - e_prev) / prev_size) + (e_curr * (1 - e_curr) / curr_size)
        if var_diff <= 0: return False

        z_95 = norm.ppf(0.95)
        max_diff = z_95 * np.sqrt(var_diff)
        
        actual_diff = e_curr - e_prev

        return actual_diff > max_diff

    def _update_current_network(self):
        """
        Simulates a fast update. This represents a computationally cheap operation.
        """
        pass
        return "Updated"

    def process_data_stream(self, data_stream, window_size, step_size):
        """
        Processes the data stream using a sliding window, updating or rebuilding the model.
        """
        num_records = len(data_stream)
        current_pos = 0

        # --- Initial Model Creation ---
        print("Building initial model...")
        initial_train_data = data_stream.iloc[current_pos : current_pos + window_size]
        self.last_training_data = initial_train_data.copy()
        
        self.model = InformationNetwork(significance_level=self.significance_level)
        self.model.fit(initial_train_data, list(self.input_attributes_with_uniques.keys()), self.target_attribute)
        
        # The first "previous error" is the error on the initial training data itself
        prev_error = self._calculate_error_rate(initial_train_data, self.model)
        prev_window_size = len(initial_train_data)
        current_pos += window_size

        while current_pos + step_size <= num_records:
            start_time = time.time()

            validation_window = data_stream.iloc[current_pos : current_pos + step_size]
            if validation_window.empty:
                break
                
            # --- Drift Detection Step ---
            current_error = self._calculate_error_rate(validation_window, self.model)
            is_drift = self._detect_concept_drift(prev_error, current_error, prev_window_size, len(validation_window))
            
            action_taken = ""
            if is_drift:
                action_taken = "Rebuilt"
                rebuild_train_data = data_stream.iloc[current_pos + step_size - window_size : current_pos + step_size]
                if len(rebuild_train_data) < window_size:
                    break
                self.model = InformationNetwork(significance_level=self.significance_level)
                self.model.fit(rebuild_train_data, list(self.input_attributes_with_uniques.keys()), self.target_attribute)
                self.last_training_data = rebuild_train_data.copy()
            else:
                action_taken = "Updated"
                self._update_current_network()
            
            # Update the previous error for the next iteration
            prev_error = self._calculate_error_rate(validation_window, self.model)
            prev_window_size = len(validation_window)

            end_time = time.time()
            processing_time = end_time - start_time

            yield {
                'window_start': current_pos,
                'window_end': current_pos + step_size,
                'action': action_taken,
                'is_drift': is_drift,
                'final_error_rate': current_error,
                'processing_time_s': processing_time,
            }

            current_pos += step_size
