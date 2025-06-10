# src/iolin_sk.py
# The high-performance version of our ensemble model.
# This version uses a powerful Gradient Boosting Classifier as its engine
# to maximize accuracy.

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier

class IOLIN_SK_Boosted:
    """
    A high-accuracy online model that uses scikit-learn's
    GradientBoostingClassifier as its core learning engine.
    """
    def __init__(self, all_input_attributes, target_attribute, n_estimators=100, max_depth=5, learning_rate=0.1):
        self.all_input_attributes = all_input_attributes
        self.target_attribute = target_attribute
        
        # --- IMPROVEMENT: Store hyperparameters for the Gradient Boosting model ---
        self.model_params = {
            'n_estimators': n_estimators, # Number of sequential trees to build
            'max_depth': max_depth,         # Controls complexity of each tree
            'learning_rate': learning_rate, # Controls how much each tree contributes
            'subsample': 0.8                # Use 80% of data for training each tree to prevent overfitting
        }
        # The Gradient Boosting model itself
        self.model = None

    def fit(self, data):
        """Fits the Gradient Boosting model on the data."""
        X_train = data[self.all_input_attributes]
        y_train = data[self.target_attribute]

        # --- Train a new scikit-learn Gradient Boosting Classifier ---
        self.model = GradientBoostingClassifier(**self.model_params)
        self.model.fit(X_train, y_train)

    def predict(self, data):
        """Makes a prediction using the trained boosting model."""
        if self.model is None:
            return [None] * len(data)

        X_test = data[self.all_input_attributes]
        return self.model.predict(X_test)
        
    def _calculate_error_rate(self, data):
        """Calculates the classification error rate for the model."""
        if self.model is None or data.empty:
            return 1.0
        
        predictions = self.predict(data)
        actuals = data[self.target_attribute].tolist()
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return 1.0 - (correct / len(predictions)) if len(predictions) > 0 else 1.0

    def process_data_stream(self, data_stream, window_size, step_size):
        """
        Processes the data stream by rebuilding the powerful boosting model on each window.
        """
        num_records = len(data_stream)
        current_pos = 0

        while current_pos + window_size + step_size <= num_records:
            start_time = time.time()

            train_data = data_stream.iloc[current_pos : current_pos + window_size]
            validation_data = data_stream.iloc[current_pos + window_size : current_pos + window_size + step_size]

            if train_data.empty or validation_data.empty:
                break
            
            print(f"  Building Gradient Boosting model for window [{current_pos}-{current_pos + window_size}]...")
            self.fit(train_data)
            
            error = self._calculate_error_rate(validation_data)
            end_time = time.time()
            processing_time = end_time - start_time

            yield {
                'window_start': current_pos,
                'error_rate': error,
                'processing_time_s': processing_time,
            }

            current_pos += step_size
