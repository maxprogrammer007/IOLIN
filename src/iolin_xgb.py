# src/iolin_xgb.py
# The ultimate high-performance model using XGBoost as its engine.

import pandas as pd
import numpy as np
import time
import xgboost as xgb

class IOLIN_XGB:
    """
    A state-of-the-art online model using XGBoost (Extreme Gradient Boosting)
    as its core learning engine to maximize accuracy.
    """
    def __init__(self, all_input_attributes, target_attribute, model_params=None):
        self.all_input_attributes = all_input_attributes
        self.target_attribute = target_attribute
        
        if model_params:
            self.model_params = model_params
        else:
            # Default powerful hyperparameters for XGBoost
            self.model_params = {
                'objective': 'multi:softmax', # For multi-class classification
                'num_class': 7,               # We have 7 target classes
                'n_estimators': 200,          # Number of boosting rounds
                'max_depth': 8,               # Max depth of each tree
                'learning_rate': 0.05,        # Step size shrinkage
                'subsample': 0.8,             # Fraction of samples used for fitting each tree
                'colsample_bytree': 0.8,      # Fraction of features used for fitting each tree
                'use_label_encoder': False,   # Suppress a deprecation warning
                'eval_metric': 'mlogloss'     # Logarithmic loss metric for evaluation
            }
        
        self.model = None

    def fit(self, data):
        """Fits the XGBoost model on the data."""
        X_train = data[self.all_input_attributes]
        y_train = data[self.target_attribute]

        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(X_train, y_train, verbose=False) # verbose=False keeps the log clean

    def predict(self, data):
        """Makes a prediction using the trained XGBoost model."""
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
        Processes the data stream by rebuilding the powerful XGBoost model on each window.
        """
        num_records = len(data_stream)
        current_pos = 0

        while current_pos + window_size + step_size <= num_records:
            start_time = time.time()

            train_data = data_stream.iloc[current_pos : current_pos + window_size]
            validation_data = data_stream.iloc[current_pos + window_size : current_pos + window_size + step_size]

            if train_data.empty or validation_data.empty:
                break
            
            print(f"  Building XGBoost model for window [{current_pos}-{current_pos + window_size}]...")
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
