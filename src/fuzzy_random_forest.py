# src/fuzzy_random_forest.py
# The ultimate improvisation: an ensemble of Fuzzy Information Networks.

import pandas as pd
import numpy as np
from scipy.stats import mode as majority_vote
from .fuzzy_information_network import FuzzyInformationNetwork
import time
class FuzzyRandomForest:
    """
    An ensemble of Fuzzy Information Network models, using the principles
    of Random Forests to achieve high accuracy and robustness.
    """
    def __init__(self, n_estimators=10, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.forest = []
        self.feature_subsets = []

    def _get_feature_subset(self, all_features):
        """Selects a random subset of features for an individual tree."""
        if self.max_features == 'sqrt':
            num_features = int(np.sqrt(len(all_features)))
        else:
            num_features = len(all_features)
        
        return np.random.choice(all_features, num_features, replace=False).tolist()

    def fit(self, data, input_attributes, target_attribute):
        """Fits the entire forest of Fuzzy Information Network models."""
        self.forest = []
        self.feature_subsets = []
        for i in range(self.n_estimators):
            print(f"    Building tree {i+1}/{self.n_estimators}...")
            # Bootstrap sampling
            bootstrap_sample = data.sample(n=len(data), replace=True)
            
            # Feature subsampling
            feature_subset = self._get_feature_subset(input_attributes)
            
            # Train a new Fuzzy IN model
            tree = FuzzyInformationNetwork()
            tree.fit(bootstrap_sample, feature_subset, target_attribute)
            
            self.forest.append(tree)
            self.feature_subsets.append(feature_subset)

    def predict(self, data):
        """Makes a prediction using majority vote from all fuzzy trees."""
        if not self.forest:
            return [None] * len(data)

        predictions = np.array([tree.predict(data) for tree in self.forest])
        
        # Majority vote
        majority_result, _ = majority_vote(predictions, axis=0, keepdims=False)
        return majority_result.tolist()
        
    def _calculate_error_rate(self, data):
        """Calculates the classification error rate for the fuzzy forest."""
        if not self.forest or data.empty:
            return 1.0
        
        predictions = self.predict(data)
        actuals = data.iloc[0:len(predictions)]['Target'].tolist() # Align actuals with predictions
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return 1.0 - (correct / len(predictions)) if len(predictions) > 0 else 1.0

    def process_data_stream(self, data_stream, input_attributes, target_attribute, window_size, step_size):
        """
        Processes the data stream by rebuilding the fuzzy forest on each window.
        """
        num_records = len(data_stream)
        current_pos = 0

        while current_pos + window_size + step_size <= num_records:
            start_time = time.time()

            train_data = data_stream.iloc[current_pos : current_pos + window_size]
            validation_data = data_stream.iloc[current_pos + window_size : current_pos + window_size + step_size]

            if train_data.empty or validation_data.empty:
                break
            
            print(f"\n--- Building Fuzzy Forest for window starting at index {current_pos} ---")
            self.fit(train_data, input_attributes, target_attribute)
            
            error = self._calculate_error_rate(validation_data)
            end_time = time.time()

            yield {
                'window_start': current_pos,
                'error_rate': error,
                'processing_time_s': end_time - start_time,
            }

            current_pos += step_size
