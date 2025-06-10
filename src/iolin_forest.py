# src/iolin_forest.py
# An improved, high-accuracy model based on the IOLIN-Forest concept.
# It uses an ensemble of InformationNetwork models to make predictions.

import pandas as pd
import numpy as np
import os
import time
from information_network import InformationNetwork
from scipy.stats import mode as majority_vote

class IOLIN_Forest:
    """
    An ensemble version of IOLIN using Random Forest principles to boost accuracy.
    Manages a collection of InformationNetwork models.
    """
    def __init__(self, input_attributes_with_uniques, target_attribute, n_estimators=10, max_features='sqrt', significance_level=0.05):
        self.input_attributes_with_uniques = input_attributes_with_uniques
        self.target_attribute = target_attribute
        self.n_estimators = n_estimators # Number of trees in the forest
        self.max_features = max_features # Number of features for each tree to consider
        self.significance_level = significance_level
        self.forest = [] # This will hold the IN model instances

    def _get_feature_subset(self):
        """Selects a random subset of features for an individual tree."""
        all_features = list(self.input_attributes_with_uniques.keys())
        if self.max_features == 'sqrt':
            num_features = int(np.sqrt(len(all_features)))
        elif isinstance(self.max_features, int):
            num_features = self.max_features
        else: # Default to all features
            num_features = len(all_features)
        
        return np.random.choice(all_features, num_features, replace=False).tolist()

    def _fit_forest(self, data):
        """Fits the entire forest of IN models on the data."""
        new_forest = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling: create a sample of the data with replacement
            bootstrap_sample = data.sample(frac=1, replace=True)
            
            # Feature subsampling: select a random subset of features
            feature_subset_list = self._get_feature_subset()
            feature_subset_dict = {k: self.input_attributes_with_uniques[k] for k in feature_subset_list}

            # Train a new IN model on the bootstrapped data and feature subset
            tree = InformationNetwork(significance_level=self.significance_level)
            tree.fit(bootstrap_sample, feature_subset_dict, self.target_attribute)
            new_forest.append(tree)
        self.forest = new_forest

    def predict(self, data):
        """Makes a prediction using majority vote from all trees in the forest."""
        if not self.forest:
            return [None] * len(data)

        # Get predictions from each tree
        predictions = np.array([tree.predict(data) for tree in self.forest])
        
        # Perform majority vote for each data point (column-wise)
        majority_result, _ = majority_vote(predictions, axis=0, keepdims=False)
        return majority_result.tolist()
        
    def _calculate_error_rate(self, data):
        """Calculates the classification error rate for the entire forest."""
        if not self.forest or data.empty:
            return 1.0
        
        predictions = self.predict(data)
        actuals = data[self.target_attribute].tolist()
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return 1.0 - (correct / len(predictions)) if len(predictions) > 0 else 1.0

    def process_data_stream(self, data_stream, window_size, step_size):
        """
        Processes the data stream. For this high-accuracy model, we will rebuild
        the entire forest on each window to ensure maximum performance.
        A more advanced version could update trees incrementally.
        """
        num_records = len(data_stream)
        current_pos = 0

        while current_pos + window_size + step_size <= num_records:
            start_time = time.time()

            train_data = data_stream.iloc[current_pos : current_pos + window_size]
            validation_data = data_stream.iloc[current_pos + window_size : current_pos + window_size + step_size]

            if train_data.empty or validation_data.empty:
                break
            
            # --- Regenerative Step: Rebuild the entire forest ---
            print(f"  Building forest for window [{current_pos}-{current_pos + window_size}]...")
            self._fit_forest(train_data)
            
            # --- Evaluation ---
            error = self._calculate_error_rate(validation_data)
            end_time = time.time()
            processing_time = end_time - start_time

            yield {
                'window_start': current_pos,
                'error_rate': error,
                'processing_time_s': processing_time,
            }

            current_pos += step_size
