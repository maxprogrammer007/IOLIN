# src/fuzzy_random_forest.py
# The TUNED version of our Fuzzy Random Forest. It accepts hyperparameters
# for its base learners to improve performance.

import pandas as pd
import numpy as np
from scipy.stats import mode as majority_vote
from .fuzzy_information_network import FuzzyInformationNetwork

class FuzzyRandomForest:
    """
    An ensemble of Fuzzy Information Network models.
    """
    def __init__(self, n_estimators=10, max_features='sqrt', max_depth=10):
        self.n_estimators = n_estimators
        self.max_features = max_features
        # --- NEW: Store max_depth for the trees ---
        self.max_depth = max_depth
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
            bootstrap_sample = data.sample(n=len(data), replace=True)
            feature_subset = self._get_feature_subset(input_attributes)
            
            # --- NEW: Pass max_depth to the fuzzy tree constructor ---
            tree = FuzzyInformationNetwork(max_depth=self.max_depth)
            tree.fit(bootstrap_sample, feature_subset, target_attribute)
            
            self.forest.append(tree)
            self.feature_subsets.append(feature_subset)

    def predict(self, data):
        """Makes a prediction using majority vote from all fuzzy trees."""
        if not self.forest: return [None] * len(data)
        predictions = np.array([tree.predict(data) for tree in self.forest])
        majority_result, _ = majority_vote(predictions, axis=0, keepdims=False)
        return majority_result.tolist()
        
    def _calculate_error_rate(self, data):
        """Calculates the classification error rate for the fuzzy forest."""
        if not self.forest or data.empty: return 1.0
        predictions = self.predict(data)
        actuals = data.iloc[0:len(predictions)]['Target'].tolist()
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return 1.0 - (correct / len(predictions)) if len(predictions) > 0 else 1.0
