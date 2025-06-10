# src/iolin_sk.py
# The high-performance version of our ensemble model.
# It uses the IOLIN framework but replaces the custom IN tree
# with a robust, industry-standard DecisionTreeClassifier from scikit-learn.

import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode as majority_vote

class IOLIN_SK_Forest:
    """
    A high-accuracy online random forest that uses scikit-learn's
    DecisionTreeClassifier as its core learning engine. This version includes
    hyperparameter tuning to improve accuracy.
    """
    def __init__(self, all_input_attributes, target_attribute, n_estimators=50, max_features='sqrt', max_depth=10, min_samples_leaf=5):
        self.all_input_attributes = all_input_attributes
        self.target_attribute = target_attribute
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.forest = [] # This will hold the scikit-learn model instances
        self.feature_subsets = [] # Store the features used for each tree
        
        # --- IMPROVEMENT: Store hyperparameters for the Decision Tree ---
        self.tree_params = {
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': 'balanced' # Handles class imbalance
        }


    def _get_feature_subset(self):
        """Selects a random subset of features for an individual tree."""
        if self.max_features == 'sqrt':
            num_features = int(np.sqrt(len(self.all_input_attributes)))
        else: # Default to all
            num_features = len(self.all_input_attributes)
        
        return np.random.choice(self.all_input_attributes, num_features, replace=False).tolist()

    def _fit_forest(self, data):
        """Fits the entire forest of DecisionTree models on the data."""
        new_forest = []
        new_feature_subsets = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_sample = data.sample(n=len(data), replace=True)
            
            # Feature subsampling
            feature_subset = self._get_feature_subset()
            
            X_train = bootstrap_sample[feature_subset]
            y_train = bootstrap_sample[self.target_attribute]

            # --- IMPROVEMENT: Train a scikit-learn Decision Tree with optimized parameters ---
            tree = DecisionTreeClassifier(**self.tree_params)
            tree.fit(X_train, y_train)
            
            new_forest.append(tree)
            new_feature_subsets.append(feature_subset)
            
        self.forest = new_forest
        self.feature_subsets = new_feature_subsets

    def predict(self, data):
        """Makes a prediction using majority vote from all trees."""
        if not self.forest:
            return [None] * len(data)

        # Get predictions from each tree using its specific feature subset
        predictions = []
        for tree, features in zip(self.forest, self.feature_subsets):
            # Ensure data has the correct columns for this tree
            X_test = data[features]
            predictions.append(tree.predict(X_test))
        
        predictions_array = np.array(predictions)
        
        # Majority vote
        majority_result, _ = majority_vote(predictions_array, axis=0, keepdims=False)
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
        Processes the data stream by rebuilding the forest on each window.
        """
        num_records = len(data_stream)
        current_pos = 0

        while current_pos + window_size + step_size <= num_records:
            start_time = time.time()

            train_data = data_stream.iloc[current_pos : current_pos + window_size]
            validation_data = data_stream.iloc[current_pos + window_size : current_pos + window_size + step_size]

            if train_data.empty or validation_data.empty:
                break
            
            print(f"  Building scikit-learn forest for window [{current_pos}-{current_pos + window_size}]...")
            self._fit_forest(train_data)
            
            error = self._calculate_error_rate(validation_data)
            end_time = time.time()
            processing_time = end_time - start_time

            yield {
                'window_start': current_pos,
                'error_rate': error,
                'processing_time_s': processing_time,
            }

            current_pos += step_size
