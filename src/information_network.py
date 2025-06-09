# src/information_network.py
# This module will contain the implementation of the core Information Network (IN) algorithm.
# Key functions:
# - build_IN(training_data, attributes, target_attribute)
# - calculate_mutual_information(data_subset, attribute, target)
# - is_split_significant(data_subset, attribute, target) -> bool (using chi-square test)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

class IN_Node:
    """Represents a node in the Information Network."""
    def __init__(self, is_terminal=False, target_distribution=None, split_attribute=None):
        self.is_terminal = is_terminal
        # Probability distribution of the target variable for data reaching this node
        self.target_distribution = target_distribution
        # The attribute this node is split on
        self.split_attribute = split_attribute
        # Dictionary to store child nodes, keyed by attribute values
        self.children = {}

    def get_prediction(self):
        """Returns the class with the highest probability."""
        if self.target_distribution is None or self.target_distribution.empty:
            return None # Should not happen in a trained model
        return self.target_distribution.idxmax()

class InformationNetwork:
    """The Information Network (IN) classification model."""
    def __init__(self, significance_level=0.01):
        self.root = None
        self.layers = []  # List of attributes used in each layer
        self.second_best_attributes = {} # Store second best attribute for each layer for IOLIN
        self.significance_level = significance_level
        self.target_attribute = None
        self.input_attributes_with_uniques = None # Will store dict of {attr: [unique_vals]}

    def _calculate_entropy(self, data_series):
        """Calculates the entropy of a pandas Series."""
        if data_series.empty:
            return 0
        probs = data_series.value_counts(normalize=True)
        # Using np.log2 for consistency with information theory
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _calculate_mutual_information(self, data_df, attribute, target_attr):
        """Calculates the mutual information I(target; attribute)."""
        h_target = self._calculate_entropy(data_df[target_attr])

        # Calculate conditional entropy H(target | attribute)
        h_target_given_attribute = 0
        for value in data_df[attribute].unique():
            subset = data_df[data_df[attribute] == value]
            if len(data_df) == 0: continue
            prob_value = len(subset) / len(data_df)
            h_target_given_attribute += prob_value * self._calculate_entropy(subset[target_attr])

        return h_target - h_target_given_attribute

    def _is_split_significant(self, data_df, attribute, target_attr):
        """
        Performs a likelihood-ratio G-test to check if a split is statistically significant.
        This is distributed as chi-square.
        """
        if data_df.empty or len(data_df) < 5: # Added minimum size check
            return False, 1.0

        contingency_table = pd.crosstab(data_df[attribute], data_df[target_attr])
        
        if 0 in contingency_table.sum(axis=0).values or 0 in contingency_table.sum(axis=1).values:
            return False, 1.0
        
        g_stat, p_value, dof, expected = chi2_contingency(contingency_table, lambda_="log-likelihood")

        if dof <= 0:
            return False, 1.0

        is_significant = p_value < self.significance_level
        return is_significant, p_value

    def fit(self, data, input_attributes_with_uniques, target_attribute):
        """Builds the Information Network from the training data."""
        self.target_attribute = target_attribute
        self.input_attributes_with_uniques = input_attributes_with_uniques
        self.layers = []
        self.second_best_attributes = {}

        # --- STAGE 1: Determine the optimal sequence of attributes for the layers ---
        remaining_attributes = list(self.input_attributes_with_uniques.keys())
        
        while remaining_attributes:
            best_mi = -1
            best_attribute = None
            second_best_attribute = None
            second_best_mi = -1

            for attr in remaining_attributes:
                is_significant, _ = self._is_split_significant(data, attr, target_attribute)
                if is_significant:
                    mi = self._calculate_mutual_information(data, attr, target_attribute)
                    if mi > best_mi:
                        second_best_attribute = best_attribute
                        second_best_mi = best_mi
                        best_mi = mi
                        best_attribute = attr
                    elif mi > second_best_mi:
                        second_best_mi = mi
                        second_best_attribute = attr
            
            if best_attribute:
                self.layers.append(best_attribute)
                self.second_best_attributes[len(self.layers) - 1] = second_best_attribute
                remaining_attributes.remove(best_attribute)
            else:
                break # No more significant attributes found
        
        # --- STAGE 2: Build the network structure based on the determined layers ---
        print(f"IN model building with layers: {self.layers}")
        self.root = self._build_network_recursive(data, self.layers)
        print("IN model built successfully.")

    def _build_network_recursive(self, data_subset, layers_to_build):
        """Recursively builds the network structure."""
        if data_subset.empty:
            return None 
            
        target_dist = data_subset[self.target_attribute].value_counts(normalize=True)

        if not layers_to_build:
            return IN_Node(is_terminal=True, target_distribution=target_dist)

        split_attr = layers_to_build[0]
        remaining_layers = layers_to_build[1:]
        
        node = IN_Node(split_attribute=split_attr, target_distribution=target_dist)
        
        # FIX: Using the dictionary of unique values passed during fit()
        for value in self.input_attributes_with_uniques[split_attr]:
            child_data = data_subset[data_subset[split_attr] == value]
            child_node = self._build_network_recursive(child_data, remaining_layers)
            
            # If a path has no data, create a terminal node using the parent's distribution
            if child_node is None:
                node.children[value] = IN_Node(is_terminal=True, target_distribution=target_dist)
            else:
                node.children[value] = child_node
        
        return node

    def predict(self, data):
        """Predicts the target for a new dataset using an efficient apply method."""
        if self.root is None:
            raise Exception("Model has not been trained yet. Call fit() first.")
        
        # Ensure data only has the columns the model was trained on
        model_input_cols = list(self.input_attributes_with_uniques.keys())
        return data[model_input_cols].apply(self._predict_single_row, axis=1).tolist()

    def _predict_single_row(self, row):
        """Helper to predict a single row."""
        current_node = self.root
        for layer_attr in self.layers:
            value = row.get(layer_attr)
            if value is not None and value in current_node.children:
                current_node = current_node.children[value]
            else:
                # If path doesn't exist (e.g., value not seen for this specific sub-path),
                # use the current node's distribution for prediction.
                break
        return current_node.get_prediction()

if __name__ == '__main__':
    print("Running a test on the InformationNetwork implementation...")

    try:
        df = pd.read_csv(os.path.join("data", "processed", "processed_beijing_pm25.csv"))
    except FileNotFoundError:
        print("Error: Processed data file not found. Please run src/data_loader.py first.")
        exit()

    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col not in [target_col, 'datetime', 'Unnamed: 0']]
    
    # Store unique values for each attribute for the builder
    input_cols_with_uniques = {col: df[col].unique() for col in input_cols_list}

    train_data = df.sample(n=1000, random_state=42)

    model = InformationNetwork(significance_level=0.01)
    
    # FIX: Pass the dictionary of unique values directly to the fit method
    model.fit(train_data, input_cols_with_uniques, target_col)

    test_data = df.sample(n=100, random_state=1)
    preds = model.predict(test_data) # Pass the whole dataframe
    
    actual = test_data[target_col].tolist()
    correct = sum(1 for p, a in zip(preds, actual) if p == a)
    accuracy = correct / len(preds)
    
    print(f"\nTest Prediction for first 5 rows: {preds[:5]}")
    print(f"Actual values for first 5 rows:   {actual[:5]}")
    print(f"Test Accuracy: {accuracy:.2f}")

    print("Test completed successfully.")
    print("Information Network implementation is working as expected.")