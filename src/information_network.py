# src/information_network.py
# This is the IMPROVED version of the Information Network (IN) algorithm.
# It evolves the model from an "oblivious" tree to a standard, more powerful
# greedy decision tree, directly addressing the professor's feedback.

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

class IN_Node:
    """Represents a node in the Information Network."""
    def __init__(self, is_terminal=False, target_distribution=None, prediction=None, split_attribute=None):
        self.is_terminal = is_terminal
        self.target_distribution = target_distribution
        self.prediction = prediction # Store the majority class prediction
        self.split_attribute = split_attribute
        self.children = {}

class InformationNetwork:
    """
    The IMPROVED Information Network (IN) classification model.
    Builds a greedy decision tree based on mutual information gain.
    """
    def __init__(self, significance_level=0.05, min_samples_leaf=10):
        self.root = None
        self.significance_level = significance_level
        self.min_samples_leaf = min_samples_leaf # Stop splitting if a node has too few samples
        self.target_attribute = None

    def _calculate_entropy(self, data_series):
        if data_series.empty: return 0
        probs = data_series.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _calculate_conditional_entropy(self, data_df, attribute, target_attr):
        conditional_entropy = 0
        for value in data_df[attribute].unique():
            subset = data_df[data_df[attribute] == value]
            prob_value = len(subset) / len(data_df)
            conditional_entropy += prob_value * self._calculate_entropy(subset[target_attr])
        return conditional_entropy
    
    def _calculate_information_gain(self, data_df, attribute, target_attr):
        h_target = self._calculate_entropy(data_df[target_attr])
        h_target_given_attribute = self._calculate_conditional_entropy(data_df, attribute, target_attr)
        return h_target - h_target_given_attribute

    def _is_split_significant(self, data_df, attribute, target_attr):
        if len(data_df) < self.min_samples_leaf * 2: return False
        contingency_table = pd.crosstab(data_df[attribute], data_df[target_attr])
        if 0 in contingency_table.sum(axis=0).values or 0 in contingency_table.sum(axis=1).values: return False
        g_stat, p_value, dof, expected = chi2_contingency(contingency_table, lambda_="log-likelihood")
        if dof <= 0: return False
        return p_value < self.significance_level

    def fit(self, data, input_attributes, target_attribute):
        """Public method to start the recursive tree building process."""
        self.target_attribute = target_attribute
        self.root = self._grow_tree_recursive(data, input_attributes)
        print("Greedy Information Network model built successfully.")

    def _grow_tree_recursive(self, data_subset, remaining_attributes):
        """Recursively grows the decision tree node by node."""
        
        # --- Base Cases (Stopping Conditions) ---
        # 1. If the node is pure (all samples have the same target)
        if len(data_subset[self.target_attribute].unique()) == 1:
            dist = data_subset[self.target_attribute].value_counts(normalize=True)
            return IN_Node(is_terminal=True, target_distribution=dist, prediction=dist.idxmax())

        # 2. If there are no more attributes to split on
        if not remaining_attributes:
            dist = data_subset[self.target_attribute].value_counts(normalize=True)
            return IN_Node(is_terminal=True, target_distribution=dist, prediction=dist.idxmax())

        # --- Find the Best Split for the Current Node ---
        best_attribute = None
        best_gain = -1

        for attr in remaining_attributes:
            if self._is_split_significant(data_subset, attr, self.target_attribute):
                gain = self._calculate_information_gain(data_subset, attr, self.target_attribute)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attr
        
        # 3. If no significant split was found
        if best_attribute is None:
            dist = data_subset[self.target_attribute].value_counts(normalize=True)
            return IN_Node(is_terminal=True, target_distribution=dist, prediction=dist.idxmax())

        # --- Recursive Step: Split the node ---
        node_dist = data_subset[self.target_attribute].value_counts(normalize=True)
        node = IN_Node(split_attribute=best_attribute, target_distribution=node_dist, prediction=node_dist.idxmax())
        
        new_remaining_attributes = [attr for attr in remaining_attributes if attr != best_attribute]
        
        for value in data_subset[best_attribute].unique():
            child_data = data_subset[data_subset[best_attribute] == value]
            node.children[value] = self._grow_tree_recursive(child_data, new_remaining_attributes)

        return node

    def predict(self, data):
        """Predicts the target for new data by traversing the tree."""
        if self.root is None: raise Exception("Model has not been trained yet.")
        return data.apply(self._predict_single_row, axis=1).tolist()

    def _predict_single_row(self, row):
        """Helper to predict a single row."""
        current_node = self.root
        while not current_node.is_terminal:
            value = row.get(current_node.split_attribute)
            if value is not None and value in current_node.children:
                current_node = current_node.children[value]
            else:
                # If path doesn't exist, use the prediction of the current node
                break
        return current_node.prediction

if __name__ == '__main__':
    print("Running a test on the IMPROVED Greedy InformationNetwork...")
    df = pd.read_csv(os.path.join("data", "processed", "processed_beijing_pm25_advanced.csv"))
    target_col = 'Target'
    input_cols = [col for col in df.columns if col not in [target_col, 'datetime', 'Unnamed: 0']]

    train_data = df.sample(n=3000, random_state=42)
    test_data = df.drop(train_data.index).sample(n=1000, random_state=1)

    model = InformationNetwork()
    model.fit(train_data, input_cols, target_col)

    preds = model.predict(test_data)
    actual = test_data[target_col].tolist()
    accuracy = sum(1 for p, a in zip(preds, actual) if p == a) / len(preds)
    
    print(f"\nTest Accuracy of Improved Greedy IN Model: {accuracy:.2%}")
