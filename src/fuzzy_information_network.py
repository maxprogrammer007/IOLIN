# src/fuzzy_information_network.py
# This is the TUNED version of our innovative Fuzzy Information Network (FIN) model.
# It includes hyperparameters like max_depth to control the node building process.

import pandas as pd
import numpy as np
import os
import sys
from .fuzzy_engine import Fuzzifier

class Fuzzy_IN_Node:
    """Represents a node in our new Fuzzy Information Network."""
    def __init__(self, prediction=None, split_attribute=None):
        self.prediction = prediction
        self.split_attribute = split_attribute
        self.children = {}

class FuzzyInformationNetwork:
    """
    The innovative Fuzzy Information Network (FIN) classification model.
    Builds a decision tree using fuzzy logic to handle data uncertainty.
    """
    def __init__(self, min_samples_leaf=10, min_gain_threshold=1e-4, max_depth=10):
        self.root = None
        self.fuzzifier = Fuzzifier()
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_threshold = min_gain_threshold
        # --- NEW: Add max_depth to control tree growth ---
        self.max_depth = max_depth
        self.target_attribute = None
        self.input_attributes = None

    def _calculate_fuzzy_entropy(self, data):
        """Calculates entropy based on the sum of membership degrees."""
        if data.empty or 'membership' not in data.columns: return 0
        class_sums = data.groupby(self.target_attribute)['membership'].sum()
        total_membership = class_sums.sum()
        if total_membership == 0: return 0
        probs = class_sums / total_membership
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _calculate_fuzzy_information_gain(self, data, attribute):
        """Calculates the information gain using fuzzy entropy."""
        total_entropy = self._calculate_fuzzy_entropy(data)
        fuzzified_data = data.apply(lambda row: self.fuzzifier.fuzzify(row[attribute], attribute), axis=1).apply(pd.Series)
        weighted_entropy = 0
        total_membership = data['membership'].sum()
        for fuzzy_set in fuzzified_data.columns:
            branch_data = data.copy()
            branch_data['membership'] *= fuzzified_data[fuzzy_set].fillna(0)
            branch_total_membership = branch_data['membership'].sum()
            if branch_total_membership > 0:
                prob_branch = branch_total_membership / total_membership
                weighted_entropy += prob_branch * self._calculate_fuzzy_entropy(branch_data)
        return total_entropy - weighted_entropy

    def fit(self, data, input_attributes, target_attribute):
        """Public method to start the recursive fuzzy tree building process."""
        self.target_attribute = target_attribute
        self.input_attributes = input_attributes
        for attr in self.input_attributes:
            self.fuzzifier.define_fuzzy_sets(data, attr)
        data_with_membership = data.copy()
        data_with_membership['membership'] = 1.0
        self.root = self._grow_tree_recursive(data_with_membership, input_attributes, depth=0)

    def _grow_tree_recursive(self, data_subset, remaining_attributes, depth):
        """Recursively grows the fuzzy decision tree."""
        class_sums = data_subset.groupby(self.target_attribute)['membership'].sum()
        prediction = class_sums.idxmax() if not class_sums.empty else None

        # --- UPDATED Base Cases (Stopping Conditions) ---
        if depth >= self.max_depth or data_subset['membership'].sum() < self.min_samples_leaf or not remaining_attributes:
            return Fuzzy_IN_Node(prediction=prediction)

        best_attribute, best_gain = None, self.min_gain_threshold
        for attr in remaining_attributes:
            gain = self._calculate_fuzzy_information_gain(data_subset, attr)
            if gain > best_gain:
                best_gain, best_attribute = gain, attr
        
        if best_attribute is None:
            return Fuzzy_IN_Node(prediction=prediction)

        node = Fuzzy_IN_Node(prediction=prediction, split_attribute=best_attribute)
        new_remaining_attributes = [attr for attr in remaining_attributes if attr != best_attribute]
        
        for f_set in self.fuzzifier.fuzzy_sets[best_attribute]:
            fuzzified_column = data_subset.apply(lambda row: self.fuzzifier.fuzzify(row[best_attribute], best_attribute).get(f_set, 0), axis=1)
            child_data = data_subset.copy()
            child_data['membership'] *= fuzzified_column
            if child_data['membership'].sum() > 1e-6:
                node.children[f_set] = self._grow_tree_recursive(child_data, new_remaining_attributes, depth + 1)
        return node

    def predict(self, data):
        if self.root is None: raise Exception("Model has not been trained yet.")
        return data.apply(self._predict_single_row, axis=1).tolist()

    def _predict_single_row(self, row):
        final_dist = self._traverse_fuzzy(row, self.root, 1.0)
        return max(final_dist, key=final_dist.get) if final_dist else None

    def _traverse_fuzzy(self, row, node, weight):
        if not node or not hasattr(node, 'children') or not node.children:
            return {node.prediction: weight} if node and node.prediction is not None else {}
        
        split_attr = node.split_attribute
        memberships = self.fuzzifier.fuzzify(row[split_attr], split_attr)
        
        aggregated_dist = {}
        activated_branches = 0
        for f_set, child_node in node.children.items():
            membership_degree = memberships.get(f_set, 0)
            if membership_degree > 0:
                activated_branches += 1
                child_dist = self._traverse_fuzzy(row, child_node, weight * membership_degree)
                for class_label, w in child_dist.items():
                    if class_label is not None:
                      aggregated_dist[class_label] = aggregated_dist.get(class_label, 0) + w
                    
        if activated_branches == 0:
            return {node.prediction: weight}
            
        return aggregated_dist
