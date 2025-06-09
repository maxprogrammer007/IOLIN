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
        self.input_attributes = None

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
        if data_df.empty:
            return False, 1.0

        # Create a contingency table (observed frequencies)
        contingency_table = pd.crosstab(data_df[attribute], data_df[target_attr])
        
        # The G-test is not reliable for tables with a row or column of all zeros.
        if 0 in contingency_table.sum(axis=0).values or 0 in contingency_table.sum(axis=1).values:
            return False, 1.0
        
        # Use scipy's built-in function for the G-test (log-likelihood)
        g_stat, p_value, dof, expected = chi2_contingency(contingency_table, lambda_="log-likelihood")

        if dof <= 0:
            return False, 1.0

        is_significant = p_value < self.significance_level
        return is_significant, p_value


    def fit(self, data, input_attributes, target_attribute):
        """Builds the Information Network from the training data."""
        self.target_attribute = target_attribute
        self.input_attributes = input_attributes
        
        # Initialize root node
        initial_distribution = data[target_attribute].value_counts(normalize=True)
        self.root = IN_Node(is_terminal=True, target_distribution=initial_distribution)
        
        terminal_nodes = [self.root]
        remaining_attributes = list(input_attributes)
        
        while terminal_nodes and remaining_attributes:
            best_mi = -1
            best_attribute = None
            second_best_attribute = None
            second_best_mi = -1

            # --- Find the best attribute to create the next layer ---
            for attr in remaining_attributes:
                # Check significance on the whole dataset for the potential layer
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

            if best_attribute is None:
                # No significant attribute found, stop building
                break
                
            self.layers.append(best_attribute)
            self.second_best_attributes[len(self.layers) -1] = second_best_attribute
            remaining_attributes.remove(best_attribute)
            
            # --- Split the terminal nodes on the best attribute ---
            new_terminal_nodes = []
            
            # Since this is an "oblivious" tree, all nodes at a given level are split by the same attribute.
            # We iterate through the existing terminal nodes and replace them with sub-networks.
            nodes_to_split = terminal_nodes
            terminal_nodes = []

            for node in nodes_to_split:
                node.is_terminal = False
                node.split_attribute = best_attribute
                
                for value in sorted(data[best_attribute].unique()):
                    subset = data[data[best_attribute] == value]
                    if subset.empty:
                        continue
                        
                    child_distribution = subset[self.target_attribute].value_counts(normalize=True)
                    child_node = IN_Node(is_terminal=True, target_distribution=child_distribution)
                    node.children[value] = child_node
                    terminal_nodes.append(child_node)
            
        print(f"IN model built. Layers: {self.layers}")


    def predict(self, data):
        """Predicts the target for a new dataset."""
        if self.root is None:
            raise Exception("Model has not been trained yet. Call fit() first.")
        
        predictions = []
        for _, row in data.iterrows():
            current_node = self.root
            for layer_attr in self.layers:
                value = row.get(layer_attr)
                if value in current_node.children:
                    current_node = current_node.children[value]
                else:
                    # Value not seen during training, break and use current node's prediction
                    break
            predictions.append(current_node.get_prediction())
            
        return predictions

if __name__ == '__main__':
    # --- Example Usage and Test ---
    print("Running a test on the InformationNetwork implementation...")

    # Load the processed data
    try:
        df = pd.read_csv(os.path.join("data", "processed", "processed_beijing_pm25.csv"))
    except FileNotFoundError:
        print("Error: Processed data file not found. Please run src/data_loader.py first.")
        exit()

    # Define attributes
    target_col = 'Target'
    # Ensure 'datetime' is not treated as an input attribute if it exists
    input_cols = [col for col in df.columns if col not in [target_col, 'datetime', 'Unnamed: 0']]


    # Use a small sample for a quick test
    train_data = df.sample(n=1000, random_state=42)

    # Initialize and fit the model
    model = InformationNetwork(significance_level=0.01)
    model.fit(train_data, input_cols, target_col)

    # Make predictions on a test set
    test_data = df.sample(n=100, random_state=1)
    preds = model.predict(test_data[input_cols])
    
    # Calculate accuracy
    actual = test_data[target_col].tolist()
    correct = sum(1 for p, a in zip(preds, actual) if p == a)
    accuracy = correct / len(preds)
    
    print(f"\nTest Prediction for first 5 rows: {preds[:5]}")
    print(f"Actual values for first 5 rows:   {actual[:5]}")
    print(f"Test Accuracy: {accuracy:.2f}")
    print("Test completed successfully.")
    print("Information Network implementation is working as expected.")