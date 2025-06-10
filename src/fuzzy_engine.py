# src/fuzzy_engine.py
# This module contains the core components for fuzzifying continuous data,
# which is the first step in building our Fuzzy Information Network (FIN).

import numpy as np

class Fuzzifier:
    """
    A class to handle the fuzzification of continuous attributes.
    It defines fuzzy sets and calculates membership degrees.
    """
    def __init__(self):
        # This dictionary will store the parameters for the fuzzy sets of each attribute.
        # Format: {'attribute_name': {'set_name': [a, b, c], ...}}
        # [a, b, c] are the points of a triangular membership function.
        self.fuzzy_sets = {}

    def _triangular_membership(self, x, a, b, c):
        """
        Calculates the membership degree for a value x in a triangular fuzzy set.
        a: left foot of the triangle (membership = 0)
        b: peak of the triangle (membership = 1)
        c: right foot of the triangle (membership = 0)
        """
        # This check handles the division by zero for vertical lines in the triangle
        term1 = (x - a) / (b - a) if b - a != 0 else 1e9
        term2 = (c - x) / (c - b) if c - b != 0 else 1e9
        return max(0, min(term1, term2))

    def define_fuzzy_sets(self, data, attribute, num_sets=3):
        """
        Automatically defines fuzzy sets for an attribute based on its data distribution.
        This version creates properly overlapping triangles.
        """
        if attribute in self.fuzzy_sets:
            return # Sets are already defined

        min_val = data[attribute].min()
        max_val = data[attribute].max()
        
        # FIX: Define points for overlapping triangles.
        # This ensures that any value has a membership in at least one set.
        q1 = data[attribute].quantile(0.25)
        median = data[attribute].quantile(0.50)
        q3 = data[attribute].quantile(0.75)
        
        self.fuzzy_sets[attribute] = {
            'Low':    [min_val, min_val, median],
            'Medium': [q1, median, q3],
            'High':   [median, max_val, max_val]
        }
        # A more robust 5-point definition for 3 sets:
        self.fuzzy_sets[attribute] = {
            'Low':    [min_val, q1, median],
            'Medium': [q1, median, q3],
            'High':   [median, q3, max_val]
        }


    def fuzzify(self, value, attribute):
        """
        Takes a crisp value and an attribute, and returns a dictionary of its
        membership degrees in each of that attribute's fuzzy sets.
        """
        if attribute not in self.fuzzy_sets:
            # Should not happen if define_fuzzy_sets is called first
            return {} # Return empty dict if no sets are defined

        memberships = {}
        for set_name, params in self.fuzzy_sets[attribute].items():
            a, b, c = params
            degree = self._triangular_membership(value, a, b, c)
            if degree > 0:
                memberships[set_name] = degree
        
        return memberships

if __name__ == '__main__':
    # --- Example Usage and Test for the Fuzzifier ---
    import pandas as pd
    
    print("Running a test on the Fuzzifier engine...")

    # Create dummy data
    dummy_data = pd.DataFrame({
        'Temperature': np.linspace(0, 40, 100),
        'Pressure': np.linspace(980, 1040, 100)
    })
    
    fuzzifier = Fuzzifier()
    
    # Define fuzzy sets for the 'Temperature' attribute based on the data
    fuzzifier.define_fuzzy_sets(dummy_data, 'Temperature')
    
    # Test the fuzzification of a few crisp values
    test_values = [5, 15, 25, 35]
    print("\n--- Fuzzifying Temperature Values ---")
    for val in test_values:
        fuzzy_degrees = fuzzifier.fuzzify(val, 'Temperature')
        print(f"Crisp Value: {val}°C -> Fuzzy Memberships: {fuzzy_degrees}")

    # Demonstrate for another attribute
    fuzzifier.define_fuzzy_sets(dummy_data, 'Pressure')
    print("\n--- Fuzzifying Pressure Values ---")
    pressure_val = 1015
    fuzzy_degrees = fuzzifier.fuzzify(pressure_val, 'Pressure')
    print(f"Crisp Value: {pressure_val} hPa -> Fuzzy Memberships: {fuzzy_degrees}")
