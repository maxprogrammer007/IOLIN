# src/fuzzy_preprocessor.py
# This module creates a preprocessor that transforms crisp numerical data
# into a set of fuzzy features, which can then be fed into a powerful
# machine learning model like XGBoost.

import pandas as pd
from .fuzzy_engine import Fuzzifier

class FuzzyPreprocessor:
    """
    Transforms a DataFrame with numerical columns into a new DataFrame
    with fuzzy features, ready for high-performance models.
    """
    def __init__(self, continuous_cols):
        self.fuzzifier = Fuzzifier()
        self.continuous_cols = continuous_cols
        self.output_features = None

    def fit(self, data):
        """
        Learns the fuzzy set boundaries from the training data.
        """
        for col in self.continuous_cols:
            self.fuzzifier.define_fuzzy_sets(data, col)
        
        # Determine the final list of output feature names
        self.output_features = []
        for col, sets in self.fuzzifier.fuzzy_sets.items():
            for set_name in sets.keys():
                self.output_features.append(f"{col}_{set_name}")

    def transform(self, data):
        """
        Applies the learned fuzzy transformation to new data.
        """
        if not self.fuzzifier.fuzzy_sets:
            raise Exception("FuzzyPreprocessor has not been fitted yet.")

        fuzzy_df = pd.DataFrame(index=data.index)

        for col in self.continuous_cols:
            fuzzified_data = data[col].apply(
                lambda x: self.fuzzifier.fuzzify(x, col)
            ).apply(pd.Series)

            # Rename columns to be specific (e.g., 'Temperature_Low')
            fuzzified_data = fuzzified_data.rename(
                columns=lambda set_name: f"{col}_{set_name}"
            )
            fuzzy_df = pd.concat([fuzzy_df, fuzzified_data], axis=1)

        # Fill any NaNs that might have been created with 0 (no membership)
        return fuzzy_df.fillna(0)

    def fit_transform(self, data):
        """Convenience method to fit and then transform the data."""
        self.fit(data)
        return self.transform(data)


if __name__ == '__main__':
    # --- Example Usage and Test for the FuzzyPreprocessor ---
    print("Running a test on the FuzzyPreprocessor...")
    
    df = pd.read_csv("data/processed/processed_beijing_pm25_advanced.csv")
    test_data = df.sample(n=10, random_state=42)
    
    # Specify which columns to fuzzify
    cols_to_fuzzify = [
        'Dew_Point', 'Temperature', 'Pressure', 'Wind_Speed',
        'pm2.5_roll_mean_3h', 'pm2.5_roll_std_3h', 'pm2.5_lag_1h'
    ]
    
    preprocessor = FuzzyPreprocessor(continuous_cols=cols_to_fuzzify)
    
    # Fit on the test data and transform it
    transformed_data = preprocessor.fit_transform(test_data)
    
    print("\nOriginal Data Sample:")
    print(test_data[cols_to_fuzzify].head(2))
    
    print("\nTransformed Fuzzy Data Sample:")
    print(transformed_data.head(2))
    
    print(f"\nOriginal number of features: {len(cols_to_fuzzify)}")
    print(f"New number of fuzzy features: {len(transformed_data.columns)}")