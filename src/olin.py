# src/olin.py
# This module will implement the regenerative OLIN algorithm.
# It will use a sliding window and call the `build_IN` function from
# information_network.py for every new window of data.

import pandas as pd
import os
import time
# FIX: Changed the import to be relative to the src directory
from information_network import InformationNetwork

class OLIN:
    """
    The regenerative On-line Information Network (OLIN) algorithm.
    This version rebuilds the model for every new window of data.
    """
    def __init__(self, input_attributes_with_uniques, target_attribute, significance_level=0.01):
        self.input_attributes_with_uniques = input_attributes_with_uniques
        self.target_attribute = target_attribute
        self.significance_level = significance_level
        self.model = None

    def process_data_stream(self, data_stream, window_size, step_size):
        """
        Processes the data stream using a sliding window, rebuilding the model each time.

        Args:
            data_stream (pd.DataFrame): The entire dataset to be processed.
            window_size (int): The number of records in each training window.
            step_size (int): The number of records to slide the window forward.

        Yields:
            dict: A dictionary containing performance metrics for each window.
        """
        num_records = len(data_stream)
        current_pos = 0

        while current_pos + window_size + step_size <= num_records:
            start_time = time.time()

            # Define the training and validation windows
            train_window_end = current_pos + window_size
            validation_window_end = train_window_end + step_size

            train_data = data_stream.iloc[current_pos:train_window_end]
            validation_data = data_stream.iloc[train_window_end:validation_window_end]

            if train_data.empty or validation_data.empty:
                break

            # --- Regenerative Step: Build a new model from scratch ---
            self.model = InformationNetwork(significance_level=self.significance_level)
            self.model.fit(train_data, self.input_attributes_with_uniques, self.target_attribute)

            # --- Evaluate the model on the validation set ---
            predictions = self.model.predict(validation_data)
            actuals = validation_data[self.target_attribute].tolist()

            correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
            error_rate = 1.0 - (correct / len(predictions)) if len(predictions) > 0 else 1.0

            end_time = time.time()
            processing_time = end_time - start_time
            classification_rate = len(validation_data) / processing_time if processing_time > 0 else float('inf')

            # Yield the results for this window
            yield {
                'window_start': current_pos,
                'window_end': validation_window_end,
                'error_rate': error_rate,
                'processing_time_s': processing_time,
                'classification_rate_rec_s': classification_rate,
                'layers_built': self.model.layers
            }

            # Slide the window forward
            current_pos += step_size


if __name__ == '__main__':
    # --- Example Usage and Test for OLIN ---
    print("Running a test on the OLIN implementation...")

    try:
        # FIX: Corrected path to work when run from the root directory
        df = pd.read_csv(os.path.join("data", "processed", "processed_beijing_pm25.csv"))
    except FileNotFoundError:
        print("Error: Processed data file not found. Please run src/data_loader.py first.")
        exit()

    # Use a smaller portion of the data for a quick test run
    test_stream_data = df.head(1500)

    target_col = 'Target'
    input_cols_list = [col for col in df.columns if col not in [target_col, 'datetime', 'Unnamed: 0']]
    input_cols_with_uniques = {col: df[col].unique() for col in input_cols_list}

    # OLIN parameters for the test
    # The paper starts with 500 records. We'll use a smaller window for a quick test.
    WINDOW_SIZE = 500
    STEP_SIZE = 100 # This will be our validation set size

    # Initialize OLIN
    olin_processor = OLIN(input_cols_with_uniques, target_col, significance_level=0.05)

    # Process the stream and collect results
    results = []
    print(f"Processing data stream with window_size={WINDOW_SIZE} and step_size={STEP_SIZE}...")
    for result in olin_processor.process_data_stream(test_stream_data, WINDOW_SIZE, STEP_SIZE):
        print(f"  Window [{result['window_start']}-{result['window_end']}]: "
              f"Error Rate={result['error_rate']:.3f}, "
              f"Time={result['processing_time_s']:.2f}s")
        results.append(result)

    if results:
        results_df = pd.DataFrame(results)
        avg_error = results_df['error_rate'].mean()
        avg_time = results_df['processing_time_s'].mean()
        print("\n--- OLIN Test Summary ---")
        print(f"Total windows processed: {len(results_df)}")
        print(f"Average Error Rate: {avg_error:.3f}")
        print(f"Average Processing Time per Window: {avg_time:.3f} seconds")
    else:
        print("No results generated. The test data stream might be too small for the window settings.")
