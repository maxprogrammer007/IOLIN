# src/iolin_lstm.py
# The state-of-the-art implementation using a Keras LSTM model.
# This version includes data scaling for optimal performance.

import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from .iolin_deep_learning_utils import create_lstm_sequences
from sklearn.preprocessing import MinMaxScaler

class IOLIN_LSTM:
    """
    A state-of-the-art online model using a Keras LSTM network.
    Includes data scaling for optimal performance.
    """
    def __init__(self, n_steps, n_features, n_classes, model_params=None):
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.scaler = MinMaxScaler() # Initialize the scaler

        if model_params is None:
            self.model_params = {'epochs': 20, 'batch_size': 64, 'verbose': 0} # More epochs
        else:
            self.model_params = model_params

    def _build_model(self):
        """Builds the Keras LSTM model architecture."""
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(self.n_steps, self.n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def fit(self, data, target_attribute):
        """Prepares data, scales it, and fits the LSTM model."""
        if self.model is None:
            self._build_model()
        
        # --- IMPROVEMENT: Scale the data ---
        input_cols = [col for col in data.columns if col != target_attribute]
        # Fit the scaler ONLY on the training data and transform it
        data_scaled = data.copy()
        data_scaled[input_cols] = self.scaler.fit_transform(data[input_cols])
        
        X_train, y_train = create_lstm_sequences(data_scaled, target_attribute, self.n_steps, self.n_features, self.n_classes)
        
        if X_train.shape[0] == 0:
            print("  Warning: Not enough data in the window to create LSTM sequences.")
            return

        self.model.fit(X_train, y_train, **self.model_params)

    def predict(self, data, target_attribute):
        """Scales new data and makes a prediction."""
        if self.model is None:
            return None

        # --- IMPROVEMENT: Scale the new data using the FITTED scaler ---
        input_cols = [col for col in data.columns if col != target_attribute]
        data_scaled = data.copy()
        data_scaled[input_cols] = self.scaler.transform(data[input_cols])

        X_test, _ = create_lstm_sequences(data_scaled, target_attribute, self.n_steps, self.n_features, self.n_classes)
        if X_test.shape[0] == 0:
            return []

        yhat = self.model.predict(X_test, verbose=0)
        return np.argmax(yhat, axis=1)
        
    def _calculate_error_rate(self, data, target_attribute):
        """Calculates the classification error rate for the model."""
        if self.model is None or len(data) <= self.n_steps:
            return 1.0
        
        predictions = self.predict(data, target_attribute)
        actuals = data[target_attribute].iloc[self.n_steps:].values
        
        if len(predictions) != len(actuals):
            return 1.0

        correct = np.sum(predictions == actuals)
        return 1.0 - (correct / len(actuals)) if len(actuals) > 0 else 1.0

    def process_data_stream(self, data_stream, target_attribute, window_size, step_size):
        """
        Processes the data stream by rebuilding the scaled LSTM model on each window.
        """
        num_records = len(data_stream)
        current_pos = 0

        while current_pos + window_size + step_size <= num_records:
            start_time = time.time()

            train_data = data_stream.iloc[current_pos : current_pos + window_size]
            validation_data = data_stream.iloc[current_pos + window_size : current_pos + window_size + step_size]

            if train_data.empty or validation_data.empty:
                break
            
            print(f"  Building LSTM model for window [{current_pos}-{current_pos + window_size}]...")
            self.fit(train_data, target_attribute)
            
            error = self._calculate_error_rate(validation_data, target_attribute)
            end_time = time.time()
            processing_time = end_time - start_time

            yield {
                'window_start': current_pos,
                'error_rate': error,
                'processing_time_s': processing_time,
            }

            current_pos += step_size
