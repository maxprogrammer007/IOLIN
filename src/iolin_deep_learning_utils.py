# src/iolin_deep_learning_utils.py
# Utilities for preparing data for deep learning models like LSTMs.

import numpy as np
from tensorflow.keras.utils import to_categorical

def create_lstm_sequences(data, target_attribute, n_steps, n_features, n_classes):
    """
    Reshapes a 2D data window into 3D sequences suitable for an LSTM.
    
    Args:
        data (pd.DataFrame): The input data window.
        target_attribute (str): The name of the target column.
        n_steps (int): The number of time steps in each sequence (e.g., 24 hours).
        n_features (int): The number of input features.
        n_classes (int): The number of target classes for one-hot encoding.

    Returns:
        A tuple of (X, y) where X is the 3D features array and y is the one-hot encoded target array.
    """
    X, y = [], []
    for i in range(len(data)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the dataset
        if end_ix > len(data)-1:
            break
        # Gather input and output parts of the pattern
        # The input is a sequence of 'n_steps' length
        # The output is the target at the end of the sequence
        input_cols = [col for col in data.columns if col != target_attribute]
        seq_x = data[input_cols].iloc[i:end_ix].values
        seq_y = data[target_attribute].iloc[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to be [samples, timesteps, n_features]
    X = X.reshape((X.shape[0], n_steps, n_features))
    # One-hot encode the target variable
    y = to_categorical(y, num_classes=n_classes)
    
    return X, y
