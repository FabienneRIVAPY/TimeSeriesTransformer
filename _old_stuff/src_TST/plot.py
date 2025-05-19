
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple



def plot_predictions(original_data: np.ndarray,
                    predictions: np.ndarray,
                    seq_len: int):
    """
    Visualizes time series forecasting results by plotting actual values against predictions.
    
    Args:
        original_data (np.ndarray): Historical time series data array
        predictions (np.ndarray): Array of predicted values
        seq_len (int): Sequence length used for prediction model
        
    Returns:
        None: Displays a matplotlib figure showing actual vs predicted values
    """
    bla = np.mean(predictions)
    bla2 = np.mean(original_data[-len(predictions):])
    plt.figure(figsize=(12, 6))
    plt.plot(predictions)#, label='Predictions')
    plt.plot(original_data[-len(predictions):], label='Actual',color='black',lw=3)
    plt.legend()
    plt.title(f'Time Series Forecasting Results\nSequence Length: {seq_len}\n{bla,bla2}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()