import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple
import math



class TimeSeriesDataset(Dataset):
    """
    A custom PyTorch Dataset class designed for time series forecasting tasks.
    Creates sequences of consecutive values from the input data for sequence-based prediction.
    
    Attributes:
        X (torch.Tensor): Input time series data
        y (torch.Tensor): Target values
        seq_len (int): Length of sequence windows
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx:idx + self.seq_len]
        y = self.y[idx + self.seq_len - 1]  # Predict next value
        return x, y
    

class TimeSeriesDatasetWithCat(Dataset):
    """
    A custom PyTorch Dataset class designed for time series forecasting tasks.
    Creates sequences of consecutive values from the input data for sequence-based prediction.
    
    Attributes:
        X (torch.Tensor): Input time series data
        y (torch.Tensor): Target values
        seq_len (int): Length of sequence windows
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, categorical_features, seq_len: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.cat_features = torch.tensor(categorical_features, dtype=torch.long)
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.X[idx:idx + self.seq_len]
        cat = self.cat_features[idx:idx + self.seq_len]
        y = self.y[idx + self.seq_len - 1]  # Predict next value
        return x, cat, y
