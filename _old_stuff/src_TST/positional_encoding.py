import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple
import math

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for transformer models.
    
    This module adds fixed positional encodings to input embeddings to preserve sequence order information.
    The encoding is based on the formula described in "Attention Is All You Need" paper.
    
    Args:
        d_model (int): Dimensionality of the embedding space
        max_len (int, optional): Maximum sequence length to encode. Defaults to 5000.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (sequence_length, batch_size, d_model)
            
        Returns:
            torch.Tensor: Input tensor with added positional encodings
        """
        x = x + self.pe[:x.size(0), :]
        return x