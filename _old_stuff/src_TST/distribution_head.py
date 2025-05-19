import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple
import math
import torch.distributions as dist


class DistributionHead(nn.Module):
    """
    Distribution head for probabilistic forecasting.
    
    Args:
        input_dim (int): Dimensionality of input features
        distribution_type (str): Type of distribution ('normal', 'student_t')
        num_parameters (int): Number of parameters needed for the distribution
    """
    def __init__(self, input_dim: int, distribution_type: str = 'normal'):
        super().__init__()
        self.distribution_type = distribution_type
        
        # For normal distribution: mean and log_std
        # For student-t: location, scale, df
        if distribution_type == 'normal':
            self.num_parameters = 2
        elif distribution_type == 'student_t':
            self.num_parameters = 3
            
        self.fc_dist = nn.Linear(input_dim, self.num_parameters * input_dim)
        
    def forward(self, x):
        params = self.fc_dist(x)
        if self.distribution_type == 'normal':
            # Split into mean and log_std
            mean = params[:, :, :params.shape[-1]//2]
            log_std = params[:, :, params.shape[-1]//2:]
            
            # Ensure positive std
            std = torch.exp(torch.clamp(log_std, min=-10))
            
            return dist.Normal(mean, std)
        elif self.distribution_type == 'student_t':
            # Split into location, scale, df
            loc = params[:, :, :params.shape[-1]//3]
            scale = params[:, :, params.shape[-1]//3:params.shape[-1]*2//3]
            df = params[:, :, params.shape[-1]*2//3:] + 3  # Minimum df=3
            
            # Ensure positive scale
            scale = torch.exp(torch.clamp(scale, min=-10))
            
            return dist.StudentT(df=df, loc=loc, scale=scale)