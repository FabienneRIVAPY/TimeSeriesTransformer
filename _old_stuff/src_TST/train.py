import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple

    

def train(model: nn.Module, 
          device: torch.device,
          train_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.MSELoss,
          scheduler: torch.optim.lr_scheduler) -> float:
    """
    Execute one epoch of training on the model.
    
    Args:
        model: Neural network model to train
        device: Device (GPU/CPU) to run computations on
        train_loader: Batch iterator for training data
        optimizer: Optimization algorithm
        criterion: Loss function (Mean Squared Error)
    
    Returns:
        float: Average loss across all training batches
    """
    model.train()
    total_loss = 0

    for batch_idx, (data, cat, target) in enumerate(train_loader):
        data, cat, target = data.to(device), cat.to(device), target.to(device)
        optimizer.zero_grad()
            
        output = model(data, cat)
        
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        total_loss += loss.item()

        
    return total_loss / len(train_loader)


def evaluate(model: nn.Module,
            device: torch.device,
            test_loader: DataLoader,
            criterion: nn.MSELoss) -> Tuple[float, torch.Tensor]:
    """
    Evaluates a PyTorch model on test data using Mean Squared Error (MSE) loss.
    
    Args:
        model: The PyTorch neural network model to evaluate
        device: The device (CPU/GPU) where computations will be performed
        test_loader: DataLoader containing the test dataset
        criterion: MSE loss function for evaluation
        
    Returns:
        Tuple[float, torch.Tensor]:
            - Average loss across all test batches
            - Array of model predictions
    """
    model.eval()
    total_loss = 0
    predictions = []
    
    with torch.no_grad():
        for data, cat, target in test_loader:
            data, cat, target = data.to(device), cat.to(device), target.to(device)
            output = model(data, cat)
                
            loss = criterion(output, target)
            total_loss += loss.item()
                
            predictions.extend(output.cpu().numpy())
        
    return total_loss / len(test_loader), np.array(predictions)


