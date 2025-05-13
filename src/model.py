import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple
import math
from src.encoder import CustomEncoderLayer
from src.decoder import CustomDecoderLayer
from src.positional_encoding import PositionalEncoding
from src.attention import FAVORPlusAttention
from src.distribution_head import DistributionHead


class TransformerModel(nn.Module):
    """
    A PyTorch implementation of a Transformer model for sequence processing.
    This model extends the traditional Transformer architecture with support for both numerical 
    and categorical inputs, incorporating probabilistic forecasting capabilities through a 
    distribution head component.

    Args:
        input_dim (int): Dimensionality of numerical input features (default: 1)
        hidden_dim (int): Dimensionality of hidden representations (default: 128)
        output_dim (int): Dimensionality of output features (default: 1)
        num_categories (int): Number of unique categories for categorical features (default: 1)
        embed_dim (int): Dimensionality of categorical feature embeddings (default: 1)
        n_heads (int): Number of attention heads in transformer layers (default: 8)
        dropout (float): Probability of dropout for regularization (default: 0.1)
        distribution_type (str): Type of probability distribution for output (default: 'normal')
        encoder_layers (int): Number of encoder layers in the transformer stack (default: 2)
        decoder_layers (int): Number of decoder layers in the transformer stack (default: 2)

    Attributes:
        linear (nn.Linear): Linear transformation layer for numerical inputs
        embedding (nn.Embedding): Embedding layer for categorical features
        final_linear (nn.Linear): Linear layer combining numerical and categorical features
        positional_encoding (PositionalEncoding): Temporal embedding layer
        encoder (ModuleList): Stack of encoder layers
        distribution_head (DistributionHead): Probabilistic forecast head
        decoder (ModuleList): Stack of decoder layers
        fc (nn.Linear): Final output linear layer

    Returns:
        torch.Tensor: Output tensor containing predicted values
    """
    def __init__(self, 
                 input_dim: int = 1,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 num_categories: int = 1, 
                 embed_dim: int = 1,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 distribution_type: str = 'student_t',
                 encoder_layers = 1,
                 decoder_layers = 1):
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)

        # Embedding layer for categorical features
        self.embedding = nn.Embedding(num_categories, embed_dim)
        self.final_linear = nn.Linear(hidden_dim + embed_dim, hidden_dim*2)
        # Positional encoding (temporal embedding)
        self.positional_encoding = PositionalEncoding(hidden_dim*2)

        # Encoder
        self.encoder_layer = CustomEncoderLayer(
        d_model=hidden_dim*2,  # Using the concatenated dimension
        n_heads=n_heads,
        dim_feedforward=hidden_dim*4,  # Double the embedding dimension
        dropout=dropout
        )
        self.encoder = nn.ModuleList([self.encoder_layer for _ in range(encoder_layers)])

        # Distribution head for probabilistic forecasts
        self.distribution_head = DistributionHead(hidden_dim*2, distribution_type)

        # Decoder
        self.decoder_layer = CustomDecoderLayer(
            d_model=hidden_dim*2,
            n_heads=n_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout
        )
        self.decoder = nn.ModuleList([self.decoder_layer for _ in range(decoder_layers)])

        # Output layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, src: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        # Embedding and positional encoding
        src = self.linear(src)
        cat = self.embedding(cat)
        combined = torch.cat((src, cat), dim=2)
        src = self.final_linear(combined)
        src = self.positional_encoding(src)

        # Encoder
        encoder_output = src
        for layer in self.encoder:
            encoder_output = layer(encoder_output)

        # Generate distribution from encoder output
        #distribution = self.distribution_head(encoder_output)

        # Decoder
        decoder_output = encoder_output
        for layer in self.decoder:
            decoder_output = layer(encoder_output,encoder_output)#distribution.mean, encoder_output)
        
        # Output
        output = self.fc(decoder_output[:, -1, :])
        return output



class simpleTransformerModel(nn.Module):
    """
    First very simple draft.
    """
    def __init__(self, 
                 input_dim: int = 1,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 num_categories: int = 1, 
                 embed_dim: int = 1,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        # switch for embedding
        self.linear = nn.Linear(input_dim, hidden_dim)
        # Embedding layer for categorical features
        self.embedding = nn.Embedding(num_categories, embed_dim)
        # Final linear layer after concatenation
        self.final_linear = nn.Linear(hidden_dim + embed_dim, hidden_dim*2)
        
        self.positional_encoding = PositionalEncoding(hidden_dim*2)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim*2,
            nhead=n_heads,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim*2,
            nhead=n_heads,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, src: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        # Embedding and positional encoding
        src = self.linear(src)

        # switch for embedding
        cat = self.embedding(cat)
        # Concatenate while preserving sequence length
        combined = torch.cat((src, cat), dim=2)
        src = self.final_linear(combined)
        src = self.positional_encoding(src)
        
        # Encoder
        encoder_output = self.encoder(src)
        
        # Decoder
        decoder_output = self.decoder(encoder_output, encoder_output)
        
        # Output
        # Take last sequence value for prediction
        output = self.fc(decoder_output[:, -1, :])
        return output
