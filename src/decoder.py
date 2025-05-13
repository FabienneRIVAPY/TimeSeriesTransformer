import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math
from src.attention import FAVORPlusAttention

class CustomDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 attention_type: str = 'scaled-dot'):
        super().__init__()
        
        # # Self-attention mechanism
        # self.self_attn = nn.MultiheadAttention(embed_dim=d_model, 
        #                                       num_heads=n_heads,
        #                                       dropout=dropout)
        
        # # Cross-attention mechanism
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, 
        #                                       num_heads=n_heads,
        #                                       dropout=dropout)
        # self.self_attn = CustomAttention(
        #     embed_dim=d_model,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     attention_type=attention_type
        # )

        # self.multihead_attn = CustomAttention(
        #     embed_dim=d_model,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     attention_type=attention_type
        # )
        self.self_attn = FAVORPlusAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout
        )
        self.multihead_attn = FAVORPlusAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout
        )
       

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = getattr(nn.functional, activation)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: Target tensor (sequence_length, batch_size, embed_dim)
            memory: Memory tensor from encoder (sequence_length, batch_size, embed_dim)
            tgt_mask: Target mask (optional)
            memory_mask: Memory mask (optional)
            tgt_key_padding_mask: Key padding mask for target (optional)
            memory_key_padding_mask: Key padding mask for memory (optional)

        Returns:
            torch.Tensor: Output tensor after processing
        """
        # Self-attention mechanism
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout1(self.self_attn(tgt2, tgt2, value=tgt2,
                                               attention_mask=tgt_mask,
                                               head_mask=tgt_key_padding_mask)[0])
        
        # Cross-attention mechanism
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.multihead_attn(tgt2, memory, value=memory,
                                                    attention_mask=memory_mask,
                                                    head_mask=memory_key_padding_mask)[0])
        
        # Feed-forward network
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.feed_forward(tgt2))
        
        return tgt