import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from performer_pytorch import FastAttention

class FAVORPlusAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 nb_features: Optional[int] = None,
                 causal: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize FAVOR+ attention
        self.attention_fn = FastAttention(
            dim_heads=self.head_dim,
            nb_features=nb_features or int(math.sqrt(self.embed_dim)),
            causal=causal
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output linear layer
        self.out = nn.Linear(embed_dim, embed_dim)

    def _reset_parameters(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (batch_size, sequence_length, embed_dim)
            key: Key tensor (batch_size, sequence_length, embed_dim)
            value: Value tensor (batch_size, sequence_length, embed_dim)
            attention_mask: Boolean mask (batch_size, sequence_length)
            head_mask: Mask for attention heads (num_heads,)
        Returns:
            Tuple containing:
            - attention_output: Output tensor (batch_size, sequence_length, embed_dim)
            - attention_weights: Attention weights (batch_size, num_heads, sequence_length, sequence_length)
        """
        batch_size = query.size(0)
        seq_length = query.size(1)

        # Reshape for multi-head attention
        q = query.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = key.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = value.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores using FAVOR+
        attention_output = self.attention_fn(q, k, v)
        
        # Transpose and reshape back
        context = attention_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        
        # Final linear layer
        attention_output = self.out(context)
        
        # Compute attention weights for compatibility
        attention_weights = torch.zeros(batch_size, self.num_heads, seq_length, seq_length, 
                                      device=query.device, dtype=torch.float32)
        
        return attention_output, attention_weights

class CustomAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 attention_type: str = 'scaled-dot',
                 scoring_function: str = 'dot-product'):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, and Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scoring functions
        self.scoring_function = scoring_function.lower()
        if self.scoring_function == 'dot-product':
            self.scale_factor = math.sqrt(self.head_dim)
        elif self.scoring_function == 'cosine':
            self.scale_factor = 1.0
            
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output linear layer
        self.out = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (batch_size, sequence_length, embed_dim)
            key: Key tensor (batch_size, sequence_length, embed_dim)
            value: Value tensor (batch_size, sequence_length, embed_dim)
            attention_mask: Boolean mask (batch_size, sequence_length)
            head_mask: Mask for attention heads (num_heads,)
            
        Returns:
            Tuple containing:
            - attention_output: Output tensor (batch_size, sequence_length, embed_dim)
            - attention_weights: Attention weights (batch_size, num_heads, sequence_length, sequence_length)
        """
        # Get batch size and sequence length
        batch_size = query.size(0)
        seq_length = query.size(1)
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        if self.scoring_function == 'dot-product':
            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        elif self.scoring_function == 'cosine':
            q_norm = q / q.norm(dim=-1, keepdim=True)
            k_norm = k / k.norm(dim=-1, keepdim=True)
            attention_scores = torch.matmul(q_norm, k_norm.transpose(-1, -2))
            
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask[:, None, :, :] * -10000.0
            
        # Compute attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply head mask if provided
        if head_mask is not None:
            attention_weights = attention_weights * head_mask[:, :, None, None]
            
        # Compute context vector
        context = torch.matmul(attention_weights, v)
        
        # Transpose and reshape back
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        
        # Final linear layer
        attention_output = self.out(context)
        
        return attention_output, attention_weights