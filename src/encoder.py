import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple
import math
from src.attention import FAVORPlusAttention


class CustomEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        attention_type: str = "scaled-dot",
        kernel_size: int = 3,
    ):
        super().__init__()

        # Multi-head attention mechanism
        # self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
        #                                       num_heads=n_heads,
        #                                       dropout=dropout)
        # self.self_attn = CustomAttention(
        #     embed_dim=d_model,
        #     num_heads=n_heads,
        #     dropout=dropout,
        #     attention_type=attention_type
        # )
        self.self_attn = FAVORPlusAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout
        )

        # Temporal convolution
        self.temporal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding="same"
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation function
        self.activation = getattr(nn.functional, activation)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention mechanism
        src2 = self.norm1(src)
        src = src + self.dropout1(
            self.self_attn(src2, src2, src2, attention_mask=src_key_padding_mask)[0]
        )  # self.self_attn(src2, src2, value=src2,
        #                                        attention_mask=src_mask,
        #                                        head_mask=src_key_padding_mask)[0])

        # Temporal convolution
        src2 = self.norm2(src)
        conv_input = src2.permute(0, 2, 1)  # Batch × Features × Sequence
        conv_output = self.temporal_conv(conv_input)
        conv_output = conv_output.permute(
            0, 2, 1
        )  # Back to Batch × Sequence × Features
        src = src + self.dropout2(conv_output)

        # Feed-forward network
        src2 = self.norm3(src)
        src = src + self.dropout3(self.feed_forward(src2))

        return src
