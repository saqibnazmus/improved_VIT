import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
import matplotlib.pyplot as plt


# Improved Multi-Head Attention (with Learnable Scaling)
class ImprovedMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(ImprovedMultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into num_heads different pieces
        values = values.view(N, value_len, self.num_heads, self.head_dim)
        keys = keys.view(N, key_len, self.num_heads, self.head_dim)
        query = query.view(N, query_len, self.num_heads, self.head_dim)

        values = values.permute(2, 0, 1, 3)  # (num_heads, N, value_len, head_dim)
        keys = keys.permute(2, 0, 1, 3)  # (num_heads, N, key_len, head_dim)
        query = query.permute(2, 0, 1, 3)  # (num_heads, N, query_len, head_dim)

        energy = torch.einsum("qnhd,knhd->qnk", [query, keys])  # (num_heads, N, query_len, key_len)
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)

        out = torch.einsum("qnk,nkhd->qnhd", [attention, values])  # (num_heads, N, query_len, head_dim)
        out = out.permute(1, 2, 0, 3).contiguous().view(N, query_len, self.num_heads * self.head_dim)
        out = self.fc_out(out)

        out = self.layernorm(out + query)
        return out

# Vision Transformer Block with Residual Connections
class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1):
        super(VisionTransformerBlock, self).__init__()
        self.attention = ImprovedMultiHeadAttention(embed_size, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_size),
            nn.GELU(),
            nn.Linear(ff_hidden_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attention_out = self.attention(x, x, x)
        x = self.layernorm1(attention_out + x)  # Residual Connection
        ff_out = self.ffn(x)
        x = self.layernorm2(ff_out + x)  # Residual Connection
        return x

# Vision Transformer Model with Learnable Positional Encoding
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000, embed_size=768, num_heads=12, num_layers=12, ff_hidden_size=2048, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Patch Embedding Layer
        self.patch_size = 16  # Patch size (16x16)
        self.conv = nn.Conv2d(3, embed_size, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Learnable Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 197, embed_size))  # 197 = 14x14 (patches)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)
        ])

        # Final Classifier
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # Patch Embedding
        x = self.conv(x)  # (batch_size, embed_size, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_size)
        
        # Add Learnable Positional Encoding
        x = x + self.positional_encoding
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classifier head
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.fc(x)

        return x

