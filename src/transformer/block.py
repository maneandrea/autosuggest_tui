import torch
import torch.nn as nn

from transformer import self_attention
from params import hyperparams

DROPOUT = hyperparams.DROPOUT


class Block(nn.Module):
    """Single transformer block"""
    def __init__(self, model_dim: int):
        super(Block, self).__init__()
        self.model_dim = model_dim
        self.mha = self_attention.MultiAttention(model_dim)
        self.ffn = FeedForward(model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x + self.mha(x))
        return self.norm2(y + self.ffn(y))


class FeedForward(nn.Module):
    """Standard feed-forward neural network"""

    HIDDEN_FACTOR = hyperparams.HIDDEN_FACTOR

    def __init__(self, model_dim: int):
        super(FeedForward, self).__init__()

        hidden_dim = model_dim * self.HIDDEN_FACTOR

        self.layers = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
