import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import hyperparams

DROPOUT = hyperparams.DROPOUT


class MultiAttention(nn.Module):
    """A multi-head self attention node"""

    N_HEADS = hyperparams.N_HEADS
    VALUE_DIM = hyperparams.VALUE_DIM

    def __init__(self, model_dim: int):
        super(MultiAttention, self).__init__()
        if model_dim % self.N_HEADS == 0:
            self.head_dim = model_dim // self.N_HEADS
        else:
            raise ValueError(f'model_dim={model_dim} must be divisible by n_heads={self.N_HEADS}')

        if self.VALUE_DIM:
            self.value_dim = self.VALUE_DIM
        else:
            self.value_dim = self.head_dim

        self.heads = nn.ModuleList([
            Attention(model_dim, self.head_dim, self.value_dim) for _ in range(self.N_HEADS)
        ])
        self.linear = nn.Linear(self.value_dim * self.N_HEADS, model_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input is (batch=b, context=t, model_dim=d)
        # Each head outputs (b,t,value_dim=v) and their cat is v*n_heads
        y = [a.forward(x) for a in self.heads]
        return self.dropout(self.linear(torch.cat(y, dim=-1)))


class Attention(nn.Module):
    """A single self attention head"""

    MAX_CONTEXT = hyperparams.CONTEXT_SIZE

    def __init__(self, model_dim: int, head_dim: int, value_dim: int):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(Attention, self).__init__()
        self.head_dim = head_dim
        self.get_key = nn.Linear(model_dim, head_dim)
        self.get_query = nn.Linear(model_dim, head_dim)
        self.get_value = nn.Linear(model_dim, value_dim)
        self.dropout = nn.Dropout(DROPOUT)

        triangular = [[j > i for j in range(self.MAX_CONTEXT)] for i in range(self.MAX_CONTEXT)]
        self.register_buffer('mask', torch.tensor(triangular, dtype=torch.bool, requires_grad=False, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input is (batch=b, context=t, model_dim=d)
        b, t, d = x.shape

        # They become (b,t,h=head_dim)
        query = self.get_query(x)
        key = self.get_key(x)
        value = self.get_value(x)

        # s and t are along the context, b is a batch dimension and h is a head dimension
        scaled_dot_prod = torch.einsum('bsh,bth->bst', query, key) * math.pow(self.head_dim, -0.5)
        # The batch dimension is broadcast into self.mask: (t, t) -> (b, t, t)
        masked = scaled_dot_prod.masked_fill(self.mask[:t, :t], -torch.inf)
        masked = self.dropout(F.softmax(masked, dim=-1))
        # v is the value dimension, which typically is the same as the head dimension
        return torch.einsum('bst,btv->bsv', masked, value)

