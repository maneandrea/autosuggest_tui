import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import block, embedding
from params import hyperparams

class Transformer(nn.Module):
    """Transformer model"""

    N_BLOCKS = hyperparams.N_BLOCKS
    EMBEDDING_DIM = hyperparams.EMBEDDING_DIM

    def __init__(self, context_size: int, initialized_emb: embedding.WordEmbedding):
        super(Transformer, self).__init__()

        model_dim = self.EMBEDDING_DIM
        self.embeddings = initialized_emb
        self.context_size = context_size

        self.w_embedding = initialized_emb.word_embedding(model_dim)
        self.p_embedding = initialized_emb.pos_embedding(model_dim, context_size)

        self.first_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.Sequential(
            *[block.Block(model_dim) for _ in range(self.N_BLOCKS)]
        )
        self.linear = nn.Linear(model_dim, initialized_emb.num_words)

    def forward(self, x: torch.LongTensor | torch.IntTensor, y: torch.LongTensor | torch.IntTensor = None):

        # Inputs x and y are both (batch=b, context=t)
        b, t = x.shape

        w_emb = self.w_embedding(x)
        p_emb = self.p_embedding(t)
        x = self.first_norm(w_emb + p_emb)
        x = self.blocks(x)

        # Here we have (b, t, num_words), the <out_of_vocabulary> token being one of the words
        x = self.linear(x)

        # Remember: F.cross_entropy
        #  1. applies softmax at the input (i.e. it expects un-normalized logits)
        #  2. takes input of shape (batch, num_classes)
        #  3. if the target is of shape (batch) then it contains class indices [0,num_classes)
        #  4. if the target is of shape (batch, num_classes) then it contains probabilities [0,1]
        if y is None:
            return x, None
        else:
            targets = y.view(b * t)
            logits = x.view(b * t, -1)
            return None, F.cross_entropy(logits, targets)

    def generate(self, x: torch.LongTensor | torch.IntTensor, alternatives: int) -> torch.LongTensor | torch.IntTensor:
        """Generates the next token"""
        # Input x is (batch=b, context=t)
        past = x[:,-self.context_size:]

        # Prediction is (b, t=self.context_size, n=num_words)
        prediction, _ = self(past)
        # but we only care about the last one, from which we compute a softmax -> (b,n), which we then flatten
        prediction = F.softmax(prediction[:, -1, :], dim=-1)
        prediction = torch.flatten(prediction)
        # we extract from this using the multinomial which turns (b*n) -> (a=alternatives)
        next_x = torch.multinomial(prediction, num_samples=alternatives, replacement=False)
        # finally we return the alternatives together with the respective logits
        return next_x, prediction[next_x]
