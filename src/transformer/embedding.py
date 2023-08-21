import math
import re
import torch
import torch.nn as nn

from transformer import tokenizer


END_WORD_REGEX = r'[\s.,;:!?]+$'

class WordEmbedding:
    """A class for initializing a word embedding vocabulary"""

    def __init__(self, stream: str | list):
        if isinstance(stream, str):
            # From the first time (text document)
            self.tokens = list(tokenizer.tokenize(stream))
        else:
            # From an already tokenized file (tokenized-<hash>.pickle)
            # or from the model zip file (vocabulary.pickle)
            self.tokens = list(stream)

        # Note that the empty string is treated as an out-of-vocabulary token
        self.decode = sorted(list(set(self.tokens))) + ['']
        self.vocabulary = {word: i for i, word in enumerate(self.decode)}
        self.num_words = len(self.decode)
        self.end_word_tokens = [
            self.vocabulary[w] for w in self.decode if re.match(END_WORD_REGEX, w)
        ]

    def word_to_idx(self, word: str) -> int:
        """Get the index of a word"""

        # The index for an out-of-vocabulary word is num_words - 1
        # (i.e. <out-of-vocabulary> is a token appended at the end)
        return self.vocabulary.get(word, self.num_words - 1)

    def idx_to_word(self, idx: int) -> str:
        """Get the word from the index"""
        try:
            return self.decode[idx]
        except IndexError:
            return '<unknown>'

    def word_embedding(self, embedding_dim: int) -> nn.Module:
        """Returns an embedding model for words"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return nn.Embedding(self.num_words, embedding_dim).to(device)

    @staticmethod
    def pos_embedding(context_size: int, embedding_dim: int) -> nn.Module:
        """Returns an embedding model for their position"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return PositionalEmbedding(context_size, embedding_dim).to(device)


class PositionalEmbedding(nn.Module):
    """Class for fixed (not learned) positional embedding"""

    def __init__(self, embedding_dim: int, context_size: int):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(PositionalEmbedding, self).__init__()

        position = torch.arange(context_size, device=device)
        freq = torch.arange(0, embedding_dim, 2, device=device)
        freq = torch.exp(freq * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(context_size, embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(torch.outer(position, freq))
        pe[:, 1::2] = torch.cos(torch.outer(position, freq))

        self.register_buffer('pe', pe)

    def forward(self, length: int) -> torch.Tensor:
        """Returns a positional embedding for a given context length"""
        return self.pe[:length]
