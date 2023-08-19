import torch
from torch.utils.data import dataloader, random_split

from transformer import embedding, tokenizer


class TextCorpus(dataloader.Dataset):
    def __init__(self, decoder: embedding.WordEmbedding, corpus: str | list, context_size: int):
        super(TextCorpus, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(corpus, str):
            tokenized = tokenizer.tokenize(corpus)
        else:
            tokenized = corpus
        # Decoder is a WordEmbedding object which has already been given a dictionary
        idxs = []
        out_of_vocab = set()
        for w in tokenized:
            new = decoder.word_to_idx(w)
            if new == decoder.num_words - 1:
                out_of_vocab.add(w)
            idxs.append(new)

        self.all_words = torch.tensor(idxs, dtype=torch.int64, device=device)
        self.len = len(self.all_words) - context_size
        self.context_size = context_size

        if out_of_vocab:
            print(f'Warning: Found {len(out_of_vocab)} words not in the vocabulary:\n\t' + ','.join(out_of_vocab))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        features = self.all_words[idx:idx+self.context_size]
        targets = self.all_words[idx+1:idx+self.context_size+1]

        return features, targets


def get_dataloaders(
        dataset: dataloader.Dataset,
        val_fraction: float,
        batch_size: int
) -> dict[str, dataloader.DataLoader]:
    """Splits a dataset into validation and training and returns the two respective DataLoaders"""

    split = random_split(dataset, [val_fraction, 1-val_fraction])

    return {
        'validate': dataloader.DataLoader(split[0], batch_size=batch_size, shuffle=True),
        'train': dataloader.DataLoader(split[1], batch_size=batch_size, shuffle=True),
    }
