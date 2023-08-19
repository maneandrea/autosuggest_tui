import sys

import torch

from train import save_load
from params import hyperparams

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Suggester:

    CONTEXT_SIZE = hyperparams.CONTEXT_SIZE
    N_SUGGESTIONS = hyperparams.N_SUGGESTIONS
    MAX_DEPTH = hyperparams.MAX_DEPTH

    def __init__(self, model_file: str):

        # self.emb = embedding.WordEmbedding(text)
        # self.model = transformer.Transformer(self.CONTEXT_SIZE, self.emb).to(DEVICE)
        self.model = save_load.load_model(model_file, context_size=self.CONTEXT_SIZE)
        if self.model is not None:
            self.emb = self.model.embeddings
            self.model.eval()
        else:
            print('Exiting')
            sys.exit(1)

    def make_word(self, token_list):
        """Concatenates tokens to make up a full word"""
        w = ''
        for tok in token_list:
            w += self.emb.idx_to_word(tok.item())

        return w

    def run_model(self, tokens):
        """Turn the input words into an integer tensor and run the transformer on it generating
        N_SUGGESTIONS suggestions by decoding tokens until a whitespace is encountered"""
        x = torch.tensor([self.emb.word_to_idx(w) for w in tokens], dtype=torch.int32, device=DEVICE)
        l_prompt = len(tokens)
        n_gen = self.N_SUGGESTIONS
        beam = n_gen
        depth = 0

        # Queue with tokens being added one by one
        queue = [(1, x)]
        suggestions = []

        # Generate n_gen words with beam search decoding
        while queue:

            depth+=1
            if depth > self.MAX_DEPTH:
                if len(suggestions) == 0:
                    return [('', 1)]
                else:
                    break

            curr_prob, curr_x = queue.pop()
            generated, probs = self.model.generate(curr_x.unsqueeze(0), beam)

            for p, g in sorted(zip(probs, generated), key=lambda pg: pg[0].item(), reverse=True):
                # We allow for the very first token to be a space (new word) or not (continuation of word)
                # if instead a new space is encountered this means the end of a suggestion (end word)
                if curr_prob == 1 or g not in self.emb.end_word_tokens:
                    queue.append((
                        curr_prob * p.item(),
                        torch.cat((curr_x, g.unsqueeze(0)))
                    ))
                else:
                    suggestions.append((
                        curr_prob * p.item(),
                        torch.cat((curr_x, g.unsqueeze(0)))[l_prompt:]
                    ))

                # print(f'sugg (l={l_prompt}):', [(self.make_word(a), p) for p, a in suggestions])
                # print('queue:', [(self.make_word(a), p) for p, a in queue])

            queue = queue[:beam]

        return [(self.make_word(a), p) for p, a in suggestions[:n_gen]]

    def suggest(self, words):
        """Takes a vector of words, pads or truncates to CONTEXT_SIZE and returns the suggested words"""
        if len(words) < self.CONTEXT_SIZE:
            looked = (self.CONTEXT_SIZE - len(words)) * [' '] + words
        else:
            looked = words[-self.CONTEXT_SIZE:]

        return self.run_model(looked)