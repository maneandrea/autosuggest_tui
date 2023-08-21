import torch
import os

from transformer import embedding, suggest, transformer
from train import data_loader,save_load
from params import hyperparams

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_ITER = hyperparams.N_ITER
EVAL_ITER = hyperparams.EVAL_ITER


def print_progress(epoch, nepochs, step, tot, result):
    """Prints the progress nicely"""
    ll = len(str(tot))
    el = len(str(nepochs))
    epoch_str = (2*el + 10) * ' ' if epoch is None else ' epoch [' + str(epoch).rjust(el) + f'/{nepochs}]'
    step_str = ' step [' + str(step).rjust(ll) + f'/{tot}]'
    v, t = result['validate'], result['train']
    print(f'{epoch_str}{step_str}\tlosses:\ttrain={t:.3f},\tvalidation={v:.3f}')


def train_model(
        corpus: str,
        corpus_hash: str,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        val_fraction: float,
        save_path: str,
        load_path: str
):
    """Train the model with some text corpus"""

    source = os.path.join(os.path.dirname(__file__), '..', '..', 'data', f'tokenized-{corpus_hash}.pickle')
    source = os.path.realpath(source)
    vocabulary = save_load.load_emb(source)

    if vocabulary:
        emb = embedding.WordEmbedding(vocabulary)
    else:
        emb = embedding.WordEmbedding(corpus)
        # We also save the embedding (actually, just the list of tokens)
        save_load.save_emb(emb, source)

    context_size = suggest.Suggester.CONTEXT_SIZE

    dataset = data_loader.TextCorpus(emb, emb.tokens, context_size)
    loaders = data_loader.get_dataloaders(dataset, val_fraction, batch_size)
    train = loaders['train']
    n_batches = len(train)

    gpt = transformer.Transformer(context_size, emb).to(DEVICE)

    if load_path:
        gpt = save_load.load_model(load_path, context_size)

    @torch.no_grad()
    def model_performance():
        gpt.eval()

        result = {'train': 0, 'validate': 0}
        for split in result.keys():
            k = 0
            for k, (x, y) in enumerate(loaders[split]):
                if k < N_ITER:
                    result[split] += gpt(x, y)[1].item()
                else:
                    break
            result[split] = result[split] / (k + 1)

        gpt.train()
        return result

    optimizer = torch.optim.Adam(gpt.parameters(), learning_rate)

    temp_save_path = ''
    for epoch in range(epochs):
        for i, (features, targets) in enumerate(train):

            # Print the intermediate results
            if i % EVAL_ITER == EVAL_ITER - 1 or i == n_batches-1:
                if i < EVAL_ITER:
                    print_progress(epoch + 1, epochs, i + 1, n_batches, model_performance())
                else:
                    print_progress(None, epochs, i + 1, n_batches, model_performance())

            _, loss = gpt(features, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch < epochs - 1:
            if save_path:
                temp_save_path = os.path.splitext(save_path)
                temp_save_path = temp_save_path[0] + '_ck' + temp_save_path[1]
            else:
                temp_save_path = 'gpt.pt_ck.zip'
            print(f' saving progress to {temp_save_path}')
            save_load.save_model(gpt, temp_save_path, overwrite=True)

    saved = False
    if save_path:
        saved = save_load.save_model(gpt, save_path)

    while not saved:
        save_path = input('Where to save the model? (Leave empty for gpt.pt.zip) ')
        if save_path == '':
            save_path = 'gpt.pt.zip'
        saved = save_load.save_model(gpt, save_path)

    if os.path.isfile(save_path):
        print(f'Saved result to {save_path}')
        if os.path.isfile(temp_save_path):
            os.remove(temp_save_path)
    else:
        print(f'Could not save file, saved result to {temp_save_path}')
        save_load.save_model(gpt, save_path, overwrite=True)
